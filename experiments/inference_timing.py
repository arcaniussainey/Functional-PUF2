"""
inference-time comparison for PUF twins and attack models.

1. **Training cost**: the one-time cost to learn an attack model from CRPs.
2. **Inference cost**: the per-authentication cost to answer fresh challenges.

Only inference cost matters once an attacker has already trained a clone.  This
script trains several attacks once, then times prediction on challenge blocks of
various sizes.  It also records the first-call latency for JAX-backed PUFs so
that JIT compilation overhead is visible instead of being hidden inside steady
state timing.

Compared models
---------------
* True Arbiter PUF: JAX implementation of ``sign(w^T Phi(C))``.
* LRAttack: logistic regression on Phi features.
* RidgeAttack: ridge regression on Phi features.
* Raw DNNAttack: MLP on raw challenge bits.
* PhiDNNAttack: MLP on literature-grounded Phi features.

Output
------
A dictionary containing median latency, per-challenge latency, and first-call
latency by model and block size.  When ``save_path`` is provided, a matplotlib
figure is written with total block latency and microseconds per challenge.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time import perf_counter
from typing import Callable, Dict, Optional, Sequence

import jax
import numpy as np
import matplotlib.pyplot as plt

from Pufs.FunctionalPuf import Arbiter, generate_challenges
from Attack.lr_attack import LRAttack
from Attack.ridge_attack import RidgeAttack
from Attack.dnn_attack import DNNAttack
from Attack.dnn_attack_2 import PhiDNNAttack


def _block(value: object) -> None:
    """Synchronise JAX arrays without imposing a JAX dependency on sklearn paths."""
    if isinstance(value, tuple):
        for item in value:
            _block(item)
    elif hasattr(value, "block_until_ready"):
        value.block_until_ready()  # type: ignore[union-attr]


def _time_call(fn: Callable[[], object], repeats: int, warmups: int) -> Dict[str, float]:
    """
    Time a callable after recording its cold first call.

    Args:
        fn: zero-argument callable to time.
        repeats: measured steady-state repetitions.
        warmups: unmeasured calls after the first call.

    Returns:
        Dict with first, median, mean, and std durations in seconds.
    """
    start = perf_counter()
    _block(fn())
    first = perf_counter() - start

    for _ in range(warmups):
        _block(fn())

    timings = []
    for _ in range(repeats):
        start = perf_counter()
        _block(fn())
        timings.append(perf_counter() - start)

    values = np.asarray(timings, dtype=float)
    return {
        "first_call_seconds": float(first),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values)),
    }


def _response_vector(response: object) -> np.ndarray:
    """Normalise PUF responses to a flat numpy vector."""
    if isinstance(response, tuple):
        response = response[1]
    return np.asarray(response).ravel().astype(np.uint8)


def run_experiment(
    n_stages: int = 64,
    n_train: int = 5_000,
    block_sizes: Sequence[int] = (1, 8, 32, 128, 512, 2048, 8192),
    repeats: int = 30,
    warmups: int = 5,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Train attack models once and compare steady-state inference time.

    Args:
        n_stages (int): Arbiter challenge bit-length.
        n_train (int): CRPs used to train each attack model.
        block_sizes (Sequence[int]): prediction batch sizes to time.
        repeats (int): measured repetitions per model/block size.
        warmups (int): unmeasured warm-up calls after the first call.
        seed (int): RNG seed.
        save_path (str, optional): figure destination.

    Returns:
        Dict[str, object]: timing table and fitted model accuracies.
    """
    rng = jax.random.PRNGKey(seed)
    rng, sk_puf, sk_train, sk_test = jax.random.split(rng, 4)
    puf = Arbiter(sk_puf, (1, n_stages))

    train_chall_jax = generate_challenges(sk_train, (n_train, n_stages))
    train_resp = _response_vector(puf.get_response(train_chall_jax))
    train_chall = np.asarray(train_chall_jax)

    test_chall_jax = generate_challenges(sk_test, (max(block_sizes), n_stages))
    test_resp = _response_vector(puf.get_response(test_chall_jax))
    test_chall = np.asarray(test_chall_jax)

    models = {
        "LR Phi": LRAttack(n_restarts=1, max_iter=2_000, random_state=seed),
        "Ridge Phi": RidgeAttack(),
        "DNN raw": DNNAttack(
            hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed, early_stopping=True
        ),
        "DNN Phi": PhiDNNAttack(
            hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed, early_stopping=True
        ),
    }

    accuracy: Dict[str, float] = {}
    for name, model in models.items():
        model.fit(train_chall, train_resp)
        accuracy[name] = model.accuracy(test_chall, test_resp)
        print(f"[exp6] {name:10s} test accuracy={accuracy[name]:.4f}")

    timings: Dict[str, Dict[int, Dict[str, float]]] = {"True PUF JAX": {}}
    timings.update({name: {} for name in models})

    for block_size in block_sizes:
        c_np = test_chall[:block_size]
        c_jax = test_chall_jax[:block_size]
        timings["True PUF JAX"][block_size] = _time_call(
            lambda c=c_jax: puf.get_response(c), repeats=repeats, warmups=warmups
        )
        print(
            f"[exp6] True PUF JAX block={block_size:5d} "
            f"median={timings['True PUF JAX'][block_size]['median_seconds'] * 1e3:.3f} ms"
        )

        for name, model in models.items():
            timings[name][block_size] = _time_call(
                lambda m=model, c=c_np: m.predict(c), repeats=repeats, warmups=warmups
            )
            print(
                f"[exp6] {name:10s} block={block_size:5d} "
                f"median={timings[name][block_size]['median_seconds'] * 1e3:.3f} ms"
            )

    results: Dict[str, object] = {
        "n_stages": n_stages,
        "n_train": n_train,
        "block_sizes": list(block_sizes),
        "accuracy": accuracy,
        "timings": timings,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Exp 6: Inference Timing ({n_stages}-stage Arbiter, n_train={n_train})")

    for name, by_block in timings.items():
        med_ms = [by_block[x]["median_seconds"] * 1e3 for x in block_sizes]
        us_per_challenge = [by_block[x]["median_seconds"] * 1e6 / x for x in block_sizes]
        axes[0].plot(block_sizes, med_ms, marker="o", label=name)
        axes[1].plot(block_sizes, us_per_challenge, marker="o", label=name)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Challenge block size")
    axes[0].set_ylabel("Median block latency (ms)")
    axes[0].set_title("Total inference latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Challenge block size")
    axes[1].set_ylabel("Median latency per challenge (µs)")
    axes[1].set_title("Amortised inference latency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp6] Figure saved to {save_path}")
    plt.show()

    return results


if __name__ == "__main__":
    run_experiment(save_path="experiments/results/exp6_inference_timing.png")
