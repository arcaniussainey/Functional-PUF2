"""
A verifier holds a precise digital twin of the true PUF (same weights, exact
copy) and knows the noise level the physical PUF operates under (a BER of
``noise_ber``).  The attacker train s one of four ML models on CRPs collected
from the noisy physical PUF, then tries to impersonate it.

The verifier evaluates candidate responses on a held-out challenge set and
computes accuracy against the *noiseless* true model (the twin).  The true PUF
will land slightly below 1.0 due to noise; a well-trained attacker may land at
a similar level, while a weak attacker will lag clearly.

Additionally, this experiment examines whether attackers can reproduce the
correct responses on *stable* challenges -- those whose delay margin is large
enough that noise rarely flips them.  The stability of each challenge is
estimated from the absolute delay margin ``|delta|`` using the verifier's twin.

Assumptions
-----------
* Noise is injected per-challenge with a Gaussian perturbation to the weight
  vector (sigma calibrated to produce the target BER) using ``noisy_get_response``.
* The attacker trains on *noisy* CRPs (i.e. responses that have already been
  flipped by noise), which is the realistic adversarial scenario.
* The verifier evaluates accuracy against the noiseless twin, not against the
  noisy physical PUF.  This matches the Rührmair et al. evaluation protocol.

Output
------
A matplotlib figure with three subplots:
  1. Test accuracy vs. number of training CRPs for each attacker.
  2. Stability-weighted accuracy (restricted to top-k most stable challenges).
  3. The distribution of ``|delta|`` values across the challenge space, with
     attacker accuracy overlaid by stability decile.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from Pufs.FunctionalPuf import (
    Arbiter, Xor, FF_Arbiter, FFLoop,
    generate_challenges,
    noisy_get_response,
    noisy_xor_get_response,
)
from Pufs.validation import calibrate_noise_for_ber
from Attack.lr_attack import LRAttack
from Attack.ridge_attack import RidgeAttack
from Attack.dnn_attack import DNNAttack
from Attack.dnn_attack_2 import PhiDNNAttack
from Attack.cmaes_attack import CMA_AVAILABLE, CMAESAttack, FeedForwardCMAESAttack


def _noisy_responses_arbiter(
    rng: jax.Array,
    puf: Arbiter,
    challenges: jax.Array,
    sigma: float,
) -> Tuple[jax.Array, np.ndarray]:
    """
    Collect noisy responses from an Arbiter PUF.

    Args:
        rng: PRNG key.
        puf: Arbiter PUF instance.
        challenges: challenge matrix.
        sigma: noise std deviation.

    Returns:
        Tuple of (updated rng, noisy response array shape (N,) dtype uint8).
    """
    rng, resp = noisy_get_response(
        rng, puf.weight, challenges, jnp.float32(sigma)
    )
    return rng, np.asarray(resp).ravel()


def _noiseless_responses(puf, challenges: jax.Array) -> np.ndarray:
    """
    Return noiseless ground-truth responses.  Handles Arbiter, Xor, and FF_Arbiter.

    For XOR PUFs the XOR-reduced scalar response (second return value of
    xor_get_response) is used.

    Args:
        puf: any PUF instance with a get_response method.
        challenges: challenge matrix.

    Returns:
        np.ndarray: shape (N,) dtype uint8.
    """
    resp = puf.get_response(challenges)
    if isinstance(resp, tuple):
        # XOR PUF returns (individual, xor_r)
        resp = resp[1]
    return np.asarray(resp).ravel()


def _delta_magnitude(puf, challenges: jax.Array) -> np.ndarray:
    """
    Return a per-challenge stability score based on the absolute delay margin.

    For a single Arbiter PUF there is one delta per challenge.  For an XOR PUF
    with k arbiters each challenge has k deltas; the stability of the XOR
    response is limited by the arbiter with the smallest |delta| (the one most
    likely to flip under noise), so the minimum is taken across the k columns.

    Args:
        puf: PUF instance.
        challenges: challenge matrix.

    Returns:
        np.ndarray: shape (N,) dtype float32, one stability value per challenge.
    """
    delta = np.asarray(puf.get_delta_response(challenges))
    abs_delta = np.abs(delta)
    if abs_delta.ndim == 2:
        # XOR / multi-arbiter: take min |delta| across arbiters (k columns)
        return abs_delta.min(axis=1)
    return abs_delta.ravel()


def run_experiment(
    puf_type: str = "arbiter",
    n_stages: int = 64,
    noise_ber: float = 0.04,
    crp_counts: Sequence[int] = (500, 1000, 2000, 5000, 10000, 20000),
    n_test: int = 10_000,
    stability_percentile: float = 75.0,
    seed: int = 0,
    save_path: Optional[str] = None,
    xor_k: int = 4,
    ff_loops: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict:
    """
    Run the raw-accuracy classification experiment.

    Args:
        puf_type (str): one of ``'arbiter'``, ``'xor'``, ``'ff'``.
        n_stages (int): number of challenge bits.
        noise_ber (float): target bit-error rate under noise.
        crp_counts (Sequence[int]): training set sizes to sweep.
        n_test (int): number of challenges in the held-out evaluation set.
        stability_percentile (float): challenges above this |delta| percentile
            are considered ``stable``.
        seed (int): base PRNG seed.
        save_path (str, optional): path to save the figure.
        xor_k (int): number of XOR arbiters (only used when puf_type='xor').
        ff_loops (Sequence[Tuple[int,int]], optional): FF loop specs for puf_type='ff'.

    Returns:
        Dict with keys ``'results'`` (per-attacker accuracy curves),
        ``'true_puf_accuracy'`` (noise-limited accuracy of the real PUF),
        ``'stable_challenge_idx'`` (boolean mask of stable challenges), and
        ``'sigma'`` (calibrated noise sigma).
    """
    rng = jax.random.PRNGKey(seed)
    rng, sk_puf, sk_calib, sk_test, sk_train = jax.random.split(rng, 5)

    if puf_type == "arbiter":
        puf = Arbiter(sk_puf, (1, n_stages))
    elif puf_type == "xor":
        puf = Xor(sk_puf, (xor_k, n_stages))
    elif puf_type == "ff":
        loops = ff_loops or [(10, 30), (20, 50)]
        puf = FF_Arbiter(int(sk_puf[0]), n_stages, [FFLoop(s, t) for s, t in loops])
    else:
        raise ValueError(f"Unknown puf_type: {puf_type!r}")

    calib_challenges = generate_challenges(sk_calib, (5000, n_stages))
    # calibrate_noise_for_ber internally uses noisy_get_response which expects a
    # single-arbiter weight row.  For XOR PUFs we calibrate on the first arbiter;
    # all arbiters share the same noise sigma in this experiment.
    calib_weight = puf.weight if puf.weight.shape[0] == 1 else puf.weight[:1]
    calib_result = calibrate_noise_for_ber(
        sk_calib, calib_weight, calib_challenges, target_ber=noise_ber
    )
    sigma = float(calib_result.sigma)
    print(f"[exp1] Calibrated sigma={sigma:.4f} for BER={noise_ber} (measured {calib_result.measured_ber:.4f})")

    test_challenges = generate_challenges(sk_test, (n_test, n_stages))
    test_challenges_np = np.asarray(test_challenges)
    true_test_resp = _noiseless_responses(puf, test_challenges)

    # Measure how stable each test challenge is using |delta|
    abs_delta = _delta_magnitude(puf, test_challenges)
    threshold = float(np.percentile(abs_delta, stability_percentile))
    stable_mask = abs_delta >= threshold
    print(
        f"[exp1] Stable challenges (|delta| >= {threshold:.2f}): "
        f"{stable_mask.sum()} / {n_test}"
    )

    # True noisy PUF accuracy against its own noiseless twin.
    # Each arbiter receives the same scalar sigma; noisy_xor_get_response
    # expects shape (k, 1) so we broadcast accordingly.
    if puf_type == "arbiter":
        rng, noisy_true_resp_raw = noisy_get_response(
            rng, puf.weight, test_challenges, jnp.float32(sigma)
        )
        noisy_true_resp = np.asarray(noisy_true_resp_raw).ravel()
    else:
        k_arb = puf.dim[0] if hasattr(puf, "dim") else puf.weight.shape[0]
        sigma_vec = jnp.full((k_arb, 1), sigma)
        rng, noisy_xor_resp_raw = noisy_xor_get_response(
            rng, puf.weight, test_challenges, sigma_vec
        )
        noisy_true_resp = np.asarray(noisy_xor_resp_raw).ravel()
    true_puf_accuracy = float(np.mean(noisy_true_resp == true_test_resp))
    print(f"[exp1] True PUF noisy accuracy vs noiseless twin: {true_puf_accuracy:.4f}")

    max_crp = max(crp_counts)
    train_challenges = generate_challenges(sk_train, (max_crp, n_stages))
    train_challenges_np = np.asarray(train_challenges)

    # Generate noisy training labels
    if puf_type == "arbiter":
        rng, noisy_train_resp = _noisy_responses_arbiter(rng, puf, train_challenges, sigma)
    else:
        sigma_vec = jnp.full((puf.dim[0], 1), sigma)
        rng, noisy_r = noisy_xor_get_response(rng, puf.weight, train_challenges, sigma_vec)
        noisy_train_resp = np.asarray(noisy_r).ravel()

    attacker_factories = {
        "LR":       lambda: LRAttack(n_restarts=3),
        "Ridge":    lambda: RidgeAttack(),
        "DNN-raw":  lambda: DNNAttack(hidden_layer_sizes=(128, 64), max_iter=30000),
        "DNN-Phi":  lambda: PhiDNNAttack(hidden_layer_sizes=(128, 64), max_iter=30000),
    }
    if CMA_AVAILABLE:
        if puf_type == "ff":
            attacker_factories["FF-CMA-ES"] = lambda: FeedForwardCMAESAttack(
                loops=ff_loops or [(10, 30), (20, 50)],
                max_fevals=20_000,
            )
        else:
            attacker_factories["CMA-ES"] = lambda: CMAESAttack(max_fevals=20_000)
    else:
        print("[exp1] Skipping CMA-ES; optional package 'cma' is not installed.")

    results: Dict[str, Dict] = {name: {"accuracy": [], "stable_accuracy": []} for name in attacker_factories}

    for n_crp in crp_counts:
        c_sub = train_challenges_np[:n_crp]
        r_sub = noisy_train_resp[:n_crp]
        for name, factory in attacker_factories.items():
            attacker_fresh = factory()
            attacker_fresh.fit(c_sub, r_sub)
            acc = attacker_fresh.accuracy(test_challenges_np, true_test_resp)
            stable_acc = attacker_fresh.accuracy(
                test_challenges_np[stable_mask], true_test_resp[stable_mask]
            )
            results[name]["accuracy"].append(acc)
            results[name]["stable_accuracy"].append(stable_acc)
            print(f"  [exp1] {name:8s} n={n_crp:6d} acc={acc:.4f} stable_acc={stable_acc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Exp 1: Raw Accuracy -- {puf_type.upper()} PUF ({n_stages} stages, BER={noise_ber})",
        fontsize=13,
    )

    ax = axes[0]
    for name, data in results.items():
        ax.semilogx(crp_counts, data["accuracy"], marker="o", label=name)
    ax.axhline(true_puf_accuracy, color="black", linestyle="--", label="True PUF (noisy)")
    ax.set_xlabel("Training CRPs")
    ax.set_ylabel("Accuracy vs noiseless twin")
    ax.set_title("Overall accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, data in results.items():
        ax.semilogx(crp_counts, data["stable_accuracy"], marker="o", label=name)
    ax.axhline(
        float(np.mean(_noiseless_responses(puf, test_challenges[stable_mask]) ==
                      true_test_resp[stable_mask])),
        color="black", linestyle="--", label="Oracle (noiseless)",
    )
    ax.set_xlabel("Training CRPs")
    ax.set_ylabel("Accuracy on stable challenges")
    ax.set_title(f"Stable challenge accuracy (|Δ| >= p{stability_percentile:.0f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.hist(abs_delta, bins=50, color="steelblue", alpha=0.7, label="|Δ| distribution")
    ax.axvline(threshold, color="red", linestyle="--", label=f"p{stability_percentile:.0f} threshold")
    ax.set_xlabel("|Δ| (delay margin)")
    ax.set_ylabel("Count")
    ax.set_title("Challenge stability distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp1] Figure saved to {save_path}")
    plt.show()

    return {
        "results": results,
        "true_puf_accuracy": true_puf_accuracy,
        "stable_challenge_mask": stable_mask,
        "sigma": sigma,
        "crp_counts": list(crp_counts),
        "puf_type": puf_type,
    }


if __name__ == "__main__":
    run_experiment(
        puf_type="arbiter",
        n_stages=64,
        noise_ber=0.04,
        crp_counts=[500, 1000, 2000, 5000, 10000, 20000, 1_000_000],
        n_test=10_000,
        save_path="experiments/results/exp1_arbiter_raw_accuracy.png",
    )
