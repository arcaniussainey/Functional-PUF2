"""
Stable-only CRP training: what if the attacker only receives
the most stable (easiest for the PUF, hardest for an attacker without the
model) challenges?

In experiment 1 the attacker receives a uniformly random sample of CRPs.
Here we consider a verifier-controlled exchange where only challenges with
the highest |delta| values are issued.  These challenges are the ones the
true PUF answers most reliably under noise.  Paradoxically, they are also
the challenges where an attacker's model is most likely to agree -- because
they sit far from the decision boundary.

The question is: if the attacker can only train on these stable CRPs, does
their overall generalisation drop?  We hypothesise that:
  - Stable CRPs are over-represented near the extremes of the feature space
    and provide limited information about the boundary region.
  - An attacker trained only on stable CRPs may achieve high accuracy on
    other stable challenges but degrade on unstable (boundary-proximal) ones.

This is the converse of experiment 1 -- we fix the number of training CRPs
and vary whether they are stable-only or random.

Assumptions
----------
* |delta| is computed using the verifier's noiseless twin, not from the
  noisy physical device.  A real verifier with a precise digital twin can
  do exactly this; an attacker who does not have the model cannot pre-select
  stable challenges this way.
* Training CRPs are still collected with noise applied (the attacker sees
  the noisy response from the physical PUF).
* Evaluation uses the noiseless twin response as ground truth.

Output
------
A figure with two subplots:
  1. Accuracy on all challenges for stable-only vs random-sample training.
  2. Accuracy broken down by |delta| decile, comparing stable-only training
     to uniform training.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional, Sequence, Tuple

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


def _noiseless_responses(puf, challenges):
    """Return noiseless responses as a flat numpy array."""
    resp = puf.get_response(challenges)
    if isinstance(resp, tuple):
        resp = resp[1]
    return np.asarray(resp).ravel()


def _noisy_responses(rng, puf, challenges, sigma, puf_type):
    """Return noisy responses from any supported PUF type."""
    if puf_type == "arbiter":
        rng, resp = noisy_get_response(rng, puf.weight, challenges, jnp.float32(sigma))
        return rng, np.asarray(resp).ravel()
    sigma_vec = jnp.full((puf.dim[0], 1), sigma)
    rng, resp = noisy_xor_get_response(rng, puf.weight, challenges, sigma_vec)
    return rng, np.asarray(resp).ravel()


def run_experiment(
    puf_type: str = "arbiter",
    n_stages: int = 64,
    noise_ber: float = 0.04,
    n_train: int = 10_000,
    stable_fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    n_test: int = 20_000,
    seed: int = 0,
    save_path: Optional[str] = None,
    xor_k: int = 4,
    ff_loops: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict:
    """
    Run the stable-CRP training experiment.

    Args:
        puf_type (str): one of ``'arbiter'``, ``'xor'``, ``'ff'``.
        n_stages (int): challenge bit-length.
        noise_ber (float): target bit-error rate.
        n_train (int): total number of training CRPs available in the pool.
        stable_fractions (Sequence[float]): fractions of top-|delta| challenges
            to keep in the stable-only training sets.  ``1.0`` is the full
            random set (no filtering).
        n_test (int): size of the held-out evaluation set.
        seed (int): base PRNG seed.
        save_path (str, optional): path to save the figure.
        xor_k (int): number of XOR arbiters (puf_type='xor' only).
        ff_loops (Sequence[Tuple[int,int]], optional): FF loop specs.

    Returns:
        Dict containing accuracy curves per attacker per stable-fraction.
    """
    rng = jax.random.PRNGKey(seed)
    rng, sk_puf, sk_calib, sk_pool, sk_test = jax.random.split(rng, 5)

    if puf_type == "arbiter":
        puf = Arbiter(sk_puf, (1, n_stages))
    elif puf_type == "xor":
        puf = Xor(sk_puf, (xor_k, n_stages))
    elif puf_type == "ff":
        loops = ff_loops or [(10, 30), (20, 50)]
        puf = FF_Arbiter(int(sk_puf[0]), n_stages, [FFLoop(s, t) for s, t in loops])
    else:
        raise ValueError(f"Unknown puf_type: {puf_type!r}")

    calib_chall = generate_challenges(sk_calib, (5000, n_stages))
    calib_weight = puf.weight if puf.weight.shape[0] == 1 else puf.weight[:1]
    calib = calibrate_noise_for_ber(sk_calib, calib_weight, calib_chall, target_ber=noise_ber)
    sigma = float(calib.sigma)
    print(f"[exp2] sigma={sigma:.4f} for BER={noise_ber}")

    # Build the full training pool
    pool_chall = generate_challenges(sk_pool, (n_train, n_stages))
    pool_chall_np = np.asarray(pool_chall)
    rng, pool_noisy_resp = _noisy_responses(rng, puf, pool_chall, sigma, puf_type)
    pool_true_resp = _noiseless_responses(puf, pool_chall)

    # Compute |delta| for every training CRP using the twin (verifier privilege)
    raw_delta = np.asarray(puf.get_delta_response(pool_chall))
    abs_raw = np.abs(raw_delta)
    # For XOR PUFs (k arbiters) take min |delta| across columns; scalar otherwise.
    abs_delta_pool = abs_raw.min(axis=1) if abs_raw.ndim == 2 else abs_raw.ravel()

    # Sort pool indices by |delta|, descending -- most stable first
    sorted_idx = np.argsort(abs_delta_pool)[::-1]

    test_chall = generate_challenges(sk_test, (n_test, n_stages))
    test_chall_np = np.asarray(test_chall)
    test_true_resp = _noiseless_responses(puf, test_chall)

    _test_delta_raw = np.abs(np.asarray(puf.get_delta_response(test_chall)))
    test_abs_delta = _test_delta_raw.min(axis=1) if _test_delta_raw.ndim == 2 else _test_delta_raw.ravel()
    # Create 10 decile buckets
    decile_edges = np.percentile(test_abs_delta, np.linspace(0, 100, 11))
    decile_labels = [f"D{i+1}" for i in range(10)]

    attacker_factories = {
        "LR":    lambda: LRAttack(n_restarts=3),
        "Ridge": lambda: RidgeAttack(),
        "DNN":   lambda: DNNAttack(hidden_layer_sizes=(128, 64), max_iter=300),
    }

    # results[attacker_name][stable_fraction] -> {overall_acc, decile_accs}
    all_results: Dict[str, Dict[str, Dict]] = {
        name: {} for name in attacker_factories
    }

    for frac in stable_fractions:
        n_keep = max(50, int(n_train * frac))
        if frac < 1.0:
            # Take the top n_keep most stable challenges
            sel_idx = sorted_idx[:n_keep]
        else:
            # Use all pool challenges in original (random) order
            sel_idx = np.arange(n_train)

        c_sub = pool_chall_np[sel_idx]
        r_sub = pool_noisy_resp[sel_idx]
        actual_frac = float(frac)

        print(f"[exp2] Training fraction={actual_frac:.2f}, n_keep={n_keep}")

        for name, factory in attacker_factories.items():
            model = factory()
            model.fit(c_sub, r_sub)
            overall_acc = model.accuracy(test_chall_np, test_true_resp)

            decile_accs = []
            for i in range(10):
                lo, hi = decile_edges[i], decile_edges[i + 1]
                mask = (test_abs_delta >= lo) & (test_abs_delta <= hi)
                if mask.sum() == 0:
                    decile_accs.append(float("nan"))
                else:
                    decile_accs.append(model.accuracy(test_chall_np[mask], test_true_resp[mask]))

            all_results[name][actual_frac] = {
                "overall_acc": overall_acc,
                "decile_accs": decile_accs,
            }
            print(f"  {name:8s} overall={overall_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Exp 2: Stable-only CRP Training -- {puf_type.upper()} PUF "
        f"({n_stages} stages, BER={noise_ber})",
        fontsize=13,
    )

    ax = axes[0]
    for name, frac_results in all_results.items():
        fracs = sorted(frac_results.keys())
        accs = [frac_results[f]["overall_acc"] for f in fracs]
        ax.plot([f * 100 for f in fracs], accs, marker="o", label=name)
    ax.set_xlabel("% of most-stable challenges used for training")
    ax.set_ylabel("Accuracy vs noiseless twin")
    ax.set_title("Overall test accuracy vs stability filter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(10)
    width = 0.25
    name_list = list(all_results.keys())
    # Show decile accuracy for stable-only (smallest frac) vs full random (frac=1.0)
    frac_show = [min(stable_fractions), 1.0]
    colors = ["steelblue", "firebrick"]
    for col_i, (name, color) in enumerate(zip(name_list[:2], colors)):
        for line_i, frac in enumerate(frac_show):
            if frac not in all_results[name]:
                continue
            d_accs = all_results[name][frac]["decile_accs"]
            offset = (col_i * len(frac_show) + line_i) * width - (len(name_list) * len(frac_show) / 2) * width
            ax.bar(
                x + offset, d_accs, width,
                label=f"{name} frac={frac:.0%}", alpha=0.7, color=color,
                hatch="/" if line_i == 1 else "",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}\n(|Δ| ≥ p{i*10})" for i in range(10)], fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-decile accuracy: stable-only vs random training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp2] Figure saved to {save_path}")
    plt.show()

    return {
        "results": all_results,
        "sigma": sigma,
        "puf_type": puf_type,
        "n_stages": n_stages,
    }


if __name__ == "__main__":
    run_experiment(
        puf_type="arbiter",
        n_stages=64,
        noise_ber=0.04,
        n_train=20_000,
        stable_fractions=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
        n_test=10_000,
        save_path="experiments/results/exp2_stable_crp.png",
    )
