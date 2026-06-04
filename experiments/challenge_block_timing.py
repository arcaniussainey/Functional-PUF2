"""
The verifier issues a block of ``x`` challenges.  The responder must compute
all ``x`` responses, apply a shared secret delay function ``f(x)``, and return
a verifier-specified subset of ``m`` responses.

The verifier's expected round-trip time is:

    R_expected = 2*L + G*x + f(x)

where:
  * ``L``   -- one-way network latency (known to both parties)
  * ``G``   -- per-challenge generation time (known to verifier, characterised empirically)
  * ``f(x)``-- secret delay function known only to the verifier and the true PUF
               (prevents the attacker from computing ``G*x + f(x)`` / x trivially)

The verifier accepts a response if:
    |R_observed - R_expected| <= margin

AND the returned ``m`` responses match the verifier's twin predictions.

The attacker faces two problems:
  1. **Timing**: the attacker does not know ``f(x)``.  It can estimate ``G``
     from single-CRP exchanges (experiment 4) and guess at ``f(x)`` -- but
     any error in its guess of ``G + f(x)/x`` accumulates over ``x``.
  2. **Accuracy**: the attacker must answer at least some fraction of the
     ``m`` subset correctly to pass the response check; a poorly trained
     model fails here even if the timing is correct.

Assumptions
-----------
* ``f(x) = alpha * x^beta + gamma`` is the secret delay function.  The
  verifier knows all three parameters; the attacker knows only the functional
  form but not the parameters (it must estimate or guess).
* The attacker can observe ``k_obs`` single-CRP exchanges to estimate ``G``
  but cannot directly observe ``f(x)`` because the single-CRP protocol does
  not exercise it.
* The ``m`` returned challenges are a random subset of the ``x`` block,
  selected by the verifier after the block is issued.  The attacker must
  answer all ``x`` challenges in order to guarantee answering the subset.
* The attacker's response accuracy on the subset is bounded by the quality
  of its ML model; a weak attacker will fail even with perfect timing.

Output
------
Three subplots:
  1. Verifier accept rate (timing only) vs margin for true PUF and attacker
     variants (different f(x) estimation strategies).
  2. Response accuracy on the returned subset vs training CRP count.
  3. Combined accept rate (timing AND accuracy) for a range of thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from Pufs.FunctionalPuf import (
    Arbiter,
    generate_challenges,
    noisy_get_response,
)
from Pufs.validation import calibrate_noise_for_ber
from Attack.lr_attack import LRAttack


def f_delay(x: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Shared secret delay function known to the verifier and the true PUF.

    ``f(x) = alpha * x^beta + gamma``

    The shape parameter ``beta < 1`` makes the function sub-linear (overhead
    per-challenge decreases with block size), which is realistic for hardware
    that amortises setup costs.

    Args:
        x (np.ndarray): block sizes.
        alpha (float): scale parameter.
        beta (float): shape parameter, should be in (0, 1).
        gamma (float): fixed offset.

    Returns:
        np.ndarray: delay values matching the shape of x.
    """
    return alpha * np.power(x.astype(float), beta) + gamma


def _simulate_block_rtt(
    n: int,
    block_size: int,
    latency: float,
    G: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate ``n`` block-protocol round-trip times.

    Args:
        n (int): number of simulated exchanges.
        block_size (int): challenge block size x.
        latency (float): one-way network latency.
        G (float): per-challenge generation time.
        alpha, beta, gamma (float): f(x) parameters.
        sigma_net (float): network latency jitter std deviation.
        sigma_gen (float): generation time jitter std deviation per challenge.
        rng: numpy random generator.

    Returns:
        np.ndarray: shape (n,) RTT values.
    """
    x_arr = np.array([block_size], dtype=float)
    fx = float(f_delay(x_arr, alpha, beta, gamma)[0])
    net_noise = rng.normal(0.0, sigma_net, size=n)
    # Generation noise scales with block size
    gen_noise = rng.normal(0.0, sigma_gen * np.sqrt(block_size), size=n)
    return 2.0 * latency + G * block_size + fx + net_noise + gen_noise


def run_experiment(
    n_stages: int = 64,
    noise_ber: float = 0.04,
    block_sizes: Sequence[int] = (10, 25, 50, 100, 200, 800),
    subset_fraction: float = 0.3,
    G_true: float = 0.01,
    G_attacker: float = 0.001,
    latency: float = 5.0,
    alpha: float = 0.5,
    beta: float = 0.7,
    gamma: float = 1.0,
    sigma_net: float = 0.3,
    sigma_gen: float = 0.005,
    n_exchanges: int = 5_000,
    n_train_crp_values: Sequence[int] = (500, 2000, 10_000, 1_000_000),
    k_obs: int = 500,
    margins: Optional[np.ndarray] = None,
    accuracy_threshold: float = 0.90,
    seed: int = 0,
    save_path: Optional[str] = None,
    xor_k: int = 4,
) -> Dict:
    """
    Run the challenge-block timing authentication experiment.

    Args:
        n_stages (int): challenge bit-length.
        noise_ber (float): target BER under noise.
        block_sizes (Sequence[int]): block sizes x to sweep.
        subset_fraction (float): fraction of block responses returned.
        G_true (float): true PUF per-challenge generation time.
        G_attacker (float): attacker per-challenge generation time.
        latency (float): one-way network latency.
        alpha, beta, gamma (float): secret delay function parameters.
        sigma_net (float): network jitter std deviation.
        sigma_gen (float): per-challenge generation jitter std deviation.
        n_exchanges (int): simulated authentication exchanges per block size.
        n_train_crp_values (Sequence[int]): attacker training set sizes.
        k_obs (int): observations used by attacker to estimate G_true.
        margins (np.ndarray, optional): timing margin sweep.
        accuracy_threshold (float): minimum subset accuracy required to pass.
        seed (int): RNG seed.
        save_path (str, optional): figure save path.
        xor_k (int): XOR arbiter count (unused, reserved for XOR variant).

    Returns:
        Dict with accept rates and accuracy results.
    """
    rng_np = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)

    if margins is None:
        margins = np.linspace(0.1, 6.0, 80)

    rng_jax, sk_puf, sk_calib, sk_eval, sk_train = jax.random.split(rng_jax, 5)
    puf = Arbiter(sk_puf, (1, n_stages))

    calib_chall = generate_challenges(sk_calib, (5000, n_stages))
    calib = calibrate_noise_for_ber(sk_calib, puf.weight, calib_chall, target_ber=noise_ber)
    sigma_noise = float(calib.sigma)
    print(f"[exp5] sigma_noise={sigma_noise:.4f} for BER={noise_ber}")

    # Attacker estimates G by observing k_obs single-CRP exchange RTTs
    x1 = np.array([1], dtype=float)
    single_rtt_obs = _simulate_block_rtt(
        k_obs, 1, latency, G_true, alpha, beta, gamma, sigma_net, sigma_gen, rng_np
    )
    # Attacker only knows R = 2*L + G + f(1) + noise, and f(1) = alpha + gamma
    # The attacker knows f's functional form but not its parameters,
    # so it guesses f(1) = 0 and f(x)/x = 0 for all x.
    # This is the worst-case adversary assumption -- it underestimates the delay.
    G_attacker_est = float(np.mean(single_rtt_obs)) - 2.0 * latency
    print(f"[exp5] Attacker G estimate (ignoring f): {G_attacker_est:.4f}, true G+f(1)={G_true + f_delay(x1, alpha, beta, gamma)[0]:.4f}")

    # Train attacker ML models at each CRP count
    max_crp = max(n_train_crp_values)
    train_chall = generate_challenges(sk_train, (max_crp, n_stages))
    rng_jax, noisy_train_resp_raw = noisy_get_response(
        rng_jax, puf.weight, train_chall, jnp.float32(sigma_noise)
    )
    noisy_train_resp = np.asarray(noisy_train_resp_raw).ravel()
    train_chall_np = np.asarray(train_chall)

    trained_models: Dict[int, LRAttack] = {}
    for n_crp in n_train_crp_values:
        m = LRAttack(n_restarts=3)
        m.fit(train_chall_np[:n_crp], noisy_train_resp[:n_crp])
        trained_models[n_crp] = m
        print(f"[exp5] Trained LRAttack on {n_crp} CRPs")

    eval_chall_jax = generate_challenges(sk_eval, (500, n_stages))
    eval_chall_np = np.asarray(eval_chall_jax)
    noiseless_resp_eval = np.asarray(puf.get_response(eval_chall_jax)).ravel()

    results: Dict[str, object] = {
        "block_sizes": list(block_sizes),
        "timing_accept": {},
        "combined_accept": {},
        "subset_accuracy": {},
    }

    # For each block size compute timing and accuracy metrics
    for x in block_sizes:
        x_arr = np.array([x], dtype=float)
        R_expected = 2.0 * latency + G_true * x + float(f_delay(x_arr, alpha, beta, gamma)[0])
        print(f"\n[exp5] Block size x={x}, R_expected={R_expected:.3f}")

        # True PUF timing
        rtt_true = _simulate_block_rtt(
            n_exchanges, x, latency, G_true, alpha, beta, gamma, sigma_net, sigma_gen, rng_np
        )

        # Attacker timing -- uses estimated G and guesses f(x) = f(1) * x (linear extrapolation)
        # This is a mild adversary; a naive attacker guesses f(x) = 0 entirely
        f1_est = G_attacker_est - G_attacker  # attacker's entire estimate of overhead
        f_x_guess = f1_est * x  # linear extrapolation from single-CRP obs
        rtt_attacker = _simulate_block_rtt(
            n_exchanges, x, latency, G_attacker,
            0.0, 1.0, f_x_guess,  # attacker's guessed f(x) baked into gamma
            sigma_net, sigma_gen, rng_np,
        )

        # Timing accept rates
        true_timing_accept = np.array([float(np.mean(np.abs(rtt_true - R_expected) <= m)) for m in margins])
        attacker_timing_accept = np.array([float(np.mean(np.abs(rtt_attacker - R_expected) <= m)) for m in margins])

        results["timing_accept"][x] = {
            "true": true_timing_accept.tolist(),
            "attacker": attacker_timing_accept.tolist(),
        }

        # Subset response accuracy for each attacker model
        n_subset = max(1, int(x * subset_fraction))
        subset_idx = rng_np.choice(len(eval_chall_np), size=min(n_subset, len(eval_chall_np)), replace=False)
        subset_chall = eval_chall_np[subset_idx]
        subset_resp = noiseless_resp_eval[subset_idx]

        acc_by_ncrp = {}
        for n_crp, model in trained_models.items():
            acc = model.accuracy(subset_chall, subset_resp)
            acc_by_ncrp[n_crp] = acc
            print(f"  n_crp={n_crp} subset acc={acc:.4f}")

        results["subset_accuracy"][x] = acc_by_ncrp

        # Combined accept: timing AND accuracy both pass
        combined_by_ncrp = {}
        for n_crp, model in trained_models.items():
            subset_acc = acc_by_ncrp[n_crp]
            acc_pass = subset_acc >= accuracy_threshold
            combined = attacker_timing_accept * (1.0 if acc_pass else 0.0)
            combined_by_ncrp[n_crp] = {
                "combined_accept": combined.tolist(),
                "acc_passes": acc_pass,
            }
        results["combined_accept"][x] = combined_by_ncrp

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Exp 5: Challenge-Block Timing Protocol\n"
        f"G_true={G_true}, G_atk={G_attacker}, f(x)={alpha}*x^{beta}+{gamma}, BER={noise_ber}",
        fontsize=11,
    )

    # Subplot 1: timing accept rates for one representative block size
    x_rep = block_sizes[len(block_sizes) // 2]
    ax = axes[0]
    ax.plot(margins, results["timing_accept"][x_rep]["true"], label=f"True PUF (x={x_rep})", color="black")
    ax.plot(margins, results["timing_accept"][x_rep]["attacker"], label=f"Attacker (x={x_rep})", color="firebrick", linestyle="--")
    for x in block_sizes:
        ax.plot(margins, results["timing_accept"][x]["true"], alpha=0.3, color="black")
        ax.plot(margins, results["timing_accept"][x]["attacker"], alpha=0.3, color="firebrick", linestyle="--")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Timing accept rate")
    ax.set_title("Timing-only accept rate vs margin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: subset accuracy vs block size per attacker training budget
    ax = axes[1]
    for n_crp in n_train_crp_values:
        accs = [results["subset_accuracy"][x][n_crp] for x in block_sizes]
        ax.plot(block_sizes, accs, marker="o", label=f"LR n_crp={n_crp}")
    ax.axhline(accuracy_threshold, color="gray", linestyle="--", label=f"Threshold {accuracy_threshold:.0%}")
    ax.set_xlabel("Block size x")
    ax.set_ylabel("Subset response accuracy")
    ax.set_title(f"Response accuracy on {subset_fraction:.0%} subset")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Subplot 3: combined accept for the best attacker model vs margin, per block size
    ax = axes[2]
    best_ncrp = max(n_train_crp_values)
    for x in block_sizes:
        combined = results["combined_accept"][x][best_ncrp]["combined_accept"]
        ax.plot(margins, combined, label=f"x={x}")
    true_timing_x_rep = results["timing_accept"][x_rep]["true"]
    ax.plot(margins, true_timing_x_rep, color="black", linewidth=2, linestyle=":", label=f"True PUF timing (x={x_rep})")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Combined accept rate")
    ax.set_title(f"Combined (timing + accuracy >= {accuracy_threshold:.0%}), best attacker n={best_ncrp}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp5] Figure saved to {save_path}")
    plt.show()

    return results


if __name__ == "__main__":
    run_experiment(
        n_stages=64,
        noise_ber=0.04,
        block_sizes=[10, 25, 50, 100, 200, 800],
        subset_fraction=0.3,
        G_true=0.01,
        G_attacker=0.001,
        latency=5.0,
        alpha=0.5,
        beta=0.7,
        gamma=1.0,
        sigma_net=0.3,
        sigma_gen=0.005,
        n_exchanges=5_000,
        n_train_crp_values=[500, 2000, 10_000, 1_000_000],
        save_path="experiments/results/exp5_block_timing.png",
    )
