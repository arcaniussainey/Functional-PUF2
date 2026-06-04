"""
* The true PUF is modelled using ``drift_additive``, which applies per-step
  Gaussian weight perturbations.  ``sigma_per_step`` controls the rate of drift.
* The verifier measures the aging rate by computing the BER between consecutive
  weight snapshots on a calibration challenge set.
* Two attacker strategies are simulated:
    - ``no_retrain``: trained at t=0, never updated.
    - ``periodic_retrain``: retrained every ``retrain_interval`` steps, but
      is forced to collect new CRPs (noisy) from the physical PUF.
* BER is computed between the current physical PUF (aged) and the noiseless
  initial twin, reflecting the question "how wrong would the responder be if
  it always used its t=0 model?".

Output
------
A figure with two subplots:
  1. Error rate vs. age step for the true PUF, no-retrain attacker, and
     periodic-retrain attacker, plus the predicted aging curve the verifier
     can compute.
  2. The rolling delta (additional error per step) overlaid on the verifier's
     expected rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from Pufs.FunctionalPuf import (
    Arbiter,
    generate_challenges,
    get_response,
    noisy_get_response,
)
from Pufs.aging import drift_additive
from Pufs.validation import calibrate_noise_for_ber
from Attack.lr_attack import LRAttack


def _responses_from_weight(weight, challenges):
    """Return noiseless Arbiter responses from an explicit weight matrix."""
    return np.asarray(get_response(weight, challenges)).ravel()


def _ber_between_weights(weight_a, weight_b, challenges):
    """Compute the fraction of challenges where weight_a and weight_b disagree."""
    ra = _responses_from_weight(weight_a, challenges)
    rb = _responses_from_weight(weight_b, challenges)
    return float(np.mean(ra != rb))


def run_experiment(
    n_stages: int = 64,
    noise_ber: float = 0.04,
    sigma_per_step: float = 0.05,
    n_age_steps: int = 80,
    n_eval: int = 1_000_000,
    n_train_crp: int = 1_000_000,
    retrain_interval: int = 200,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Run the aging-classification experiment for a single Arbiter PUF.

    Args:
        n_stages (int): challenge bit-length.
        noise_ber (float): baseline noise BER at t=0.
        sigma_per_step (float): std deviation of additive weight drift per age step.
        n_age_steps (int): total number of age steps to simulate.
        n_eval (int): evaluation challenge set size.
        n_train_crp (int): CRPs available to the attacker at each (re-)training.
        retrain_interval (int): steps between periodic retrains.
        seed (int): base PRNG seed.
        save_path (str, optional): figure save path.

    Returns:
        Dict with per-step error curves and the verifier's expected rate.
    """
    rng = jax.random.PRNGKey(seed)
    rng, sk_puf, sk_calib, sk_eval, sk_age, sk_train = jax.random.split(rng, 6)

    puf = Arbiter(sk_puf, (1, n_stages))
    initial_weight = puf.weight

    calib_chall = generate_challenges(sk_calib, (5000, n_stages))
    calib = calibrate_noise_for_ber(sk_calib, initial_weight, calib_chall, target_ber=noise_ber)
    sigma_noise = float(calib.sigma)
    print(f"[exp3] sigma_noise={sigma_noise:.4f} for BER={noise_ber}")

    eval_chall = generate_challenges(sk_eval, (n_eval, n_stages))
    eval_chall_np = np.asarray(eval_chall)
    true_t0_resp = _responses_from_weight(initial_weight, eval_chall)

    # Build the full aging history for the physical PUF
    weight_history = np.asarray(
        drift_additive(sk_age, initial_weight, n_age_steps, sigma_per_step=sigma_per_step)
    )
    print(f"[exp3] Weight history shape: {weight_history.shape}")

    # Measure average BER per step (verifier's known aging rate)
    step_bers = []
    for t in range(1, n_age_steps + 1):
        ber = _ber_between_weights(
            jnp.asarray(weight_history[0]), jnp.asarray(weight_history[t]), eval_chall
        )
        step_bers.append(ber)
    print(f"[exp3] BER at last step vs t=0: {step_bers[-1]:.4f}")

    # True PUF error: noisy physical PUF vs noiseless t=0 twin
    true_puf_errors = []
    for t in range(n_age_steps + 1):
        w_t = jnp.asarray(weight_history[t])
        rng, resp_noisy = noisy_get_response(rng, w_t, eval_chall, jnp.float32(sigma_noise))
        ber = float(np.mean(np.asarray(resp_noisy).ravel() != true_t0_resp))
        true_puf_errors.append(ber)

    # Attacker strategy 1: no retraining -- trained on noisy CRPs at t=0
    train_chall = generate_challenges(sk_train, (n_train_crp, n_stages))
    train_chall_np = np.asarray(train_chall)
    rng, noisy_train_resp = noisy_get_response(
        rng, initial_weight, train_chall, jnp.float32(sigma_noise)
    )
    attacker_no_retrain = LRAttack(n_restarts=3)
    attacker_no_retrain.fit(train_chall_np, np.asarray(noisy_train_resp).ravel())

    no_retrain_errors = []
    for t in range(n_age_steps + 1):
        w_t = jnp.asarray(weight_history[t])
        true_resp_t = _responses_from_weight(w_t, eval_chall)
        pred = attacker_no_retrain.predict(eval_chall_np)
        no_retrain_errors.append(float(np.mean(pred != true_resp_t)))

    # Attacker strategy 2: periodic retraining every retrain_interval steps
    periodic_errors = []
    current_attacker = LRAttack(n_restarts=3)
    current_attacker.fit(train_chall_np, np.asarray(noisy_train_resp).ravel())

    for t in range(n_age_steps + 1):
        if t > 0 and t % retrain_interval == 0:
            rng, sk_new_train = jax.random.split(rng)
            new_train_chall = generate_challenges(sk_new_train, (n_train_crp, n_stages))
            w_retrain = jnp.asarray(weight_history[t])
            rng, new_noisy_resp = noisy_get_response(
                rng, w_retrain, new_train_chall, jnp.float32(sigma_noise)
            )
            current_attacker = LRAttack(n_restarts=3)
            current_attacker.fit(
                np.asarray(new_train_chall),
                np.asarray(new_noisy_resp).ravel(),
            )
            print(f"[exp3] Periodic retrain at t={t}")

        w_t = jnp.asarray(weight_history[t])
        true_resp_t = _responses_from_weight(w_t, eval_chall)
        pred = current_attacker.predict(eval_chall_np)
        periodic_errors.append(float(np.mean(pred != true_resp_t)))

    time_steps = list(range(n_age_steps + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Exp 3: Aging Behaviour -- Arbiter PUF ({n_stages} stages, "
        f"BER={noise_ber}, sigma_step={sigma_per_step})",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(time_steps, true_puf_errors, label="True PUF (noisy, aging)", color="black")
    ax.plot(time_steps, no_retrain_errors, label="Attacker (no retrain)", color="firebrick", linestyle="--")
    ax.plot(time_steps, periodic_errors, label=f"Attacker (retrain/{retrain_interval} steps)", color="steelblue", linestyle="-.")
    ax.plot(
        time_steps[1:], step_bers,
        label="Verifier predicted BER (vs t=0 twin)", color="green", linestyle=":", alpha=0.8,
    )
    for t in range(retrain_interval, n_age_steps + 1, retrain_interval):
        ax.axvline(t, color="steelblue", alpha=0.15, linestyle="--")
    ax.set_xlabel("Age step")
    ax.set_ylabel("Error rate vs t=0 noiseless twin")
    ax.set_title("Error rate over simulated aging")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Rolling delta: d(error)/d(step)
    def rolling_delta(series, window=5):
        arr = np.asarray(series, dtype=float)
        grad = np.gradient(arr)
        return np.convolve(grad, np.ones(window) / window, mode="same")

    ax = axes[1]
    ax.plot(time_steps, rolling_delta(true_puf_errors), label="True PUF rate", color="black")
    ax.plot(time_steps, rolling_delta(no_retrain_errors), label="No-retrain attacker rate", color="firebrick", linestyle="--")
    ax.plot(time_steps, rolling_delta(periodic_errors), label=f"Periodic-retrain rate", color="steelblue", linestyle="-.")
    for t in range(retrain_interval, n_age_steps + 1, retrain_interval):
        ax.axvline(t, color="steelblue", alpha=0.15, linestyle="--")
    ax.set_xlabel("Age step")
    ax.set_ylabel("Rolling Δ error (smoothed gradient)")
    ax.set_title("Error rate delta per step (step-function signature)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp3] Figure saved to {save_path}")
    plt.show()

    return {
        "true_puf_errors": true_puf_errors,
        "no_retrain_errors": no_retrain_errors,
        "periodic_errors": periodic_errors,
        "step_bers": step_bers,
        "sigma_noise": sigma_noise,
        "sigma_per_step": sigma_per_step,
        "n_age_steps": n_age_steps,
        "retrain_interval": retrain_interval,
    }


if __name__ == "__main__":
    run_experiment(
        n_stages=64,
        noise_ber=0.04,
        sigma_per_step=0.05,
        n_age_steps=80,
        n_train_crp=10_000,
        retrain_interval=20,
        save_path="experiments/results/exp3_aging.png",
    )
