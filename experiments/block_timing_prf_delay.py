"""
challenge-block timing with a keyed PRF delay schedule.

This experiment replaces the learnable delay curve from exp5b with a keyed,
per-exchange pseudorandom delay.  The old form

    f(x) = alpha*x**beta + gamma

is intentionally easy to learn if an attacker can observe RTTs at several block
sizes.  A deterministic function of only the public block size is not a useful
secret after enough observations.  The replacement in this file is:

    d_i = delay_min + delay_span * U(HMAC-SHA256(K, transcript_i))

where the transcript binds the session nonce, block size, and optionally the
challenge-block digest.  The true device and verifier share K; an attacker sees
nonce and challenge material but cannot compute the delay without K.

Security interpretation:
  * This is no longer a pure Strong-PUF timing trick.  It is a Controlled-PUF or
    PUF-plus-secret-coprocessor design assumption.
  * If K leaks, the attack collapses to the oracle-key curve and timing stops
    distinguishing a software clone.
  * If K is protected, the attacker's best timing strategy is distributional:
    use the delay prior, choose the median delay, or sample a blind delay.
  * The verifier should still check response correctness.  PRF timing only
    removes the attacker's ability to intentionally align RTTs.

Run from the project root with:

    python experiments/exp5c_block_timing_prf_delay.py
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from Attack.lr_attack import LRAttack
from Pufs.FunctionalPuf import Arbiter, generate_challenges, noisy_get_response
from Pufs.validation import calibrate_noise_for_ber


def _u64_to_unit_interval(value: int) -> float:
    """Map an unsigned 64-bit integer to a float in [0, 1)."""
    return value / float(1 << 64)


def _challenge_digest_from_rng(rng: np.random.Generator, n_bytes: int = 32) -> bytes:
    """
    Return synthetic challenge-block digest bytes.

    The timing experiment does not need to materialise every challenge block for
    every exchange.  A random digest is enough to model transcript binding and
    keeps the experiment cheap for large block sizes.
    """
    return rng.bytes(n_bytes)


def prf_delay(
    key: bytes,
    nonce: bytes,
    block_size: int,
    challenge_digest: bytes,
    delay_min: float,
    delay_span: float,
    quantum: Optional[float] = None,
) -> float:
    """
    Compute a keyed pseudorandom delay for one authentication exchange.

    Args:
        key (bytes): verifier/device shared PRF key.
        nonce (bytes): unique verifier nonce for this exchange.
        block_size (int): challenge block size.
        challenge_digest (bytes): digest of the challenge block transcript.
        delay_min (float): minimum artificial delay.
        delay_span (float): pseudorandom delay range above delay_min.
        quantum (float, optional): quantize the result to this timing quantum.

    Returns:
        float: artificial delay value.
    """
    if delay_span < 0.0:
        raise ValueError("delay_span must be nonnegative.")
    if delay_min < 0.0:
        raise ValueError("delay_min must be nonnegative.")

    msg = b"PUF-TIMING-PRF-v1" + nonce
    msg += struct.pack(">Q", int(block_size))
    msg += hashlib.sha256(challenge_digest).digest()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    sample = struct.unpack(">Q", digest[:8])[0]
    delay = delay_min + delay_span * _u64_to_unit_interval(sample)
    if quantum is not None and quantum > 0.0:
        delay = round(delay / quantum) * quantum
    return float(delay)


def make_prf_delays(
    key: bytes,
    block_size: int,
    n_exchanges: int,
    delay_min: float,
    delay_span: float,
    rng: np.random.Generator,
    nonce_bytes: int = 16,
    quantum: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nonces and keyed delays for repeated exchanges at one block size.

    Args:
        key (bytes): PRF key.
        block_size (int): challenge block size.
        n_exchanges (int): number of exchanges to simulate.
        delay_min (float): minimum artificial delay.
        delay_span (float): pseudorandom delay range above delay_min.
        rng (np.random.Generator): simulation RNG.
        nonce_bytes (int): bytes per exchange nonce.
        quantum (float, optional): timing quantization quantum.

    Returns:
        Tuple[np.ndarray, np.ndarray]: nonces as object array and delay vector.
    """
    nonces = np.empty(n_exchanges, dtype=object)
    delays = np.empty(n_exchanges, dtype=float)
    for i in range(n_exchanges):
        nonce = rng.bytes(nonce_bytes)
        challenge_digest = _challenge_digest_from_rng(rng)
        nonces[i] = nonce
        delays[i] = prf_delay(
            key,
            nonce,
            block_size,
            challenge_digest,
            delay_min,
            delay_span,
            quantum=quantum,
        )
    return nonces, delays


def simulate_true_rtt(
    block_size: int,
    latency: float,
    G_true: float,
    delays: np.ndarray,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate true-device RTTs against the verifier's keyed schedule."""
    net_noise = rng.normal(0.0, sigma_net, size=delays.shape[0])
    gen_noise = rng.normal(0.0, sigma_gen * np.sqrt(block_size), size=delays.shape[0])
    return 2.0 * latency + G_true * block_size + delays + net_noise + gen_noise


def simulate_attacker_rtt(
    block_size: int,
    latency: float,
    G_true_est: float,
    G_attacker: float,
    guessed_delays: np.ndarray,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate an attacker that waits according to guessed delay values.

    The attacker may be faster than the true PUF but cannot wait negative time.
    ``G_true_est`` gives the attacker the benefit of a correct estimate of the
    verifier's characterised per-challenge generation term.
    """
    target_compute = G_true_est * block_size + guessed_delays
    own_compute = G_attacker * block_size
    wait = np.maximum(0.0, target_compute - own_compute)
    net_noise = rng.normal(0.0, sigma_net, size=guessed_delays.shape[0])
    gen_noise = rng.normal(0.0, sigma_gen * np.sqrt(block_size), size=guessed_delays.shape[0])
    return 2.0 * latency + own_compute + wait + net_noise + gen_noise


def timing_accept_rate(
    observed_rtt: np.ndarray,
    expected_rtt: np.ndarray,
    margins: np.ndarray,
) -> np.ndarray:
    """Return timing accept probability for each absolute timing margin."""
    return np.asarray([
        float(np.mean(np.abs(observed_rtt - expected_rtt) <= margin))
        for margin in margins
    ])


def _flat_response(response: object) -> np.ndarray:
    """Normalise a PUF response object to a flat uint8 vector."""
    if isinstance(response, tuple):
        response = response[1]
    return np.asarray(response).ravel().astype(np.uint8)


def _train_lr_models(
    puf: Arbiter,
    rng_jax: jax.Array,
    n_stages: int,
    noise_ber: float,
    n_train_crp_values: Sequence[int],
    seed: int,
) -> Tuple[Dict[int, LRAttack], jax.Array]:
    """Calibrate noisy CRPs and train LR models for response-side checks."""
    rng_jax, sk_calib, sk_train = jax.random.split(rng_jax, 3)
    calib_chall = generate_challenges(sk_calib, (5000, n_stages))
    calib = calibrate_noise_for_ber(sk_calib, puf.weight, calib_chall, target_ber=noise_ber)
    sigma_noise = float(calib.sigma)
    print(f"[exp5c] sigma_noise={sigma_noise:.4f} for BER={noise_ber}")

    max_crp = max(n_train_crp_values)
    train_chall = generate_challenges(sk_train, (max_crp, n_stages))
    rng_jax, noisy_train_resp_raw = noisy_get_response(
        rng_jax, puf.weight, train_chall, jnp.float32(sigma_noise)
    )
    train_chall_np = np.asarray(train_chall)
    noisy_train_resp = np.asarray(noisy_train_resp_raw).ravel()

    models: Dict[int, LRAttack] = {}
    for n_crp in n_train_crp_values:
        model = LRAttack(n_restarts=3, random_state=seed)
        model.fit(train_chall_np[:n_crp], noisy_train_resp[:n_crp])
        models[int(n_crp)] = model
        print(f"[exp5c] trained LR on {n_crp} CRPs")
    return models, rng_jax


def run_experiment(
    n_stages: int = 64,
    noise_ber: float = 0.04,
    block_sizes: Sequence[int] = (10, 25, 50, 100, 200, 800),
    subset_fraction: float = 0.3,
    G_true: float = 0.01,
    G_attacker: float = 0.001,
    latency: float = 5.0,
    delay_min: float = 0.75,
    delay_span: float = 8.0,
    delay_quantum: Optional[float] = None,
    sigma_net: float = 0.3,
    sigma_gen: float = 0.005,
    n_exchanges: int = 5_000,
    n_train_crp_values: Sequence[int] = (500, 2_000, 10_000, 50_000),
    margins: Optional[np.ndarray] = None,
    accuracy_threshold: float = 0.90,
    seed: int = 0,
    key: Optional[bytes] = None,
    save_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run a PRF-delay block timing experiment.

    Args:
        n_stages (int): Arbiter challenge bit-length.
        noise_ber (float): target BER for noisy training CRPs.
        block_sizes (Sequence[int]): block sizes to evaluate.
        subset_fraction (float): response subset fraction checked by verifier.
        G_true (float): characterised true-PUF per-challenge generation time.
        G_attacker (float): attacker's per-challenge inference time.
        latency (float): one-way network latency.
        delay_min (float): minimum keyed artificial delay.
        delay_span (float): keyed pseudorandom delay range above delay_min.
        delay_quantum (float, optional): quantize keyed delays to this quantum.
        sigma_net (float): network jitter standard deviation.
        sigma_gen (float): generation jitter per sqrt(block size).
        n_exchanges (int): simulated authentication exchanges per block size.
        n_train_crp_values (Sequence[int]): CRP budgets for LR attackers.
        margins (np.ndarray, optional): timing-acceptance margin sweep.
        accuracy_threshold (float): required subset accuracy for response pass.
        seed (int): RNG seed.
        key (bytes, optional): PRF key.  A fresh 256-bit key is generated if None.
        save_path (str, optional): figure destination.

    Returns:
        Dict[str, object]: timing, response, and combined accept results.
    """
    rng_np = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)
    if margins is None:
        margins = np.linspace(0.05, 6.0, 100)
    if key is None:
        key = secrets.token_bytes(32)

    rng_jax, sk_puf, sk_eval = jax.random.split(rng_jax, 3)
    puf = Arbiter(sk_puf, (1, n_stages))
    trained_models, rng_jax = _train_lr_models(
        puf, rng_jax, n_stages, noise_ber, n_train_crp_values, seed
    )

    eval_count = max(2_000, max(block_sizes))
    eval_chall_jax = generate_challenges(sk_eval, (eval_count, n_stages))
    eval_chall_np = np.asarray(eval_chall_jax)
    noiseless_resp_eval = _flat_response(puf.get_response(eval_chall_jax))

    results: Dict[str, object] = {
        "block_sizes": list(block_sizes),
        "delay_parameters": {
            "delay_min": delay_min,
            "delay_span": delay_span,
            "delay_quantum": delay_quantum,
        },
        "timing_accept": {},
        "combined_accept": {},
        "subset_accuracy": {},
        "attack_modes": ["mean_delay", "sampled_prior", "oracle_key"],
    }

    representative_delays: Optional[np.ndarray] = None

    for block_size in block_sizes:
        _, true_delays = make_prf_delays(
            key,
            int(block_size),
            n_exchanges,
            delay_min,
            delay_span,
            rng_np,
            quantum=delay_quantum,
        )
        if representative_delays is None:
            representative_delays = true_delays.copy()

        expected_rtt = 2.0 * latency + G_true * int(block_size) + true_delays
        true_rtt = simulate_true_rtt(
            int(block_size), latency, G_true, true_delays, sigma_net, sigma_gen, rng_np
        )

        mean_guess = np.full(n_exchanges, delay_min + 0.5 * delay_span, dtype=float)
        sampled_prior_guess = delay_min + rng_np.random(n_exchanges) * delay_span
        oracle_guess = true_delays.copy()

        attacker_mean_rtt = simulate_attacker_rtt(
            int(block_size), latency, G_true, G_attacker, mean_guess, sigma_net, sigma_gen, rng_np
        )
        attacker_sampled_rtt = simulate_attacker_rtt(
            int(block_size), latency, G_true, G_attacker, sampled_prior_guess, sigma_net, sigma_gen, rng_np
        )
        attacker_oracle_rtt = simulate_attacker_rtt(
            int(block_size), latency, G_true, G_attacker, oracle_guess, sigma_net, sigma_gen, rng_np
        )

        true_accept = timing_accept_rate(true_rtt, expected_rtt, margins)
        mean_accept = timing_accept_rate(attacker_mean_rtt, expected_rtt, margins)
        sampled_accept = timing_accept_rate(attacker_sampled_rtt, expected_rtt, margins)
        oracle_accept = timing_accept_rate(attacker_oracle_rtt, expected_rtt, margins)

        results["timing_accept"][int(block_size)] = {
            "true": true_accept.tolist(),
            "mean_delay": mean_accept.tolist(),
            "sampled_prior": sampled_accept.tolist(),
            "oracle_key": oracle_accept.tolist(),
        }

        n_subset = max(1, int(int(block_size) * subset_fraction))
        subset_idx = rng_np.choice(eval_count, size=min(n_subset, eval_count), replace=False)
        subset_chall = eval_chall_np[subset_idx]
        subset_resp = noiseless_resp_eval[subset_idx]

        acc_by_ncrp: Dict[int, float] = {}
        combined_by_ncrp: Dict[int, Dict[str, object]] = {}
        for n_crp, model in trained_models.items():
            subset_acc = model.accuracy(subset_chall, subset_resp)
            acc_pass = subset_acc >= accuracy_threshold
            acc_by_ncrp[int(n_crp)] = float(subset_acc)
            gate = 1.0 if acc_pass else 0.0
            combined_by_ncrp[int(n_crp)] = {
                "acc_passes": bool(acc_pass),
                "mean_delay": (mean_accept * gate).tolist(),
                "sampled_prior": (sampled_accept * gate).tolist(),
                "oracle_key": (oracle_accept * gate).tolist(),
            }
            print(
                f"[exp5c] block={block_size:4d}, n_crp={n_crp:7d}, "
                f"subset_acc={subset_acc:.4f}, passes={acc_pass}"
            )

        results["subset_accuracy"][int(block_size)] = acc_by_ncrp
        results["combined_accept"][int(block_size)] = combined_by_ncrp

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Exp 5c: Keyed PRF Delay Schedule\n"
        "A public f(x) curve is learnable; a per-exchange HMAC delay is not predictable without K.",
        fontsize=11,
    )

    rep_block = int(block_sizes[len(block_sizes) // 2])
    curves = results["timing_accept"][rep_block]
    axes[0].plot(margins, curves["true"], label=f"True PUF x={rep_block}")
    axes[0].plot(margins, curves["mean_delay"], linestyle="--", label="Mean-delay attacker")
    axes[0].plot(margins, curves["sampled_prior"], linestyle=":", label="Sampled-prior attacker")
    axes[0].plot(margins, curves["oracle_key"], linestyle="-.", label="Oracle-key attacker")
    axes[0].set_xlabel("Timing margin")
    axes[0].set_ylabel("Accept rate")
    axes[0].set_title("Timing-only accept")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    if representative_delays is not None:
        axes[1].hist(representative_delays, bins=40, alpha=0.7)
    axes[1].axvline(delay_min + 0.5 * delay_span, linestyle="--", label="Prior median")
    axes[1].set_xlabel("PRF delay")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-exchange keyed delay distribution")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    best_ncrp = int(max(n_train_crp_values))
    for block_size in block_sizes:
        combined = results["combined_accept"][int(block_size)][best_ncrp]["mean_delay"]
        axes[2].plot(margins, combined, label=f"x={block_size}")
    axes[2].set_xlabel("Timing margin")
    axes[2].set_ylabel("Combined accept rate")
    axes[2].set_title(f"Mean-delay attacker + response check, n={best_ncrp}")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp5c] Figure saved to {save_path}")
    plt.show()

    return results


if __name__ == "__main__":
    run_experiment(save_path="experiments/results/exp5c_block_timing_prf_delay.png")
