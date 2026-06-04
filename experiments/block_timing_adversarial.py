"""
challenge-block timing with a joint timing-model attacker.

This is the adversarial counterpart to ``challenge_block_timing.py``.  The base
experiment assumes a weak attacker that estimates a single effective generation
constant from one-CRP observations and extrapolates poorly.  Here the attacker
is stronger: it observes block exchanges at several block sizes and jointly
fits the verifier timing model

    R(x) - 2L = G*x + alpha*x^beta + gamma + noise.

The fit is nonlinear only in ``beta``.  For each beta on a grid, the remaining
parameters ``G``, ``alpha``, and ``gamma`` are estimated by ordinary least
squares; the beta with the lowest residual error is retained.  This avoids a
SciPy dependency while still giving the attacker a meaningful joint estimate of
both the per-challenge generation term and the secret delay function.

The result is a more conservative security experiment: timing protection is
credited only to residual uncertainty after the attacker has learned the timing
curve, while response security is still determined by the ML model's CRP-based
prediction accuracy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from Pufs.FunctionalPuf import Arbiter, generate_challenges, noisy_get_response
from Pufs.validation import calibrate_noise_for_ber
from Attack.lr_attack import LRAttack


@dataclass(frozen=True)
class JointTimingEstimate:
    """Parameters learned by the joint timing attacker."""

    G: float
    alpha: float
    beta: float
    gamma: float
    rmse: float
    n_observations: int


def f_delay(x: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Secret verifier delay function.

    Args:
        x (np.ndarray): block-size array.
        alpha (float): scale of the nonlinear component.
        beta (float): shape exponent.
        gamma (float): fixed offset.

    Returns:
        np.ndarray: ``alpha*x**beta + gamma``.
    """
    return alpha * np.power(x.astype(float), beta) + gamma


def timing_curve(
    x: np.ndarray,
    latency: float,
    G: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Return the verifier's expected round-trip time for each block size."""
    return 2.0 * latency + G * x.astype(float) + f_delay(x, alpha, beta, gamma)


def _simulate_true_rtt(
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
    """Simulate true-PUF round-trip times for one block size."""
    x = np.asarray([block_size], dtype=float)
    expected = float(timing_curve(x, latency, G, alpha, beta, gamma)[0])
    net_noise = rng.normal(0.0, sigma_net, size=n)
    gen_noise = rng.normal(0.0, sigma_gen * np.sqrt(block_size), size=n)
    return expected + net_noise + gen_noise


def _simulate_attacker_rtt(
    n: int,
    block_size: int,
    latency: float,
    G_attacker: float,
    estimated_total_compute: float,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate attacker RTT after it intentionally delays to match the estimate.

    ``estimated_total_compute`` is the attacker's learned value for
    ``G_true*x + f(x)``.  The attacker's own computation takes
    ``G_attacker*x``; any remaining nonnegative time is added as intentional
    waiting.  If the attacker is slower than the estimate, it cannot wait
    backwards and therefore sends as soon as it has computed responses.
    """
    own_compute = G_attacker * block_size
    wait_time = max(0.0, estimated_total_compute - own_compute)
    net_noise = rng.normal(0.0, sigma_net, size=n)
    gen_noise = rng.normal(0.0, sigma_gen * np.sqrt(block_size), size=n)
    return 2.0 * latency + own_compute + wait_time + net_noise + gen_noise


def _make_timing_observations(
    calibration_block_sizes: Sequence[int],
    observations_per_size: int,
    latency: float,
    G_true: float,
    alpha: float,
    beta: float,
    gamma: float,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect synthetic adversarial timing observations across block sizes."""
    x_values = []
    rtt_values = []
    for block_size in calibration_block_sizes:
        rtts = _simulate_true_rtt(
            observations_per_size,
            int(block_size),
            latency,
            G_true,
            alpha,
            beta,
            gamma,
            sigma_net,
            sigma_gen,
            rng,
        )
        x_values.append(np.full(observations_per_size, float(block_size)))
        rtt_values.append(rtts)
    return np.concatenate(x_values), np.concatenate(rtt_values)


def fit_joint_timing_model(
    block_sizes: np.ndarray,
    observed_rtts: np.ndarray,
    latency: float,
    beta_grid: Optional[np.ndarray] = None,
    nonnegative: bool = True,
) -> JointTimingEstimate:
    """
    Estimate ``G``, ``alpha``, ``beta``, and ``gamma`` jointly from RTT data.

    Args:
        block_sizes (np.ndarray): observed block size for each RTT sample.
        observed_rtts (np.ndarray): observed round-trip times.
        latency (float): known one-way latency.
        beta_grid (np.ndarray, optional): beta candidates.  Defaults to a dense
            grid over ``[0.1, 1.5]``.
        nonnegative (bool): reject fits with negative ``G``, ``alpha``, or
            ``gamma``.  This encodes the physical assumption that generation
            time and deliberate delay are nonnegative.

    Returns:
        JointTimingEstimate: best grid-search/least-squares estimate.
    """
    x = np.asarray(block_sizes, dtype=float).ravel()
    y = np.asarray(observed_rtts, dtype=float).ravel() - 2.0 * latency
    if x.shape[0] != y.shape[0]:
        raise ValueError("block_sizes and observed_rtts must have equal length.")
    if beta_grid is None:
        beta_grid = np.linspace(0.1, 1.5, 281)

    best: Optional[JointTimingEstimate] = None
    for beta_candidate in beta_grid:
        design = np.column_stack([x, np.power(x, beta_candidate), np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        G_hat, alpha_hat, gamma_hat = [float(v) for v in coef]
        if nonnegative and (G_hat < 0.0 or alpha_hat < 0.0 or gamma_hat < 0.0):
            continue
        residual = y - design @ coef
        rmse = float(np.sqrt(np.mean(residual**2)))
        candidate = JointTimingEstimate(
            G=G_hat,
            alpha=alpha_hat,
            beta=float(beta_candidate),
            gamma=gamma_hat,
            rmse=rmse,
            n_observations=int(x.shape[0]),
        )
        if best is None or candidate.rmse < best.rmse:
            best = candidate

    if best is None:
        return fit_joint_timing_model(
            block_sizes, observed_rtts, latency, beta_grid=beta_grid, nonnegative=False
        )
    return best


def _flat_response(response: object) -> np.ndarray:
    """Normalise a PUF response object to a flat ``uint8`` vector."""
    if isinstance(response, tuple):
        response = response[1]
    return np.asarray(response).ravel().astype(np.uint8)


def run_experiment(
    n_stages: int = 64,
    noise_ber: float = 0.04,
    block_sizes: Sequence[int] = (10, 25, 50, 100, 200, 800),
    calibration_block_sizes: Sequence[int] = (1, 5, 10, 25, 50, 100, 200, 800),
    observations_per_size: int = 100,
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
    n_train_crp_values: Sequence[int] = (500, 2_000, 10_000, 50_000),
    margins: Optional[np.ndarray] = None,
    accuracy_threshold: float = 0.90,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run the adversarial joint-timing challenge-block experiment.

    Args:
        n_stages (int): Arbiter challenge bit-length.
        noise_ber (float): target BER for noisy training CRPs.
        block_sizes (Sequence[int]): authentication block sizes to evaluate.
        calibration_block_sizes (Sequence[int]): block sizes the attacker can
            observe while fitting the timing model.
        observations_per_size (int): timing samples per calibration block size.
        subset_fraction (float): fraction of responses checked by verifier.
        G_true (float): true per-challenge PUF generation time.
        G_attacker (float): attacker's per-challenge inference time.
        latency (float): one-way network latency assumed known.
        alpha, beta, gamma (float): true secret delay parameters.
        sigma_net (float): network jitter standard deviation.
        sigma_gen (float): generation jitter per sqrt(block size).
        n_exchanges (int): simulated authentication exchanges per block size.
        n_train_crp_values (Sequence[int]): CRP budgets for LR models.
        margins (np.ndarray, optional): timing-acceptance margin sweep.
        accuracy_threshold (float): required subset accuracy for response pass.
        seed (int): RNG seed.
        save_path (str, optional): figure destination.

    Returns:
        Dict containing fitted timing parameters, timing accept curves, response
        accuracies, and combined accept curves.
    """
    rng_np = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)
    if margins is None:
        margins = np.linspace(0.1, 6.0, 80)

    obs_x, obs_rtt = _make_timing_observations(
        calibration_block_sizes,
        observations_per_size,
        latency,
        G_true,
        alpha,
        beta,
        gamma,
        sigma_net,
        sigma_gen,
        rng_np,
    )
    estimate = fit_joint_timing_model(obs_x, obs_rtt, latency)
    print(
        "[exp5b] joint timing estimate: "
        f"G={estimate.G:.5f}, alpha={estimate.alpha:.5f}, "
        f"beta={estimate.beta:.3f}, gamma={estimate.gamma:.5f}, rmse={estimate.rmse:.4f}"
    )

    rng_jax, sk_puf, sk_calib, sk_eval, sk_train = jax.random.split(rng_jax, 5)
    puf = Arbiter(sk_puf, (1, n_stages))

    calib_chall = generate_challenges(sk_calib, (5000, n_stages))
    calib = calibrate_noise_for_ber(sk_calib, puf.weight, calib_chall, target_ber=noise_ber)
    sigma_noise = float(calib.sigma)
    print(f"[exp5b] sigma_noise={sigma_noise:.4f} for BER={noise_ber}")

    max_crp = max(n_train_crp_values)
    train_chall = generate_challenges(sk_train, (max_crp, n_stages))
    rng_jax, noisy_train_resp_raw = noisy_get_response(
        rng_jax, puf.weight, train_chall, jnp.float32(sigma_noise)
    )
    noisy_train_resp = np.asarray(noisy_train_resp_raw).ravel()
    train_chall_np = np.asarray(train_chall)

    trained_models: Dict[int, LRAttack] = {}
    for n_crp in n_train_crp_values:
        model = LRAttack(n_restarts=3, random_state=seed)
        model.fit(train_chall_np[:n_crp], noisy_train_resp[:n_crp])
        trained_models[int(n_crp)] = model
        print(f"[exp5b] trained LR on {n_crp} CRPs")

    eval_count = max(2_000, max(block_sizes))
    eval_chall_jax = generate_challenges(sk_eval, (eval_count, n_stages))
    eval_chall_np = np.asarray(eval_chall_jax)
    noiseless_resp_eval = _flat_response(puf.get_response(eval_chall_jax))

    results: Dict[str, object] = {
        "block_sizes": list(block_sizes),
        "calibration_block_sizes": list(calibration_block_sizes),
        "timing_estimate": estimate.__dict__,
        "true_timing_parameters": {
            "G": G_true,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
        "timing_accept": {},
        "combined_accept": {},
        "subset_accuracy": {},
    }

    for block_size in block_sizes:
        x = np.asarray([block_size], dtype=float)
        expected = float(timing_curve(x, latency, G_true, alpha, beta, gamma)[0])
        estimated_total_compute = float(
            estimate.G * block_size
            + f_delay(x, estimate.alpha, estimate.beta, estimate.gamma)[0]
        )
        print(
            f"\n[exp5b] block={block_size}, true_expected={expected:.3f}, "
            f"attacker_compute_est={estimated_total_compute:.3f}"
        )

        true_rtt = _simulate_true_rtt(
            n_exchanges,
            int(block_size),
            latency,
            G_true,
            alpha,
            beta,
            gamma,
            sigma_net,
            sigma_gen,
            rng_np,
        )
        attacker_rtt = _simulate_attacker_rtt(
            n_exchanges,
            int(block_size),
            latency,
            G_attacker,
            estimated_total_compute,
            sigma_net,
            sigma_gen,
            rng_np,
        )

        true_accept = np.asarray([
            float(np.mean(np.abs(true_rtt - expected) <= margin)) for margin in margins
        ])
        attacker_accept = np.asarray([
            float(np.mean(np.abs(attacker_rtt - expected) <= margin)) for margin in margins
        ])
        results["timing_accept"][int(block_size)] = {
            "true": true_accept.tolist(),
            "joint_attacker": attacker_accept.tolist(),
        }

        n_subset = max(1, int(block_size * subset_fraction))
        subset_idx = rng_np.choice(eval_count, size=min(n_subset, eval_count), replace=False)
        subset_chall = eval_chall_np[subset_idx]
        subset_resp = noiseless_resp_eval[subset_idx]

        acc_by_ncrp: Dict[int, float] = {}
        combined_by_ncrp: Dict[int, Dict[str, object]] = {}
        for n_crp, model in trained_models.items():
            subset_acc = model.accuracy(subset_chall, subset_resp)
            acc_pass = subset_acc >= accuracy_threshold
            acc_by_ncrp[n_crp] = subset_acc
            combined_by_ncrp[n_crp] = {
                "combined_accept": (attacker_accept * (1.0 if acc_pass else 0.0)).tolist(),
                "acc_passes": bool(acc_pass),
            }
            print(
                f"  [exp5b] n_crp={n_crp:7d} subset_acc={subset_acc:.4f} "
                f"passes={acc_pass}"
            )

        results["subset_accuracy"][int(block_size)] = acc_by_ncrp
        results["combined_accept"][int(block_size)] = combined_by_ncrp

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Exp 5b: Joint Timing-Model Attacker\n"
        f"true f(x)={alpha}*x^{beta}+{gamma}; fitted beta={estimate.beta:.3f}, rmse={estimate.rmse:.3f}",
        fontsize=11,
    )

    x_rep = block_sizes[len(block_sizes) // 2]
    axes[0].plot(margins, results["timing_accept"][x_rep]["true"], label=f"True PUF x={x_rep}")
    axes[0].plot(
        margins,
        results["timing_accept"][x_rep]["joint_attacker"],
        linestyle="--",
        label=f"Joint attacker x={x_rep}",
    )
    axes[0].set_xlabel("Timing margin")
    axes[0].set_ylabel("Accept rate")
    axes[0].set_title("Timing-only accept")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(obs_x, obs_rtt - 2.0 * latency, s=8, alpha=0.2, label="Observed RTT - 2L")
    x_dense = np.linspace(min(calibration_block_sizes), max(calibration_block_sizes), 200)
    true_compute = G_true * x_dense + f_delay(x_dense, alpha, beta, gamma)
    fit_compute = estimate.G * x_dense + f_delay(x_dense, estimate.alpha, estimate.beta, estimate.gamma)
    axes[1].plot(x_dense, true_compute, label="True G*x + f(x)")
    axes[1].plot(x_dense, fit_compute, linestyle="--", label="Joint fit")
    axes[1].set_xlabel("Block size x")
    axes[1].set_ylabel("Compute/delay component")
    axes[1].set_title("Joint estimate of G and f(x)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    best_ncrp = max(n_train_crp_values)
    for block_size in block_sizes:
        combined = results["combined_accept"][int(block_size)][int(best_ncrp)]["combined_accept"]
        axes[2].plot(margins, combined, label=f"x={block_size}")
    axes[2].set_xlabel("Timing margin")
    axes[2].set_ylabel("Combined accept rate")
    axes[2].set_title(f"Timing + accuracy, best LR n={best_ncrp}")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp5b] Figure saved to {save_path}")
    plt.show()

    return results


if __name__ == "__main__":
    run_experiment(save_path="experiments/results/exp5b_block_timing_adversarial.png")
