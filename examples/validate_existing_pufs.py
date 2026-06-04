"""
Validate built-in PUF models and save research-oriented plots.

Run from the repository root::

    python examples/validate_existing_pufs.py --out validation_results

The script deliberately keeps matplotlib usage outside pytest.  Tests should
assert numerical contracts; this script is for inspecting distributions,
calibrating a 4% BER noise level, and comparing bit-influence profiles.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import jax
import matplotlib.pyplot as plt
import numpy as np

from Pufs.FunctionalPuf import Arbiter, Xor, generate_challenges
from Pufs.aging import ArbiterPUF_Aging
from Pufs.randomness import DistributionSpec, sample_distribution
from Pufs.validation import (
    bit_influence,
    calibrate_noise_for_ber,
    response_balance,
    validate_aging_determinism,
    validate_distribution,
)


def _save_noise_histogram(out: Path, samples: np.ndarray) -> None:
    """Save a histogram for the generated Gaussian noise samples."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples.reshape(-1), bins=80, density=True)
    ax.set_title("Gaussian noise sample distribution")
    ax.set_xlabel("sample value")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out / "noise_distribution.png", dpi=160)
    plt.close(fig)


def _save_ber_curve(out: Path, curve: tuple[tuple[float, float], ...], target_ber: float) -> None:
    """Save the empirical BER curve used for sigma calibration."""
    sigmas = [point[0] for point in curve]
    bers = [point[1] for point in curve]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sigmas, bers, marker="o")
    ax.axhline(target_ber, linestyle="--", label=f"target BER = {target_ber:.2%}")
    ax.set_title("Empirical BER under per-read Gaussian weight noise")
    ax.set_xlabel("sigma")
    ax.set_ylabel("bit-error rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ber_calibration_curve.png", dpi=160)
    plt.close(fig)


def _save_bit_influence(out: Path, arbiter_influence: np.ndarray, xor_influence: np.ndarray) -> None:
    """Save challenge-bit influence profiles for Arbiter and XOR PUFs."""
    xs = np.arange(1, arbiter_influence.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, arbiter_influence, marker=".", label="Arbiter")
    ax.plot(xs, xor_influence, marker=".", label="3-XOR")
    ax.set_title("Challenge-bit influence")
    ax.set_xlabel("challenge bit index")
    ax.set_ylabel("Pr(response changes after bit flip)")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bit_influence.png", dpi=160)
    plt.close(fig)


def _write_summary(out: Path, rows: list[dict[str, object]]) -> None:
    """Write validation metrics as a compact CSV."""
    path = out / "validation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="base validation seed")
    parser.add_argument("--crps", type=int, default=20_000, help="challenge rows")
    parser.add_argument("--out", type=Path, default=Path("validation_results"), help="output directory")
    return parser.parse_args()


def main() -> None:
    """Run validation checks and write plots/CSV outputs."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    arb_key, xor_key, challenge_key, noise_key, sample_key, age_key = jax.random.split(key, 6)

    challenges = generate_challenges(challenge_key, (args.crps, 64))
    arbiter = Arbiter(arb_key, (1, 64))
    xor_puf = Xor(xor_key, (3, 64))

    normal_spec = DistributionSpec.normal(loc=0.0, scale=0.5)
    normal_validation = validate_distribution(
        sample_key,
        normal_spec,
        (50_000,),
        expected_mean=0.0,
        expected_std=0.5,
        mean_tolerance=0.02,
        std_tolerance=0.02,
    )
    _save_noise_histogram(args.out, np.asarray(sample_distribution(sample_key, normal_spec, (50_000,))))

    ber_result = calibrate_noise_for_ber(
        noise_key,
        arbiter.weight,
        challenges,
        target_ber=0.04,
        sigma_min=0.0,
        sigma_max=4.0,
        n_candidates=48,
        tolerance=0.01,
    )
    _save_ber_curve(args.out, ber_result.curve, ber_result.target_ber)

    arbiter_influence = bit_influence(arbiter.get_response, challenges).influence
    xor_influence = bit_influence(xor_puf.get_response, challenges).influence
    _save_bit_influence(args.out, arbiter_influence, xor_influence)

    aging_result = validate_aging_determinism(
        lambda: ArbiterPUF_Aging(arb_key, stages=64),
        age_key,
        challenges[:5_000],
        n_steps=8,
        model="additive",
        model_kwargs={"sigma_per_step": 0.05},
    )

    rows = [
        {
            "model": "Arbiter64",
            "weight_shape": tuple(int(value) for value in arbiter.weight.shape),
            "response_balance": response_balance(arbiter.get_response(challenges)),
            "bit_influence_mean": float(np.mean(arbiter_influence)),
            "bit_influence_std": float(np.std(arbiter_influence)),
            "noise_distribution_passed": normal_validation.passed,
            "calibrated_sigma_for_4pct_ber": ber_result.sigma,
            "measured_ber": ber_result.measured_ber,
            "ber_passed": ber_result.passed,
            "aging_deterministic": aging_result.passed,
        },
        {
            "model": "Xor3x64",
            "weight_shape": tuple(int(value) for value in xor_puf.weight.shape),
            "response_balance": response_balance(xor_puf.get_response(challenges)),
            "bit_influence_mean": float(np.mean(xor_influence)),
            "bit_influence_std": float(np.std(xor_influence)),
            "noise_distribution_passed": normal_validation.passed,
            "calibrated_sigma_for_4pct_ber": "",
            "measured_ber": "",
            "ber_passed": "",
            "aging_deterministic": "",
        },
    ]
    _write_summary(args.out, rows)

    print(f"Wrote validation plots and CSV to {args.out.resolve()}")
    print(f"Arbiter weight shape: {arbiter.weight.shape} for 64 challenge bits")
    print(f"Gaussian noise validation passed: {normal_validation.passed}")
    print(f"4% BER calibration: sigma={ber_result.sigma:.4f}, measured={ber_result.measured_ber:.4f}")
    print(f"Aging deterministic replay passed: {aging_result.passed}")


if __name__ == "__main__":
    main()
