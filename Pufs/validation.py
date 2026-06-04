"""
Reusable validation helpers for PUF experiments.

The functions in this module are intentionally NumPy-friendly wrappers around
public PUF primitives.  They are meant for tests, notebooks, and scripts that
need repeatable numerical checks without embedding plotting code in pytest
files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.aging import AgingPUF
from Pufs.primitives import (
    Challenge,
    PRNGKey,
    Response,
    Weight,
    generate_challenges,
    get_response,
    noisy_get_response,
)
from Pufs.randomness import DistributionSpec, sample_distribution

ArrayLike = np.ndarray | Sequence[int] | Sequence[float] | jax.Array
ResponseFn = Callable[[Challenge], Response]
AgingFactory = Callable[[], AgingPUF]


@dataclass(frozen=True)
class DistributionSummary:
    """Compact empirical summary of a sampled distribution."""

    count: int
    mean: float
    std: float
    minimum: float
    q01: float
    q05: float
    q50: float
    q95: float
    q99: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the summary as a plain dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "q01": self.q01,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
            "q99": self.q99,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class DistributionValidation:
    """Result of validating sampled noise against expected moments."""

    spec: DistributionSpec
    shape: tuple[int, ...]
    summary: DistributionSummary
    expected_mean: Optional[float]
    expected_std: Optional[float]
    mean_error: Optional[float]
    std_error: Optional[float]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """Return serialisable validation fields."""
        return {
            "family": self.spec.family,
            "shape": self.shape,
            "summary": self.summary.as_dict(),
            "expected_mean": self.expected_mean,
            "expected_std": self.expected_std,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BERCalibrationResult:
    """Empirical noise calibration result for a target bit-error rate."""

    target_ber: float
    sigma: float
    measured_ber: float
    tolerance: float
    n_challenges: int
    n_candidates: int
    passed: bool
    curve: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, object]:
        """Return serialisable calibration fields."""
        return {
            "target_ber": self.target_ber,
            "sigma": self.sigma,
            "measured_ber": self.measured_ber,
            "tolerance": self.tolerance,
            "n_challenges": self.n_challenges,
            "n_candidates": self.n_candidates,
            "passed": self.passed,
            "curve": self.curve,
        }


@dataclass(frozen=True)
class AgingDeterminismResult:
    """Result of replaying an aging model with the same initial conditions."""

    weight_history_equal: bool
    response_equal: bool
    n_ages: int
    response_shape: tuple[int, ...]

    @property
    def passed(self) -> bool:
        """Return True when weights and responses replay exactly."""
        return self.weight_history_equal and self.response_equal

    def as_dict(self) -> dict[str, object]:
        """Return serialisable determinism fields."""
        return {
            "weight_history_equal": self.weight_history_equal,
            "response_equal": self.response_equal,
            "n_ages": self.n_ages,
            "response_shape": self.response_shape,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BitInfluenceResult:
    """Per-challenge-bit response influence summary."""

    influence: np.ndarray
    mean: float
    std: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, object]:
        """Return serialisable influence fields."""
        return {
            "influence": self.influence.tolist(),
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def _flatten_response(response: object) -> np.ndarray:
    """Return a flat binary vector from Arbiter or XOR-style response output."""
    if isinstance(response, tuple):
        response = response[-1]
    return np.asarray(response, dtype=np.uint8).reshape(-1)


def distribution_summary(values: ArrayLike) -> DistributionSummary:
    """Return common empirical distribution statistics for ``values``."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("values must not be empty.")
    q01, q05, q50, q95, q99 = np.quantile(array, [0.01, 0.05, 0.50, 0.95, 0.99])
    return DistributionSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        std=float(np.std(array)),
        minimum=float(np.min(array)),
        q01=float(q01),
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
        q99=float(q99),
        maximum=float(np.max(array)),
    )


def validate_distribution(
    rng: PRNGKey,
    spec: DistributionSpec,
    shape: Sequence[int],
    *,
    expected_mean: Optional[float] = None,
    expected_std: Optional[float] = None,
    mean_tolerance: float = 0.02,
    std_tolerance: float = 0.02,
) -> DistributionValidation:
    """
    Sample a configured distribution and validate its empirical moments.

    Args:
        rng (PRNGKey): sampling key.
        spec (DistributionSpec): distribution to sample.
        shape (Sequence[int]): sample shape.
        expected_mean (Optional[float]): expected empirical mean, if checked.
        expected_std (Optional[float]): expected empirical standard deviation, if checked.
        mean_tolerance (float): allowed absolute mean error.
        std_tolerance (float): allowed absolute std error.

    Returns:
        DistributionValidation: sampled summary and pass/fail status.
    """
    out_shape = tuple(int(value) for value in shape)
    samples = sample_distribution(rng, spec, out_shape)
    summary = distribution_summary(samples)

    mean_error = None if expected_mean is None else abs(summary.mean - expected_mean)
    std_error = None if expected_std is None else abs(summary.std - expected_std)
    passed = True
    if mean_error is not None:
        passed = passed and mean_error <= mean_tolerance
    if std_error is not None:
        passed = passed and std_error <= std_tolerance

    return DistributionValidation(
        spec=spec,
        shape=out_shape,
        summary=summary,
        expected_mean=expected_mean,
        expected_std=expected_std,
        mean_error=mean_error,
        std_error=std_error,
        passed=bool(passed),
    )


def bit_error_rate(reference: ArrayLike, measured: ArrayLike) -> float:
    """Return the normalised Hamming distance between two response vectors."""
    ref = np.asarray(reference, dtype=np.uint8).reshape(-1)
    got = np.asarray(measured, dtype=np.uint8).reshape(-1)
    if ref.shape != got.shape:
        raise ValueError("reference and measured responses must have matching shapes.")
    return float(np.mean(ref != got))


def noise_ber(
    rng: PRNGKey,
    weight: Weight,
    challenges: Challenge,
    sigma: float,
) -> float:
    """Measure BER induced by one Gaussian noisy-response evaluation."""
    clean = get_response(weight, challenges)
    _next_rng, noisy = noisy_get_response(rng, weight, challenges, jnp.float32(sigma))
    return bit_error_rate(clean, noisy)


def calibrate_noise_for_ber(
    rng: PRNGKey,
    weight: Weight,
    challenges: Challenge,
    *,
    target_ber: float = 0.04,
    sigma_min: float = 0.0,
    sigma_max: float = 8.0,
    n_candidates: int = 48,
    tolerance: float = 0.01,
) -> BERCalibrationResult:
    """
    Find a Gaussian noise sigma that empirically approaches a target BER.

    This is a validation calibration, not a symbolic guarantee.  Increase the
    number of challenges to tighten the empirical confidence of the measured
    BER for a particular experiment.
    """
    if not 0.0 <= target_ber <= 0.5:
        raise ValueError("target_ber must be in [0, 0.5].")
    if sigma_min < 0.0 or sigma_max <= sigma_min:
        raise ValueError("sigma bounds must satisfy 0 <= sigma_min < sigma_max.")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2.")

    sigmas = np.linspace(float(sigma_min), float(sigma_max), int(n_candidates))
    subkeys = jax.random.split(rng, int(n_candidates))
    curve = tuple(
        (float(sigma), noise_ber(subkey, weight, challenges, float(sigma)))
        for subkey, sigma in zip(subkeys, sigmas)
    )
    sigma, measured = min(curve, key=lambda item: abs(item[1] - target_ber))
    return BERCalibrationResult(
        target_ber=float(target_ber),
        sigma=float(sigma),
        measured_ber=float(measured),
        tolerance=float(tolerance),
        n_challenges=int(challenges.shape[0]),
        n_candidates=int(n_candidates),
        passed=abs(float(measured) - float(target_ber)) <= float(tolerance),
        curve=curve,
    )


def bit_influence(response_fn: ResponseFn, challenges: Challenge) -> BitInfluenceResult:
    """
    Estimate response-change rate for flipping each challenge bit.

    Args:
        response_fn (ResponseFn): callable that accepts a challenge matrix and
            returns an Arbiter response or XOR response tuple.
        challenges (Challenge): challenge matrix, shape (N, n).\n
    Returns:
        BitInfluenceResult: one influence value per challenge bit.
    """
    baseline = _flatten_response(response_fn(challenges))
    values = []
    for bit_index in range(int(challenges.shape[1])):
        flipped = challenges.at[:, bit_index].multiply(-1)
        values.append(bit_error_rate(baseline, _flatten_response(response_fn(flipped))))
    influence = np.asarray(values, dtype=np.float64)
    return BitInfluenceResult(
        influence=influence,
        mean=float(np.mean(influence)),
        std=float(np.std(influence)),
        minimum=float(np.min(influence)),
        maximum=float(np.max(influence)),
    )


def arbiter_bit_influence(weight: Weight, challenges: Challenge) -> BitInfluenceResult:
    """Estimate bit influence for a paper-correct Arbiter weight matrix."""
    return bit_influence(lambda chall: get_response(weight, chall), challenges)


def validate_aging_determinism(
    factory: AgingFactory,
    age_rng: PRNGKey,
    challenges: Challenge,
    *,
    n_steps: int,
    model: str = "additive",
    model_kwargs: Optional[Mapping[str, float]] = None,
) -> AgingDeterminismResult:
    """
    Replay an aging PUF construction twice and compare histories/responses.

    ``factory`` must construct a fresh PUF with identical initial conditions on
    every call.  The same ``age_rng`` is then used for both aging runs.
    """
    kwargs = dict(model_kwargs or {})
    first = factory().add_aging(age_rng, n_steps=n_steps, model=model, **kwargs)
    second = factory().add_aging(age_rng, n_steps=n_steps, model=model, **kwargs)

    first_response = _flatten_response(first.get_response(challenges))
    second_response = _flatten_response(second.get_response(challenges))
    return AgingDeterminismResult(
        weight_history_equal=bool(jnp.array_equal(first.weight_history, second.weight_history)),
        response_equal=bool(np.array_equal(first_response, second_response)),
        n_ages=int(first.weight_history.shape[0]),
        response_shape=tuple(int(value) for value in np.asarray(first_response).shape),
    )


def response_balance(response: object) -> float:
    """Return the fraction of one-responses for Arbiter or XOR output."""
    return float(np.mean(_flatten_response(response)))


def make_challenges(seed: int, n_challenges: int, n_stages: int) -> Challenge:
    """Generate deterministic challenges for validation scripts."""
    return generate_challenges(jax.random.PRNGKey(seed), (int(n_challenges), int(n_stages)))
