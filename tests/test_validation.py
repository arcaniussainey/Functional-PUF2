"""Validation-component tests for research-facing PUF checks."""

from __future__ import annotations

import jax
import numpy as np

from Pufs.aging import ArbiterPUF_Aging
from Pufs.primitives import (
    generate_challenges,
    generate_weights,
    get_response,
    phi_from_challenges,
)
from Pufs.randomness import DistributionSpec
from Pufs.validation import (
    arbiter_bit_influence,
    bit_error_rate,
    calibrate_noise_for_ber,
    validate_aging_determinism,
    validate_distribution,
)

KEY = jax.random.PRNGKey(404)


def test_64_stage_arbiter_uses_65_coordinate_feature_vector() -> None:
    """A 64-bit challenge must be evaluated against Phi(C) with a bias term."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    weight = generate_weights(sk_w, (1, 64))
    challenges = generate_challenges(sk_c, (128, 64))
    phi = phi_from_challenges(challenges)

    assert weight.shape == (1, 65)
    assert phi.shape == (128, 65)
    np.testing.assert_array_equal(
        get_response(weight, challenges),
        (np.asarray(phi @ weight.T) > 0).astype(np.uint8),
    )


def test_distribution_validation_pins_gaussian_noise_moments() -> None:
    """Distribution validation should detect the intended normal noise scale."""
    result = validate_distribution(
        KEY,
        DistributionSpec.normal(loc=0.0, scale=0.5),
        (20_000,),
        expected_mean=0.0,
        expected_std=0.5,
        mean_tolerance=0.02,
        std_tolerance=0.02,
    )

    assert result.passed
    assert abs(result.summary.mean) < 0.02
    assert abs(result.summary.std - 0.5) < 0.02


def test_noise_calibration_can_hit_four_percent_ber() -> None:
    """Grid calibration should find a sigma close to 4% empirical BER."""
    sk_w, sk_c, sk_noise = jax.random.split(KEY, 3)
    weight = generate_weights(sk_w, (1, 64))
    challenges = generate_challenges(sk_c, (5_000, 64))

    result = calibrate_noise_for_ber(
        sk_noise,
        weight,
        challenges,
        target_ber=0.04,
        sigma_min=0.0,
        sigma_max=4.0,
        n_candidates=32,
        tolerance=0.015,
    )

    assert result.passed
    assert 0.025 <= result.measured_ber <= 0.055
    assert result.sigma > 0.0


def test_aging_validation_replays_weight_history_and_responses() -> None:
    """Aging validation should prove deterministic replay for fixed keys."""
    puf_key, age_key, challenge_key = jax.random.split(KEY, 3)
    challenges = generate_challenges(challenge_key, (256, 32))

    result = validate_aging_determinism(
        lambda: ArbiterPUF_Aging(puf_key, stages=32),
        age_key,
        challenges,
        n_steps=5,
        model="additive",
        model_kwargs={"sigma_per_step": 0.05},
    )

    assert result.passed
    assert result.n_ages == 6
    assert result.response_shape == (256,)


def test_bit_influence_matches_manual_single_bit_flip() -> None:
    """Bit-influence helper should match explicit Hamming-distance logic."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    weight = generate_weights(sk_w, (1, 16))
    challenges = generate_challenges(sk_c, (1_000, 16))
    result = arbiter_bit_influence(weight, challenges)

    baseline = get_response(weight, challenges)
    flipped = challenges.at[:, 0].multiply(-1)
    expected_first = bit_error_rate(baseline, get_response(weight, flipped))

    assert result.influence.shape == (16,)
    assert np.all((0.0 <= result.influence) & (result.influence <= 1.0))
    assert np.isclose(result.influence[0], expected_first)
    assert result.maximum > 0.0
