"""Regression tests for aging, feed-forward, statistics, and new pipeline ops."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.aging import ArbiterPUF_Aging, XorPUF_Aging, drift_additive
from Pufs.feedforward import FF_Arbiter, FF_Arbiter_Expression, FF_Arbiter_Symbolic, FF_XOR
from Pufs.feedforward_core import required_xor_weight_rows
from Pufs.pipeline import (
    compose,
    evaluate_feedforward_response,
    evaluate_feedforward_xor_response,
    generate_challenges,
    generate_weights,
)
from Pufs.primitives import generate_challenges as generate_raw_challenges
from Pufs.statistics import bit_aliasing, inter_distance, reliability, summary, uniformity


def test_aging_pufs_keep_expected_shapes() -> None:
    """Aging classes preserve standard response layout conventions."""
    key = jax.random.PRNGKey(0)
    puf_key, age_key, challenge_key = jax.random.split(key, 3)
    challenges = generate_raw_challenges(challenge_key, (32, 16))

    arbiter = ArbiterPUF_Aging(puf_key, stages=16)
    arbiter.add_aging(age_key, n_steps=4)
    assert arbiter.get_response(challenges, age_idx=0).shape == (32, 1)
    assert arbiter.response_all_ages(challenges).shape == (5, 32, 1)

    xor_puf = XorPUF_Aging(puf_key, k=3, stages=16)
    xor_puf.add_aging(age_key, n_steps=4)
    individual, xor_response = xor_puf.get_response(challenges)
    assert individual.shape == (32, 3)
    assert xor_response.shape == (32,)
    assert xor_puf.response_all_ages(challenges).shape == (5, 32)


def test_drift_additive_is_deterministic_for_seed() -> None:
    """Aging drift is deterministic when the same PRNG key is reused."""
    key = jax.random.PRNGKey(5)
    weight = jnp.ones((2, 4), dtype=jnp.float32)
    first = drift_additive(key, weight, n_steps=3, sigma_per_step=0.1)
    second = drift_additive(key, weight, n_steps=3, sigma_per_step=0.1)
    assert jnp.array_equal(first, second)


def test_feedforward_variants_agree_for_non_cascading_loop() -> None:
    """Classical, symbolic, and expression FF variants share public semantics."""
    challenges = generate_raw_challenges(jax.random.PRNGKey(1), (64, 16))
    loops = [(2, 12)]
    classical = FF_Arbiter(0, 16, loops).get_response(challenges)
    symbolic = FF_Arbiter_Symbolic(0, 16, loops).get_response(challenges)
    expression = FF_Arbiter_Expression(0, 16, loops).get_response(challenges)
    assert jnp.array_equal(classical, symbolic)
    assert jnp.array_equal(classical, expression)


def test_feedforward_xor_shapes() -> None:
    """Feed-forward XOR returns individual and reduced responses."""
    challenges = generate_raw_challenges(jax.random.PRNGKey(2), (64, 16))
    puf = FF_XOR(0, 16, [[(2, 12)], [(1, 10)], [(3, 14)]])
    individual, xor_response = puf.get_response(challenges)
    assert individual.shape == (64, 3)
    assert xor_response.shape == (64,)


def test_pipeline_feedforward_ops_are_deterministic() -> None:
    """Operation-spec feed-forward pipeline is deterministic for a fixed seed."""
    loop_specs = (((2, 12),), ((1, 10),))
    rows = required_xor_weight_rows(loop_specs)
    pipeline = compose(
        generate_weights(n_stages=16, k=rows),
        generate_challenges(n_challenges=64),
        evaluate_feedforward_xor_response(loop_specs),
    )
    first = pipeline.run(jax.random.PRNGKey(9), compiled=True).response[1]
    second = pipeline.run(jax.random.PRNGKey(9), compiled=True).response[1]
    assert jnp.array_equal(first, second)


def test_single_feedforward_pipeline_op_shape() -> None:
    """Single-component feed-forward pipeline op returns ``(N, 1)``."""
    state = compose(
        generate_weights(n_stages=16, k=2),
        generate_challenges(n_challenges=32),
        evaluate_feedforward_response([(2, 12)]),
    ).run(jax.random.PRNGKey(10), compiled=True)
    assert state.response.shape == (32, 1)


def test_statistics_metrics() -> None:
    """Statistics helpers return expected values for a known response matrix."""
    responses = np.asarray([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
    assert uniformity(responses[0]) == 0.5
    assert inter_distance(responses) == 0.5
    assert reliability(responses[0], responses[0]) == 1.0
    assert np.allclose(bit_aliasing(responses), np.asarray([0.5, 1.0, 0.0, 0.5]))
    assert summary(responses)["uniformity"] == 0.5
