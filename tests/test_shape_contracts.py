"""
Shape-contract tests for Functional-PUF primitives and thin class wrappers.

These tests keep shape expectations outside the production hot path.  To add a
new primitive or composed operation, add one table entry here instead of
sprinkling runtime asserts through numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import (
    Arbiter,
    Xor,
    generate_challenges,
    generate_weights,
    get_delta_response,
    get_response,
    noisy_get_response,
    noisy_xor_get_response,
    xor_get_response,
)
from Pufs.pipeline import (
    PipelineState,
    compose,
    evaluate_response,
    evaluate_xor_response,
    generate_challenges as pipe_generate_challenges,
    use_challenges,
    use_weights,
)

KEY = jax.random.PRNGKey(123)


@dataclass(frozen=True)
class ShapeCase:
    """One table-driven shape contract."""

    name: str
    fn: Callable[[], object]
    expected: object


def _shape_tree(value: object) -> object:
    """Return the shape tree for arrays or nested tuples/lists of arrays."""
    if isinstance(value, tuple):
        return tuple(_shape_tree(v) for v in value)
    if isinstance(value, list):
        return [_shape_tree(v) for v in value]
    return value.shape  # type: ignore[union-attr]


def test_core_shape_contracts() -> None:
    """Pin public primitive and class-wrapper output shapes."""
    sk_w, sk_c, sk_n = jax.random.split(KEY, 3)
    w1 = generate_weights(sk_w, (1, 32))
    w3 = generate_weights(sk_w, (3, 32))
    c = generate_challenges(sk_c, (100, 32))
    xor_puf = Xor(sk_w, (3, 32))
    arb = Arbiter(sk_w, (1, 32))
    sigma = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)

    cases = [
        ShapeCase("generate_weights single", lambda: w1, (1, 32)),
        ShapeCase("generate_weights xor", lambda: w3, (3, 32)),
        ShapeCase("generate_challenges", lambda: c, (100, 32)),
        ShapeCase("get_response single", lambda: get_response(w1, c), (100, 1)),
        ShapeCase("get_response xor rows", lambda: get_response(w3, c), (100, 3)),
        ShapeCase("get_delta_response", lambda: get_delta_response(w3, c), (100, 3)),
        ShapeCase("xor_get_response", lambda: xor_get_response(w3, c), ((100, 3), (100,))),
        ShapeCase(
            "noisy_get_response",
            lambda: noisy_get_response(sk_n, w1, c, jnp.float32(0.1)),
            ((2,), (100, 1)),
        ),
        ShapeCase(
            "noisy_xor_get_response",
            lambda: noisy_xor_get_response(sk_n, w3, c, sigma),
            ((2,), (100, 1)),
        ),
        ShapeCase("Arbiter.__call__", lambda: arb(c), (100, 1)),
        ShapeCase("Xor.__call__", lambda: xor_puf(c), ((100, 3), (100,))),
    ]

    for case in cases:
        assert _shape_tree(case.fn()) == case.expected, case.name


def test_response_dtype_and_domain_contracts() -> None:
    """Pin binary response domains without runtime asserts in production code."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    w = generate_weights(sk_w, (3, 32))
    c = generate_challenges(sk_c, (100, 32))
    individual, xored = xor_get_response(w, c)

    assert individual.dtype == jnp.uint8
    assert xored.dtype == jnp.uint8
    assert set(np.unique(np.array(individual))).issubset({0, 1})
    assert set(np.unique(np.array(xored))).issubset({0, 1})


def test_pipeline_is_immutable_and_shape_correct() -> None:
    """Pipeline composition should not mutate existing pipeline objects."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    w = generate_weights(sk_w, (3, 32))
    c = generate_challenges(sk_c, (100, 32))

    base = compose(use_weights(w))
    extended = base.add(use_challenges(c)).add(evaluate_xor_response())

    assert len(base.steps) == 1
    assert len(extended.steps) == 3

    state = extended.run(KEY)
    assert _shape_tree(state.response) == ((100, 3), (100,))


def test_pipeline_state_is_immutable() -> None:
    """PipelineState is intentionally frozen to prevent accidental side effects."""
    state = PipelineState(rng=KEY)
    try:
        state.challenge = generate_challenges(KEY, (1, 8))  # type: ignore[misc]
    except Exception as exc:  # dataclasses.FrozenInstanceError without importing internals
        assert exc.__class__.__name__ == "FrozenInstanceError"
    else:  # pragma: no cover - reaching this would violate the contract
        raise AssertionError("PipelineState accepted mutation")


def test_arbiter_pipeline_response_shape() -> None:
    """Single-arbiter pipeline evaluation keeps the canonical (N, 1) layout."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    w = generate_weights(sk_w, (1, 32))
    c = generate_challenges(sk_c, (100, 32))
    pipeline = compose(use_weights(w), use_challenges(c), evaluate_response())
    assert pipeline.run(KEY).response.shape == (100, 1)


def test_generated_challenge_pipeline_shape() -> None:
    """Challenge-generation pipeline reports the same shape as the primitive."""
    pipeline = compose(pipe_generate_challenges(64, n_stages=16))
    assert pipeline.run(KEY).challenge.shape == (64, 16)
