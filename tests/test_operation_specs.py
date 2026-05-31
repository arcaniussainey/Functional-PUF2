"""
Contracts for the immutable OperationSpec DSL.

These tests pin the architecture-level expectations: specs are static and
hashable, runtime arrays live in OperationState, built-in operation sequences can
be lowered/cached, and future developers can register custom operations without
editing the executor.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import generate_challenges, generate_weights, get_response, xor_get_response
from Pufs.operations import (
    OperationSpec,
    OperationState,
    apply_operation,
    lower_operations,
    op_add_gaussian_noise,
    op_evaluate_response,
    op_evaluate_xor_response,
    op_generate_challenges,
    op_pipeline,
    register_operation,
    registered_operations,
)
from Pufs.pipeline import PipelineState, compose, evaluate_response, evaluate_xor_response

KEY = jax.random.PRNGKey(2026)


def test_operation_specs_are_immutable_hashable_and_reject_array_params() -> None:
    """Specs should be cache-safe static data, not runtime array containers."""
    spec = op_generate_challenges(32, n_stages=16)
    assert spec.kind == "generate_challenges"
    same_spec = OperationSpec.create(
        "generate_challenges",
        n_challenges=32,
        n_stages=16,
    )
    assert hash(spec) == hash(same_spec)

    try:
        spec.kind = "other"  # type: ignore[misc]
    except Exception as exc:
        assert exc.__class__.__name__ == "FrozenInstanceError"
    else:  # pragma: no cover - this would break the architecture contract
        raise AssertionError("OperationSpec accepted mutation")

    try:
        OperationSpec.create("bad", challenge=generate_challenges(KEY, (4, 8)))
    except TypeError as exc:
        assert "arrays" in str(exc)
    else:  # pragma: no cover - this would poison the compile cache
        raise AssertionError("OperationSpec accepted runtime array parameter")


def test_lowered_xor_pipeline_matches_direct_primitive() -> None:
    """Compiled specs must preserve primitive response semantics exactly."""
    sk_w, sk_c, sk_run = jax.random.split(KEY, 3)
    weight = generate_weights(sk_w, (3, 32))
    challenge = generate_challenges(sk_c, (64, 32))

    specs = (op_evaluate_xor_response(),)
    executor = lower_operations(specs, jit=True)
    state = executor(OperationState(rng=sk_run, weight=weight, challenge=challenge))

    direct = xor_get_response(weight, challenge)
    compiled = state.response
    assert compiled is not None
    np.testing.assert_array_equal(compiled[0], direct[0])
    np.testing.assert_array_equal(compiled[1], direct[1])


def test_operation_pipeline_generates_challenges_and_evaluates_response() -> None:
    """OperationPipeline keeps the DSL readable while lowering through specs."""
    sk_w, sk_run = jax.random.split(KEY, 2)
    weight = generate_weights(sk_w, (1, 16))

    pipe = op_pipeline(op_generate_challenges(20), op_evaluate_response())
    state = pipe.run(OperationState(rng=sk_run, weight=weight), jit=True)

    assert state.challenge.shape == (20, 16)
    assert state.response.shape == (20, 1)
    np.testing.assert_array_equal(state.response, get_response(weight, state.challenge))


def test_pipeline_compiled_path_uses_specs_when_arrays_are_in_state() -> None:
    """The legacy pipeline wrapper can lower spec-backed sequences."""
    sk_w, sk_c, sk_run = jax.random.split(KEY, 3)
    weight = generate_weights(sk_w, (3, 32))
    challenge = generate_challenges(sk_c, (40, 32))

    pipeline = compose(evaluate_xor_response())
    state = pipeline.run_from(
        PipelineState(rng=sk_run, weight=weight, challenge=challenge),
        compiled=True,
    )

    direct = xor_get_response(weight, challenge)
    assert state.response is not None
    np.testing.assert_array_equal(state.response[0], direct[0])
    np.testing.assert_array_equal(state.response[1], direct[1])


def test_pipeline_binding_steps_remain_supported_but_not_lowerable() -> None:
    """Array-binding compatibility steps are explicit Python-level boundaries."""
    sk_w, sk_c, sk_run = jax.random.split(KEY, 3)
    weight = generate_weights(sk_w, (1, 8))
    challenge = generate_challenges(sk_c, (10, 8))

    from Pufs.pipeline import use_challenges, use_weights  # local import documents legacy path

    pipeline = compose(use_weights(weight), use_challenges(challenge), evaluate_response())
    assert pipeline.run(sk_run).response.shape == (10, 1)

    try:
        pipeline.run(sk_run, compiled=True)
    except ValueError as exc:
        assert "not OperationSpec-backed" in str(exc)
    else:  # pragma: no cover - lowering bound arrays would risk cache poisoning
        raise AssertionError("Compiled pipeline accepted runtime array binding steps")


def test_custom_operation_registration_extends_executor() -> None:
    """Future operations can be added by registering a handler, not editing lower()."""
    kind = "test_scale_weight"

    if kind not in registered_operations():
        def _scale_weight(state: OperationState, spec: OperationSpec) -> OperationState:
            factor = jnp.float32(spec.get("factor"))
            if state.weight is None:
                raise ValueError("test_scale_weight: weight must be set")
            return replace(state, weight=state.weight * factor)

        register_operation(kind, _scale_weight, doc="Test-only operation for registry extension.")

    sk_w, sk_c = jax.random.split(KEY, 2)
    weight = generate_weights(sk_w, (1, 8))
    challenge = generate_challenges(sk_c, (6, 8))
    specs = (
        OperationSpec.create(kind, factor=2.0),
        op_evaluate_response(),
    )

    initial_state = OperationState(rng=KEY, weight=weight, challenge=challenge)
    state = lower_operations(specs, jit=True)(initial_state)
    direct = get_response(weight * 2.0, challenge)
    np.testing.assert_array_equal(state.response, direct)


def test_apply_operation_debug_path_matches_lowered_path() -> None:
    """Single-step debug execution and lowered execution share semantics."""
    sk_w, sk_c = jax.random.split(KEY, 2)
    weight = generate_weights(sk_w, (1, 8))
    challenge = generate_challenges(sk_c, (6, 8))
    init = OperationState(rng=KEY, weight=weight, challenge=challenge)
    spec = op_evaluate_response()

    debug_state = apply_operation(spec, init)
    lowered_state = lower_operations((spec,), jit=False)(init)
    np.testing.assert_array_equal(debug_state.response, lowered_state.response)


def test_noise_spec_preserves_shape_and_determinism_for_fixed_key() -> None:
    """Stochastic specs consume state.rng deterministically under fixed inputs."""
    sk_w, sk_c, sk_run = jax.random.split(KEY, 3)
    weight = generate_weights(sk_w, (1, 16))
    challenge = generate_challenges(sk_c, (20, 16))
    specs = (op_add_gaussian_noise(0.05), op_evaluate_response())
    executor = lower_operations(specs, jit=True)

    a = executor(OperationState(rng=sk_run, weight=weight, challenge=challenge))
    b = executor(OperationState(rng=sk_run, weight=weight, challenge=challenge))

    assert a.weight.shape == weight.shape
    assert a.response.shape == (20, 1)
    np.testing.assert_array_equal(a.weight, b.weight)
    np.testing.assert_array_equal(a.response, b.response)
