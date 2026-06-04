"""
Compatibility pipeline DSL for PUF operations.

A PUFPipeline describes a sequence of labelled steps applied left-to-right.
Each step transforms the pipeline's running state, which carries:

    rng       -- JAX PRNG key threaded through stochastic steps
    weight    -- the current weight matrix  (may be None before generation)
    challenge -- the current challenge matrix (may be None before generation)
    response  -- the current response (may be None)

Design note
-----------
The pipeline object itself is immutable.  Built-in spec-backed steps can be
lowered with ``pipeline.lower()`` or executed via ``run(..., compiled=True)``.
Steps that capture runtime arrays, such as ``use_weights(weight)``, intentionally
remain Python-level binding steps; for compiled execution, put those arrays in
PipelineState directly and compose only OperationSpec-backed steps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

from Pufs.primitives import PRNGKey, Weight, Challenge
from Pufs.operations import (
    OperationSpec,
    OperationState as PipelineState,
    apply_operation,
    lower_operations,
    op_add_gaussian_noise,
    op_age,
    op_evaluate_response,
    op_evaluate_xor_response,
    op_evaluate_feedforward_response,
    op_evaluate_feedforward_xor_response,
    op_generate_challenges,
    op_generate_weights,
)



# Step and spec wrappers


class OperationStep:
    """
    Callable compatibility wrapper around an immutable OperationSpec.

    The wrapper lets older code continue to treat a pipeline entry as a
    ``PipelineState -> PipelineState`` callable, while the spec remains
    available for lowering/caching.
    """

    __slots__ = ("spec", "label")

    def __init__(self, spec: OperationSpec, label: str = "") -> None:
        self.spec = spec
        self.label = label or _label_for_spec(spec)

    @property
    def __name__(self) -> str:
        """Function-like name used by compose() and __repr__()."""
        return self.label

    def __call__(self, state: PipelineState) -> PipelineState:
        """Apply the wrapped operation spec to *state*."""
        return apply_operation(self.spec, state)

    def __repr__(self) -> str:
        """Human-readable step representation."""
        return self.label


# A Step is any callable that takes and returns a PipelineState.
Step = Callable[[PipelineState], PipelineState]
Composable = Union[Step, OperationSpec]


def _label_for_spec(spec: OperationSpec) -> str:
    """Create a compact label from an operation spec."""
    if not spec.params:
        return f"{spec.kind}()"
    params = ", ".join(f"{key}={value!r}" for key, value in spec.params)
    return f"{spec.kind}({params})"


def _normalise_step(step: Composable) -> Step:
    """Convert OperationSpec values to callable OperationStep wrappers."""
    if isinstance(step, OperationSpec):
        return OperationStep(step)
    return step


def _step_label(step: Step) -> str:
    """Return a stable label for a step-like object."""
    return getattr(step, "__name__", repr(step))



# PUFPipeline


class PUFPipeline:
    """
    Immutable ordered sequence of pipeline steps.

    ``add`` returns a new PUFPipeline, so callers cannot accidentally alter an
    existing pipeline shared across tests or experiments.  When every step is an
    OperationStep, ``lower`` can produce a cached executor for the sequence.
    """

    __slots__ = ("_steps",)

    def __init__(self, steps: Optional[Tuple[Tuple[str, Step], ...]] = None) -> None:
        """Initialise with an optional pre-built step tuple."""
        self._steps: Tuple[Tuple[str, Step], ...] = tuple(steps or ())

    @property
    def steps(self) -> Tuple[Tuple[str, Step], ...]:
        """Read-only view of labelled pipeline steps."""
        return self._steps

    @property
    def specs(self) -> Tuple[OperationSpec, ...]:
        """
        Return the immutable specs for a fully spec-backed pipeline.

        Raises
        ------
        ValueError
            If any step is a custom Python callable or captures runtime arrays.
        """
        specs = []
        for label, step in self._steps:
            if not isinstance(step, OperationStep):
                raise ValueError(
                    f"Pipeline step {label!r} is not OperationSpec-backed; "
                    "use run(..., compiled=False) or move runtime arrays into "
                    "PipelineState before lowering."
                )
            specs.append(step.spec)
        return tuple(specs)

    def add(self, step: Composable, label: str = "") -> "PUFPipeline":
        """
        Append a step/spec and return a new pipeline.

        Args:
            step: callable PipelineState -> PipelineState, or an OperationSpec.
            label: optional name shown in __repr__.

        Returns:
            PUFPipeline: new pipeline with step appended.
        """
        normalised = _normalise_step(step)
        name = label or _step_label(normalised)
        return PUFPipeline(self._steps + ((name, normalised),))

    def lower(self, *, jit: bool = True) -> Callable[[PipelineState], PipelineState]:
        """
        Lower a fully spec-backed pipeline into a cached executor.

        For compiled execution with explicit arrays, initialise state with the
        arrays first, e.g. ``PipelineState(rng=key, weight=w, challenge=c)`` and
        compose only compute/generation specs.
        """
        return lower_operations(self.specs, jit=jit)

    def run(self, rng: PRNGKey, *, compiled: bool = False) -> PipelineState:
        """
        Execute all steps from a fresh state seeded with *rng*.

        Args:
            rng: starting PRNG key.
            compiled: when True, lower spec-backed steps into a cached executor.

        Returns:
            PipelineState: state after all steps have run.
        """
        return self.run_from(PipelineState(rng=rng), compiled=compiled)

    def run_from(self, state: PipelineState, *, compiled: bool = False) -> PipelineState:
        """
        Execute all steps from an existing state.

        This lets BasePUF seed a pipeline with its stored weight without
        duplicating the execution loop.  ``compiled=True`` requires that every
        step be OperationSpec-backed.
        """
        if compiled:
            return self.lower(jit=True)(state)
        for _label, step in self._steps:
            state = step(state)
        return state

    def __repr__(self) -> str:
        """Human-readable pipeline summary."""
        labels = [label for label, _ in self._steps]
        return f"PUFPipeline([{', '.join(labels)}])"



# compose() convenience constructor


def compose(*steps: Composable) -> PUFPipeline:
    """
    Build a PUFPipeline from a sequence of step callables/specs.

    Args:
        *steps: Step callables produced by factories below, or OperationSpec
                values produced by Pufs.operations factories.

    Returns:
        PUFPipeline: pipeline that runs the steps left-to-right.
    """
    labelled = []
    for step in steps:
        normalised = _normalise_step(step)
        labelled.append((_step_label(normalised), normalised))
    return PUFPipeline(tuple(labelled))



# Step factories


def generate_weights(n_stages: int, k: int = 1) -> OperationStep:
    """
    Step: generate a ``(k, n_stages + 1)`` Arbiter weight matrix.
    """
    return OperationStep(op_generate_weights(n_stages=n_stages, k=k))


def use_weights(weight: Weight) -> Step:
    """
    Step: inject a pre-built weight matrix, bypassing generation.

    This is a compatibility/binding step and is intentionally not part of the
    OperationSpec cache key.  For compiled execution, construct the initial
    PipelineState with ``weight=...`` instead.
    """
    def _step(state: PipelineState) -> PipelineState:
        return replace(state, weight=weight)

    _step.__name__ = "use_weights(prebuilt)"
    return _step


def generate_challenges(n_challenges: int, n_stages: Optional[int] = None) -> OperationStep:
    """
    Step: generate an ``(N, n_stages)`` challenge matrix from the current rng.

    If n_stages is None, it is inferred from the current Arbiter weight width.
    """
    return OperationStep(op_generate_challenges(n_challenges=n_challenges, n_stages=n_stages))


def use_challenges(challenge: Challenge) -> Step:
    """
    Step: inject a pre-built challenge matrix, bypassing generation.

    This is a compatibility/binding step and is intentionally not part of the
    OperationSpec cache key.  For compiled execution, construct the initial
    PipelineState with ``challenge=...`` instead.
    """
    def _step(state: PipelineState) -> PipelineState:
        return replace(state, challenge=challenge)

    _step.__name__ = "use_challenges(prebuilt)"
    return _step


def add_gaussian_noise(sigma: float) -> OperationStep:
    """
    Step: add Gaussian noise N(0, sigma^2) to the weight matrix.
    """
    return OperationStep(op_add_gaussian_noise(sigma=sigma))


def age(n_steps: int, sigma_per_step: float = 0.01, model: str = "additive") -> OperationStep:
    """
    Step: simulate weight drift due to device aging.

    ``model='additive'``: constant sigma per step.
    ``model='exponential'``: sigma decays as sigma_per_step * 0.95^t.
    """
    return OperationStep(op_age(n_steps=n_steps, sigma_per_step=sigma_per_step, model=model))


def evaluate_response() -> OperationStep:
    """
    Step: evaluate the Arbiter PUF response.
    """
    return OperationStep(op_evaluate_response())


def evaluate_xor_response() -> OperationStep:
    """
    Step: evaluate the XOR PUF response.
    """
    return OperationStep(op_evaluate_xor_response())


def evaluate_feedforward_response(loops: Iterable[Sequence[int]]) -> OperationStep:
    """
    Step: evaluate a feed-forward Arbiter PUF from compact FF weights.
    """
    return OperationStep(op_evaluate_feedforward_response(loops))


def evaluate_feedforward_xor_response(
    loop_specs: Iterable[Iterable[Sequence[int]]],
) -> OperationStep:
    """
    Step: evaluate a feed-forward XOR PUF from compact FF-XOR weights.
    """
    return OperationStep(op_evaluate_feedforward_xor_response(loop_specs))


def apply(fn: Callable[[PipelineState], PipelineState], label: str = "") -> Step:
    """
    Step: apply an arbitrary state transformation.

    Custom Python callables are kept for flexibility, but they are not lowerable
    into the cached OperationSpec executor.  Prefer registering a new operation
    in Pufs.operations when the step belongs in a hot or reusable path.
    """
    fn.__name__ = label or getattr(fn, "__name__", "custom_step")
    return fn
