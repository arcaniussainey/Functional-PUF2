"""
Immutable operation-spec DSL for composing PUF experiments.

The public primitives in :mod:`Pufs.primitives` are intentionally small,
functional JAX kernels.  This module sits one level above them: it describes
what should happen as immutable data, then lowers those descriptions into a
cached executor.  The design goal is to preserve a readable, DSL-like surface
without putting Python closures into hot numerical paths.

Core concepts
-------------
OperationSpec
    Frozen, hashable description of one operation.  A spec contains only static
    metadata such as ``kind='generate_challenges'`` and ``n_challenges=1000``.
    Runtime arrays such as weights and challenge matrices live in
    OperationState, not in the spec, so specs can be cached and reused safely.

OperationState
    Frozen carrier for runtime data: PRNG key, weight matrix, challenge matrix,
    and response object.  The state is a pytree so a lowered operation sequence
    can be JIT compiled by JAX.

OperationHandler
    Registry entry that maps an operation kind to a pure transformation
    ``OperationState -> OperationState``.  Future developers add new operations
    by registering a handler and creating specs for it; they do not need to
    modify the pipeline executor.

The older :mod:`Pufs.pipeline` API is now a compatibility layer over this
module where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from Pufs.primitives import (
    PRNGKey,
    Weight,
    Challenge,
    Response,
    generate_weights as _prim_generate_weights,
    generate_challenges as _prim_generate_challenges,
    get_response,
    xor_get_response,
    noisy_generate_weights,
)
from Pufs.feedforward_core import (
    feedforward_response_from_weight,
    feedforward_xor_response_from_weight,
    normalise_loops,
    normalise_xor_loop_specs,
)

ParamValue = Any
ParamTuple = Tuple[Tuple[str, ParamValue], ...]
Missing = object()



# Immutable execution state


@register_pytree_node_class
@dataclass(frozen=True)
class OperationState:
    """
    Immutable runtime state threaded through operation specs.

    The fields deliberately mirror the historical PipelineState interface so
    existing code can migrate gradually.  The object is registered as a JAX
    pytree, allowing a lowered spec sequence to be compiled as one executor.

    Attributes
    ----------
    rng:
        Current PRNG key. Stochastic operations consume this and return a new
        state with an advanced key.
    weight:
        Current weight matrix, usually shape ``(k, n_stages + 1)`` for Arbiter-family PUFs.
    challenge:
        Current challenge matrix, usually shape ``(n_challenges, n_stages)``.
    response:
        Current response.  For XOR PUFs this is commonly
        ``(individual, xor_response)``.
    """

    rng: PRNGKey
    weight: Optional[Weight] = None
    challenge: Optional[Challenge] = None
    response: Optional[Any] = None

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[()]]:
        """Return pytree children and auxiliary data for JAX transforms."""
        return (self.rng, self.weight, self.challenge, self.response), ()

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[()], children: Tuple[Any, ...]) -> "OperationState":
        """Reconstruct an OperationState from pytree children."""
        del aux_data
        rng, weight, challenge, response = children
        return cls(rng=rng, weight=weight, challenge=challenge, response=response)



# Operation specs and registry


def _freeze_value(value: Any) -> ParamValue:
    """
    Convert parameter values into hashable, cache-safe values.

    JAX/NumPy arrays are intentionally rejected.  Arrays belong in
    OperationState so compiled executors can be reused across many weight or
    challenge matrices with the same static operation sequence.
    """
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items()))

    # Avoid importing numpy just to identify arrays.  Anything with shape/dtype
    # that is not a plain scalar is almost certainly runtime data.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if getattr(value, "shape", ()) != ():
            raise TypeError(
                "OperationSpec parameters must be static and hashable; put arrays "
                "in OperationState rather than in the spec."
            )
        try:
            return value.item()
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise TypeError("Scalar array parameter could not be converted.") from exc

    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"Unhashable OperationSpec parameter: {value!r}") from exc
    return value


def _freeze_params(params: Mapping[str, Any]) -> ParamTuple:
    """Normalise keyword parameters into a deterministic, hashable tuple."""
    return tuple(sorted((str(name), _freeze_value(value)) for name, value in params.items()))


@dataclass(frozen=True)
class OperationSpec:
    """
    Immutable, hashable description of one PUF operation.

    Specs should contain static metadata only: dimensions, operation kind,
    scalar noise parameters, and similar compile-time configuration.  Runtime
    arrays such as weights/challenges should be supplied via OperationState.

    Example
    -------
    >>> spec = OperationSpec.create("generate_challenges", n_challenges=1000, n_stages=64)
    >>> spec.get("n_challenges")
    1000
    """

    kind: str
    params: ParamTuple = ()

    @classmethod
    def create(cls, kind: str, **params: Any) -> "OperationSpec":
        """Create a spec from an operation kind and static parameters."""
        if not kind:
            raise ValueError("OperationSpec kind must be a non-empty string.")
        return cls(kind=kind, params=_freeze_params(params))

    def get(self, name: str, default: Any = Missing) -> Any:
        """Return a named parameter, optionally using a default value."""
        for key, value in self.params:
            if key == name:
                return value
        if default is Missing:
            raise KeyError(f"OperationSpec {self.kind!r} has no parameter {name!r}.")
        return default

    def with_params(self, **updates: Any) -> "OperationSpec":
        """Return a copy with selected parameters changed."""
        current = dict(self.params)
        current.update(updates)
        return OperationSpec.create(self.kind, **current)


ApplyFn = Callable[[OperationState, OperationSpec], OperationState]
ValidateFn = Callable[[OperationSpec], None]


@dataclass(frozen=True)
class OperationHandler:
    """
    Registry entry for one operation kind.

    ``apply`` should be a pure transformation that returns a new OperationState.
    If the operation is stochastic, it must consume ``state.rng`` and return the
    advanced key in the returned state.
    """

    kind: str
    apply: ApplyFn
    validate: Optional[ValidateFn] = None
    doc: str = ""


_REGISTRY: Dict[str, OperationHandler] = {}


def register_operation(
    kind: str,
    apply: ApplyFn,
    *,
    validate: Optional[ValidateFn] = None,
    doc: str = "",
    replace_existing: bool = False,
) -> OperationHandler:
    """
    Register an operation handler for future specs.

    Args
    ----
    kind:
        Stable operation identifier.  This is the value stored in OperationSpec.
    apply:
        Pure state transformer for the operation.
    validate:
        Optional static validation hook executed before applying/lowering specs.
    doc:
        Human-readable operation documentation.
    replace_existing:
        Set to True to intentionally override an existing handler.

    Returns
    -------
    OperationHandler
        The frozen handler that was registered.
    """
    if not kind:
        raise ValueError("Operation kind must be a non-empty string.")
    if kind in _REGISTRY and not replace_existing:
        raise ValueError(f"Operation kind {kind!r} is already registered.")
    handler = OperationHandler(kind=kind, apply=apply, validate=validate, doc=doc)
    _REGISTRY[kind] = handler
    return handler


def get_operation_handler(kind: str) -> OperationHandler:
    """Return the registered handler for *kind*."""
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise KeyError(f"No operation handler registered for kind {kind!r}.") from exc


def registered_operations() -> Tuple[str, ...]:
    """Return the currently registered operation kinds."""
    return tuple(sorted(_REGISTRY))


def apply_operation(spec: OperationSpec, state: OperationState) -> OperationState:
    """Apply one spec to an OperationState using the registry."""
    handler = get_operation_handler(spec.kind)
    if handler.validate is not None:
        handler.validate(spec)
    return handler.apply(state, spec)



# Lowering / cached execution


@lru_cache(maxsize=128)
def _lower_specs(
    specs: Tuple[OperationSpec, ...],
    jit: bool = True,
) -> Callable[[OperationState], OperationState]:
    """
    Lower a static operation sequence into a cached executor.

    The Python loop over specs exists only in the lowered closure.  With
    ``jit=True`` JAX traces that sequence once for a given spec tuple and input
    shape, then reuses the compiled executable for subsequent calls.
    """
    for spec in specs:
        handler = get_operation_handler(spec.kind)
        if handler.validate is not None:
            handler.validate(spec)

    def _execute(state: OperationState) -> OperationState:
        for spec in specs:
            state = apply_operation(spec, state)
        return state

    return jax.jit(_execute) if jit else _execute


def lower_operations(
    specs: Iterable[OperationSpec],
    *,
    jit: bool = True,
) -> Callable[[OperationState], OperationState]:
    """
    Return a cached executor for an immutable operation sequence.

    Args
    ----
    specs:
        OperationSpec values describing the pipeline.  They must not contain
        arrays or other unhashable runtime data.
    jit:
        When True, return a JIT-compiled executor.  When False, return the same
        semantic executor without JIT; useful for debugging.
    """
    return _lower_specs(tuple(specs), jit)


@dataclass(frozen=True)
class OperationPipeline:
    """
    Immutable operation-spec sequence with a small execution API.

    This is the preferred DSL for new code.  It keeps operations declarative and
    cacheable while still allowing readable composition.
    """

    specs: Tuple[OperationSpec, ...] = ()

    def add(self, spec: OperationSpec) -> "OperationPipeline":
        """Return a new pipeline with *spec* appended."""
        return OperationPipeline(self.specs + (spec,))

    def lower(self, *, jit: bool = True) -> Callable[[OperationState], OperationState]:
        """Return a cached executor for this operation sequence."""
        return lower_operations(self.specs, jit=jit)

    def run(self, state: OperationState, *, jit: bool = True) -> OperationState:
        """Execute this operation sequence from an explicit state."""
        return self.lower(jit=jit)(state)

    def __iter__(self):
        """Iterate over specs left-to-right."""
        return iter(self.specs)


def op_pipeline(*specs: OperationSpec) -> OperationPipeline:
    """Build an immutable OperationPipeline from specs."""
    return OperationPipeline(tuple(specs))



# Built-in spec factories


def op_generate_weights(n_stages: int, k: int = 1) -> OperationSpec:
    """Spec: generate a ``(k, n_stages)`` weight matrix from ``state.rng``."""
    return OperationSpec.create("generate_weights", n_stages=int(n_stages), k=int(k))


def op_generate_challenges(n_challenges: int, n_stages: Optional[int] = None) -> OperationSpec:
    """Spec: generate a challenge matrix from ``state.rng``."""
    return OperationSpec.create(
        "generate_challenges",
        n_challenges=int(n_challenges),
        n_stages=None if n_stages is None else int(n_stages),
    )


def op_add_gaussian_noise(sigma: float) -> OperationSpec:
    """Spec: perturb the current weight matrix by Gaussian noise."""
    return OperationSpec.create("add_gaussian_noise", sigma=float(sigma))


def op_age(n_steps: int, sigma_per_step: float = 0.01, model: str = "additive") -> OperationSpec:
    """Spec: simulate device aging as repeated stochastic weight drift."""
    return OperationSpec.create(
        "age",
        n_steps=int(n_steps),
        sigma_per_step=float(sigma_per_step),
        model=str(model),
    )


def op_evaluate_response() -> OperationSpec:
    """Spec: compute a single/parallel arbiter response with get_response."""
    return OperationSpec.create("evaluate_response")


def op_evaluate_xor_response() -> OperationSpec:
    """Spec: compute per-arbiter and XOR-combined responses."""
    return OperationSpec.create("evaluate_xor_response")


def op_evaluate_feedforward_response(loops: Iterable[Tuple[int, int]]) -> OperationSpec:
    """Spec: evaluate a feed-forward Arbiter PUF.

    The current weight must use the compact feed-forward layout: row 0 is the
    main arbiter and rows 1..L are padded intermediate loop weights.
    """
    frozen_loops = tuple((int(src), int(tgt)) for src, tgt in loops)
    return OperationSpec.create("evaluate_feedforward_response", loops=frozen_loops)


def op_evaluate_feedforward_xor_response(
    loop_specs: Iterable[Iterable[Tuple[int, int]]],
) -> OperationSpec:
    """Spec: evaluate a feed-forward XOR PUF.

    ``loop_specs`` is one loop-list per XOR component.  The current weight must
    concatenate each component's compact feed-forward weight rows.
    """
    frozen_specs = tuple(
        tuple((int(src), int(tgt)) for src, tgt in component)
        for component in loop_specs
    )
    return OperationSpec.create(
        "evaluate_feedforward_xor_response",
        loop_specs=frozen_specs,
    )



# Built-in operation handlers


def _require_weight(state: OperationState, op_name: str) -> Weight:
    """Return state.weight or raise a programmer-facing error."""
    if state.weight is None:
        raise ValueError(f"{op_name}: weight must be set before this operation.")
    return state.weight


def _require_challenge(state: OperationState, op_name: str) -> Challenge:
    """Return state.challenge or raise a programmer-facing error."""
    if state.challenge is None:
        raise ValueError(f"{op_name}: challenge must be set before this operation.")
    return state.challenge


def _apply_generate_weights(state: OperationState, spec: OperationSpec) -> OperationState:
    k = int(spec.get("k"))
    n_stages = int(spec.get("n_stages"))
    rng, sk = jax.random.split(state.rng)
    return replace(state, rng=rng, weight=_prim_generate_weights(sk, (k, n_stages)))


def _apply_generate_challenges(state: OperationState, spec: OperationSpec) -> OperationState:
    n_challenges = int(spec.get("n_challenges"))
    n_stages = spec.get("n_stages", None)
    if n_stages is None:
        weight = _require_weight(state, "generate_challenges")
        n_stages = weight.shape[1] - 1
    rng, sk = jax.random.split(state.rng)
    challenge = _prim_generate_challenges(sk, (n_challenges, int(n_stages)))
    return replace(state, rng=rng, challenge=challenge)


def _apply_add_gaussian_noise(state: OperationState, spec: OperationSpec) -> OperationState:
    weight = _require_weight(state, "add_gaussian_noise")
    sigma = jnp.float32(spec.get("sigma"))
    rng, sk = jax.random.split(state.rng)
    rng_after, noisy_weight = noisy_generate_weights(sk, weight, sigma)
    return replace(state, rng=rng_after, weight=noisy_weight)


def _apply_age(state: OperationState, spec: OperationSpec) -> OperationState:
    weight = _require_weight(state, "age")
    n_steps = int(spec.get("n_steps"))
    sigma = jnp.float32(spec.get("sigma_per_step"))
    model = spec.get("model")

    def _additive_scan(carry: Weight, rng_step: PRNGKey) -> Tuple[Weight, None]:
        noise = jax.random.normal(rng_step, shape=carry.shape) * sigma
        return carry + noise, None

    def _exponential_scan(
        carry: Tuple[Weight, jax.Array], rng_and_t: Tuple[PRNGKey, jax.Array]
    ) -> Tuple[Tuple[Weight, jax.Array], None]:
        w, t = carry
        rng_step, _ = rng_and_t
        sigma_t = sigma * (jnp.float32(0.95) ** t)
        noise = jax.random.normal(rng_step, shape=w.shape) * sigma_t
        return (w + noise, t + jnp.float32(1)), None

    rng, *subkeys = jax.random.split(state.rng, n_steps + 1)
    subkeys_stacked = jnp.stack(subkeys)

    if model == "additive":
        w_aged, _ = jax.lax.scan(_additive_scan, weight, subkeys_stacked)
    elif model == "exponential":
        ts = jnp.arange(n_steps, dtype=jnp.float32)
        (w_aged, _), _ = jax.lax.scan(
            _exponential_scan,
            (weight, jnp.float32(0)),
            (subkeys_stacked, ts),
        )
    else:
        raise ValueError(f"age: unknown model {model!r}. Use 'additive' or 'exponential'.")

    return replace(state, rng=rng, weight=w_aged)


def _apply_evaluate_response(state: OperationState, spec: OperationSpec) -> OperationState:
    del spec
    weight = _require_weight(state, "evaluate_response")
    challenge = _require_challenge(state, "evaluate_response")
    return replace(state, response=get_response(weight, challenge))


def _apply_evaluate_xor_response(state: OperationState, spec: OperationSpec) -> OperationState:
    del spec
    weight = _require_weight(state, "evaluate_xor_response")
    challenge = _require_challenge(state, "evaluate_xor_response")
    return replace(state, response=xor_get_response(weight, challenge))


def _infer_stage_count(state: OperationState, op_name: str) -> int:
    """Infer challenge-stage count from the current challenge or weight matrix."""
    if state.challenge is not None:
        return int(state.challenge.shape[1])
    if state.weight is not None:
        return int(state.weight.shape[1]) - 1
    raise ValueError(f"{op_name}: challenge or weight must be set before validation.")


def _apply_evaluate_feedforward_response(
    state: OperationState,
    spec: OperationSpec,
) -> OperationState:
    weight = _require_weight(state, "evaluate_feedforward_response")
    challenge = _require_challenge(state, "evaluate_feedforward_response")
    n_stages = _infer_stage_count(state, "evaluate_feedforward_response")
    loops = normalise_loops(spec.get("loops"), n_stages)
    response = feedforward_response_from_weight(weight, challenge, loops)
    return replace(state, response=response)


def _apply_evaluate_feedforward_xor_response(
    state: OperationState,
    spec: OperationSpec,
) -> OperationState:
    weight = _require_weight(state, "evaluate_feedforward_xor_response")
    challenge = _require_challenge(state, "evaluate_feedforward_xor_response")
    n_stages = _infer_stage_count(state, "evaluate_feedforward_xor_response")
    loop_specs = normalise_xor_loop_specs(spec.get("loop_specs"), n_stages)
    response = feedforward_xor_response_from_weight(weight, challenge, loop_specs)
    return replace(state, response=response)


def _validate_age(spec: OperationSpec) -> None:
    """Validate static aging parameters early."""
    if int(spec.get("n_steps")) < 0:
        raise ValueError("age: n_steps must be non-negative.")
    if spec.get("model") not in ("additive", "exponential"):
        raise ValueError("age: model must be 'additive' or 'exponential'.")


register_operation(
    "generate_weights",
    _apply_generate_weights,
    doc="Generate a weight matrix from the state's PRNG key.",
)
register_operation(
    "generate_challenges",
    _apply_generate_challenges,
    doc="Generate a challenge matrix from the state's PRNG key.",
)
register_operation(
    "add_gaussian_noise",
    _apply_add_gaussian_noise,
    doc="Add Gaussian perturbation to the current weight matrix.",
)
register_operation(
    "age",
    _apply_age,
    validate=_validate_age,
    doc="Simulate additive or exponential stochastic weight drift.",
)
register_operation(
    "evaluate_response",
    _apply_evaluate_response,
    doc="Evaluate get_response(weight, challenge).",
)
register_operation(
    "evaluate_xor_response",
    _apply_evaluate_xor_response,
    doc="Evaluate xor_get_response(weight, challenge).",
)

register_operation(
    "evaluate_feedforward_response",
    _apply_evaluate_feedforward_response,
    doc="Evaluate a compact feed-forward Arbiter PUF.",
)
register_operation(
    "evaluate_feedforward_xor_response",
    _apply_evaluate_feedforward_xor_response,
    doc="Evaluate compact feed-forward XOR PUF weights.",
)
