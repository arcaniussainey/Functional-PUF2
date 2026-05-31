"""
Aging-aware PUF implementations and deterministic drift models.

The classes in this module keep the existing Arbiter/XOR response semantics but
add a full weight-history tensor.  An experiment can therefore compare the same
physical PUF at multiple simulated ages without allocating one Python object per
snapshot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class
import numpy as np

from Pufs.base import BasePUF
from Pufs.primitives import (
    Challenge,
    Delta,
    PRNGKey,
    Response,
    Weight,
    get_delta_response,
    get_response,
    xor_get_response,
)

PhiCacheKey = Tuple[Tuple[int, ...], str]
_PHI_CACHE: dict[PhiCacheKey, jax.Array] = {}


class AgingPUFProtocol(Protocol):
    """Small structural protocol shared by the aging PUF classes."""

    def get_weight(self, age_idx: int = -1) -> jax.Array:
        """Return the stored weight snapshot for ``age_idx``."""

    def get_response(self, challenge: Challenge, age_idx: int = -1):
        """Evaluate the PUF at ``age_idx``."""


def to_phi(challenges: Challenge) -> jax.Array:
    """
    Convert challenges from ``{-1, +1}`` into suffix-product feature vectors.

    ``phi[i]`` is the product of challenge bits ``i`` through ``n - 1``.  The
    helper is useful for explicit aging experiments that repeatedly evaluate the
    same challenge matrix against many weight snapshots.
    """
    return jnp.cumprod(challenges[:, ::-1], axis=1)[:, ::-1].astype(jnp.float32)


def clear_phi_cache() -> None:
    """Clear the process-local feature-transform cache used by ``get_phi``."""
    _PHI_CACHE.clear()


def get_phi(challenges: Challenge) -> jax.Array:
    """
    Return a cached suffix-product transform for an eager challenge matrix.

    The cache key hashes the full array content.  This is intentionally safer
    than hashing only a prefix because repeated scientific experiments often
    reuse challenge matrices with identical shapes but different tails.
    """
    raw = np.asarray(challenges)
    digest = hashlib.sha256(raw.tobytes()).hexdigest()
    key = (tuple(raw.shape), digest)
    if key not in _PHI_CACHE:
        _PHI_CACHE[key] = to_phi(jnp.asarray(challenges))
    return _PHI_CACHE[key]


@jax.jit
def _response_from_phi(phi: jax.Array, weight: Weight) -> Response:
    """Return binary responses for a precomputed phi matrix and weights."""
    delta = phi @ weight.T
    return ((jnp.sign(delta) + 1) / 2).astype(jnp.uint8)


@jax.jit
def _delta_from_phi(phi: jax.Array, weight: Weight) -> Delta:
    """Return raw delay deltas for a precomputed phi matrix and weights."""
    return (phi @ weight.T).astype(jnp.float32)


@jax.jit
def _xor_response_from_phi(phi: jax.Array, weight: Weight) -> Response:
    """Return the XOR-reduced response for a precomputed phi matrix."""
    individual = _response_from_phi(phi, weight)
    return lax.reduce(individual, jnp.uint8(0), lax.bitwise_xor, (1,))


def drift_additive(
    rng: PRNGKey,
    weight: Weight,
    n_steps: int,
    sigma_per_step: float = 0.01,
) -> jax.Array:
    """
    Simulate an additive Gaussian random-walk aging process.

    The returned history includes the starting weight at index ``0`` and then
    ``n_steps`` aged snapshots.  The model is deterministic for a fixed PRNG key.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if n_steps == 0:
        return weight[None, ...]

    sigma = jnp.float32(sigma_per_step)

    def _step(carry: Weight, rng_step: PRNGKey) -> tuple[Weight, Weight]:
        next_weight = carry + jax.random.normal(rng_step, shape=carry.shape) * sigma
        return next_weight, next_weight

    subkeys = jax.random.split(rng, n_steps)
    _, history = jax.lax.scan(_step, weight, subkeys)
    return jnp.concatenate([weight[None, ...], history], axis=0)


def drift_exponential(
    rng: PRNGKey,
    weight: Weight,
    n_steps: int,
    base_sigma: float = 0.05,
    decay: float = 0.95,
) -> jax.Array:
    """
    Simulate early-life degradation with exponentially decaying noise.

    ``sigma_t = base_sigma * decay ** t``.  The returned history includes the
    starting weight at index ``0`` and then ``n_steps`` aged snapshots.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if n_steps == 0:
        return weight[None, ...]
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in the interval (0, 1].")

    base = jnp.float32(base_sigma)
    decay_value = jnp.float32(decay)

    def _step(carry: tuple[Weight, jax.Array], rng_step: PRNGKey):
        current_weight, step_idx = carry
        sigma_t = base * (decay_value**step_idx)
        next_weight = current_weight + jax.random.normal(
            rng_step, shape=current_weight.shape
        ) * sigma_t
        return (next_weight, step_idx + jnp.float32(1)), next_weight

    subkeys = jax.random.split(rng, n_steps)
    _, history = jax.lax.scan(_step, (weight, jnp.float32(0)), subkeys)
    return jnp.concatenate([weight[None, ...], history], axis=0)


def drift_history(
    rng: PRNGKey,
    weight: Weight,
    n_steps: int,
    model: str = "additive",
    **kwargs: float,
) -> jax.Array:
    """Dispatch to a named aging model and return a full weight history."""
    if model == "additive":
        return drift_additive(rng, weight, n_steps, **kwargs)
    if model == "exponential":
        return drift_exponential(rng, weight, n_steps, **kwargs)
    raise ValueError(f"Unknown aging model: {model!r}.")


@dataclass(frozen=True)
class AgeRef:
    """
    Lightweight immutable handle to one age snapshot of an aging PUF.

    The handle stores a reference to the parent object and an index.  It does
    not copy weights, so it is safe to create many labels for the same history.
    """

    puf: AgingPUFProtocol
    age_idx: int
    label: str = ""

    def get_weight(self) -> jax.Array:
        """Return the selected weight snapshot."""
        return self.puf.get_weight(self.age_idx)

    def get_response(self, challenges: Challenge):
        """Evaluate the selected age snapshot against ``challenges``."""
        return self.puf.get_response(challenges, self.age_idx)

    def __repr__(self) -> str:
        """Return a compact debugging representation."""
        suffix = f" ({self.label})" if self.label else ""
        return f"AgeRef(puf={type(self.puf).__name__}, age_idx={self.age_idx}{suffix})"


@register_pytree_node_class
class ArbiterPUF_Aging(BasePUF):
    """Single-arbiter PUF with an append-only simulated aging history."""

    def __init__(self, rng: PRNGKey, stages: int) -> None:
        """Create a fresh aging-aware Arbiter PUF."""
        super().__init__(rng, (1, stages))
        self.stages = int(stages)
        self.weight_history = self.weight[None, :, :]

    def tree_flatten(self):
        """Return JAX pytree children and auxiliary data."""
        children = (self.weight, self.weight_history)
        aux_data = (self.rng, self.stages)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct an aging PUF from pytree data."""
        rng, stages = aux_data
        weight, weight_history = children
        obj = cls.__new__(cls)
        obj.rng = rng
        obj.dim = (1, stages)
        obj.weight = weight
        obj.stages = stages
        obj.weight_history = weight_history
        return obj

    @property
    def n_ages(self) -> int:
        """Return the number of stored age snapshots."""
        return int(self.weight_history.shape[0])

    def add_aging(
        self,
        rng: PRNGKey,
        n_steps: int,
        model: str = "additive",
        **kwargs,
    ) -> "ArbiterPUF_Aging":
        """Append ``n_steps`` simulated aging snapshots and return ``self``."""
        latest_weight = self.weight_history[-1]
        new_history = drift_history(rng, latest_weight, n_steps, model, **kwargs)
        self.weight_history = jnp.concatenate([self.weight_history, new_history[1:]], axis=0)
        return self

    def ref(self, age_idx: int = -1, label: str = "") -> AgeRef:
        """Return an immutable reference to one stored age snapshot."""
        return AgeRef(puf=self, age_idx=age_idx, label=label)

    def get_weight(self, age_idx: int = -1) -> jax.Array:
        """Return the weight matrix at ``age_idx`` with shape ``(1, stages)``."""
        return self.weight_history[age_idx]

    def _compute_response(self, weight: Weight, challenge: Challenge) -> Response:
        """Compute an Arbiter response for the supplied weight matrix."""
        return get_response(weight, challenge)

    def get_response(self, challenge: Challenge, age_idx: int = -1) -> Response:
        """Evaluate a selected age snapshot against a challenge matrix."""
        return get_response(self.get_weight(age_idx), challenge)

    def get_delta_response(self, challenge: Challenge, age_idx: int = -1) -> Delta:
        """Return delay deltas for a selected age snapshot."""
        return get_delta_response(self.get_weight(age_idx), challenge)

    def get_phi_response(self, challenge: Challenge, age_idx: int = -1) -> Response:
        """Evaluate using explicit suffix-product features for research studies."""
        return _response_from_phi(get_phi(challenge), self.get_weight(age_idx))

    def response_all_ages(self, challenges: Challenge) -> jax.Array:
        """Return responses for every stored age snapshot."""
        return jax.vmap(lambda weight: get_response(weight, challenges))(self.weight_history)

    def clone(self) -> "ArbiterPUF_Aging":
        """Return a new object with the same weight history."""
        obj = type(self).__new__(type(self))
        obj.rng = self.rng
        obj.dim = self.dim
        obj.weight = self.weight
        obj.stages = self.stages
        obj.weight_history = self.weight_history
        return obj

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"ArbiterPUF_Aging(stages={self.stages}, n_ages={self.n_ages})"


@register_pytree_node_class
class XorPUF_Aging(BasePUF):
    """XOR PUF with an append-only simulated aging history."""

    def __init__(self, rng: PRNGKey, k: int, stages: int) -> None:
        """Create a fresh aging-aware XOR PUF."""
        super().__init__(rng, (k, stages))
        self.k = int(k)
        self.stages = int(stages)
        self.weight_history = self.weight[None, :, :]

    def tree_flatten(self):
        """Return JAX pytree children and auxiliary data."""
        children = (self.weight, self.weight_history)
        aux_data = (self.rng, self.k, self.stages)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct an aging XOR PUF from pytree data."""
        rng, k_value, stages = aux_data
        weight, weight_history = children
        obj = cls.__new__(cls)
        obj.rng = rng
        obj.dim = (k_value, stages)
        obj.weight = weight
        obj.k = k_value
        obj.stages = stages
        obj.weight_history = weight_history
        return obj

    @property
    def n_ages(self) -> int:
        """Return the number of stored age snapshots."""
        return int(self.weight_history.shape[0])

    def add_aging(
        self,
        rng: PRNGKey,
        n_steps: int,
        model: str = "additive",
        **kwargs,
    ) -> "XorPUF_Aging":
        """Append independent simulated aging histories for each XOR arbiter."""
        latest_weight = self.weight_history[-1]
        new_history = drift_history(rng, latest_weight, n_steps, model, **kwargs)
        self.weight_history = jnp.concatenate([self.weight_history, new_history[1:]], axis=0)
        return self

    def ref(self, age_idx: int = -1, label: str = "") -> AgeRef:
        """Return an immutable reference to one stored age snapshot."""
        return AgeRef(puf=self, age_idx=age_idx, label=label)

    def get_weight(self, age_idx: int = -1) -> jax.Array:
        """Return the XOR weight matrix at ``age_idx``."""
        return self.weight_history[age_idx]

    def _compute_response(self, weight: Weight, challenge: Challenge):
        """Compute individual and XOR-reduced responses."""
        return xor_get_response(weight, challenge)

    def get_response(self, challenge: Challenge, age_idx: int = -1):
        """Evaluate a selected age snapshot against a challenge matrix."""
        return xor_get_response(self.get_weight(age_idx), challenge)

    def response_all_ages(self, challenges: Challenge) -> jax.Array:
        """Return XOR-reduced responses for every stored age snapshot."""
        return jax.vmap(lambda weight: xor_get_response(weight, challenges)[1])(
            self.weight_history
        )

    def clone(self) -> "XorPUF_Aging":
        """Return a new object with the same weight history."""
        obj = type(self).__new__(type(self))
        obj.rng = self.rng
        obj.dim = self.dim
        obj.weight = self.weight
        obj.k = self.k
        obj.stages = self.stages
        obj.weight_history = self.weight_history
        return obj

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"XorPUF_Aging(k={self.k}, stages={self.stages}, n_ages={self.n_ages})"


AgingPUF = Union[ArbiterPUF_Aging, XorPUF_Aging]
