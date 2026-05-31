"""
Feed-forward Arbiter PUF classes.

A feed-forward PUF injects one or more intermediate arbiter outputs back into
later challenge positions.  The classes here preserve the library's BasePUF
interface while storing the extra intermediate weights in a compact matrix:
row ``0`` is the main arbiter and rows ``1..L`` are padded loop weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from Pufs.base import BasePUF
from Pufs.feedforward_core import (
    LoopSpec,
    LoopSpecs,
    XorLoopSpecs,
    feedforward_response_from_weight,
    feedforward_xor_response_from_weight,
    generate_weight_np,
    normalise_loops,
    normalise_xor_loop_specs,
    padded_prefix_weight,
    required_weight_rows,
    required_xor_weight_rows,
    validate_non_cascading,
)
from Pufs.primitives import Challenge, PRNGKey, Response, Weight

SeedLike = int | PRNGKey


def _seed_to_int(seed: SeedLike) -> int:
    """Convert an integer or JAX key into a deterministic NumPy seed."""
    if isinstance(seed, int):
        return seed
    raw = np.asarray(seed, dtype=np.uint32).reshape(-1)
    return int(np.bitwise_xor.reduce(raw, dtype=np.uint32))


@dataclass(frozen=True)
class FFLoop:
    """Public descriptor for one feed-forward loop."""

    src: int
    tgt: int

    @classmethod
    def from_pair(cls, pair: Sequence[int]) -> "FFLoop":
        """Create a loop descriptor from ``(src, tgt)``."""
        src, tgt = pair
        return cls(src=int(src), tgt=int(tgt))

    def as_tuple(self) -> LoopSpec:
        """Return the descriptor as a plain immutable pair."""
        return self.src, self.tgt


def _normalise_public_loops(loops: Iterable[Sequence[int] | FFLoop], n_stages: int) -> LoopSpecs:
    """Convert user-provided loop descriptors into canonical loop specs."""
    pairs = [loop.as_tuple() if isinstance(loop, FFLoop) else tuple(loop) for loop in loops]
    return normalise_loops(pairs, n_stages)


def _build_feedforward_weight(seed: int, n_stages: int, loops: LoopSpecs) -> jax.Array:
    """Generate the compact weight matrix for one feed-forward component."""
    rng = np.random.default_rng(seed)
    rows = [generate_weight_np(rng, n_stages)]
    rows.extend(padded_prefix_weight(rng, src + 1, n_stages) for src, _tgt in loops)
    return jnp.asarray(np.vstack(rows), dtype=jnp.float32)


@register_pytree_node_class
class FF_Arbiter(BasePUF):
    """Feed-forward Arbiter PUF that supports arbitrary loop topologies."""

    def __init__(self, rng: SeedLike, n: int, loops: Iterable[Sequence[int] | FFLoop]) -> None:
        """Create a feed-forward Arbiter PUF from loop descriptors."""
        self.n = int(n)
        self.loops = _normalise_public_loops(loops, self.n)
        self.seed = _seed_to_int(rng)
        self.rng = jax.random.PRNGKey(self.seed)
        self.weight = _build_feedforward_weight(self.seed, self.n, self.loops)
        self.dim = tuple(int(value) for value in self.weight.shape)

    def tree_flatten(self):
        """Return JAX pytree children and auxiliary data."""
        return (self.weight,), (self.seed, self.n, self.loops)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a feed-forward Arbiter PUF from pytree data."""
        seed, n_stages, loops = aux_data
        (weight,) = children
        obj = cls.__new__(cls)
        obj.seed = seed
        obj.n = n_stages
        obj.loops = loops
        obj.rng = jax.random.PRNGKey(seed)
        obj.weight = weight
        obj.dim = tuple(int(value) for value in weight.shape)
        return obj

    @property
    def n_loops(self) -> int:
        """Return the number of feed-forward loops."""
        return len(self.loops)

    @property
    def main_weight(self) -> jax.Array:
        """Return the main arbiter weight row."""
        return self.weight[0]

    def _compute_response(self, weight: Weight, challenge: Challenge) -> Response:
        """Evaluate this feed-forward Arbiter PUF."""
        return feedforward_response_from_weight(weight, challenge, self.loops)

    def clone(self) -> "FF_Arbiter":
        """Return a new feed-forward PUF with the same parameters and weights."""
        obj = type(self).__new__(type(self))
        obj.seed = self.seed
        obj.n = self.n
        obj.loops = self.loops
        obj.rng = self.rng
        obj.weight = self.weight
        obj.dim = self.dim
        return obj

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        loop_text = ", ".join(f"({src}->{tgt})" for src, tgt in self.loops)
        return f"FF_Arbiter(n={self.n}, loops=[{loop_text}])"


@register_pytree_node_class
class FF_Arbiter_Symbolic(FF_Arbiter):
    """
    Feed-forward Arbiter PUF restricted to non-cascading topologies.

    The current implementation shares the vectorized evaluator with
    ``FF_Arbiter`` but validates the symbolic-topology constraint at
    construction time.  This keeps the public type available for experiments
    that distinguish arbitrary versus symbolic-safe feed-forward layouts.
    """

    def __init__(self, rng: SeedLike, n: int, loops: Iterable[Sequence[int] | FFLoop]) -> None:
        """Create a symbolic-safe feed-forward Arbiter PUF."""
        super().__init__(rng, n, loops)
        validate_non_cascading(self.loops)

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        loop_text = ", ".join(f"({src}->{tgt})" for src, tgt in self.loops)
        return f"FF_Arbiter_Symbolic(n={self.n}, loops=[{loop_text}])"


@register_pytree_node_class
class FF_Arbiter_Expression(FF_Arbiter_Symbolic):
    """Expression-oriented alias for non-cascading feed-forward PUFs."""

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        loop_text = ", ".join(f"({src}->{tgt})" for src, tgt in self.loops)
        return f"FF_Arbiter_Expression(n={self.n}, loops=[{loop_text}])"


@register_pytree_node_class
class FF_XOR(BasePUF):
    """XOR composition of independent feed-forward Arbiter PUFs."""

    def __init__(
        self,
        rng: SeedLike,
        n: int,
        loop_specs: Iterable[Iterable[Sequence[int] | FFLoop]],
    ) -> None:
        """Create a feed-forward XOR PUF from per-component loop specs."""
        self.n = int(n)
        self.seed = _seed_to_int(rng)
        raw_specs = [
            [loop.as_tuple() if isinstance(loop, FFLoop) else tuple(loop) for loop in component]
            for component in loop_specs
        ]
        self.loop_specs = normalise_xor_loop_specs(raw_specs, self.n)
        self.k = len(self.loop_specs)
        self.rng = jax.random.PRNGKey(self.seed)
        self.weight = self._build_weight_matrix()
        self.dim = tuple(int(value) for value in self.weight.shape)

    def _build_weight_matrix(self) -> jax.Array:
        """Generate and concatenate every component's compact weight matrix."""
        rows = []
        for index, loops in enumerate(self.loop_specs):
            component = _build_feedforward_weight(self.seed + index, self.n, loops)
            rows.append(component)
        return jnp.concatenate(rows, axis=0)

    def tree_flatten(self):
        """Return JAX pytree children and auxiliary data."""
        return (self.weight,), (self.seed, self.n, self.loop_specs)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a feed-forward XOR PUF from pytree data."""
        seed, n_stages, loop_specs = aux_data
        (weight,) = children
        obj = cls.__new__(cls)
        obj.seed = seed
        obj.n = n_stages
        obj.loop_specs = loop_specs
        obj.k = len(loop_specs)
        obj.rng = jax.random.PRNGKey(seed)
        obj.weight = weight
        obj.dim = tuple(int(value) for value in weight.shape)
        return obj

    @property
    def required_rows(self) -> int:
        """Return the compact weight rows required by this topology."""
        return required_xor_weight_rows(self.loop_specs)

    def component_weight(self, index: int) -> jax.Array:
        """Return the compact weight matrix for one XOR component."""
        if not 0 <= index < self.k:
            raise IndexError(f"component index must be in [0, {self.k}).")
        offset = 0
        for component_index, loops in enumerate(self.loop_specs):
            row_count = required_weight_rows(loops)
            if component_index == index:
                return self.weight[offset : offset + row_count]
            offset += row_count
        raise IndexError(index)

    def _compute_response(self, weight: Weight, challenge: Challenge):
        """Evaluate individual component responses and the XOR reduction."""
        return feedforward_xor_response_from_weight(weight, challenge, self.loop_specs)

    def clone(self) -> "FF_XOR":
        """Return a new feed-forward XOR PUF with the same weights."""
        obj = type(self).__new__(type(self))
        obj.seed = self.seed
        obj.n = self.n
        obj.loop_specs = self.loop_specs
        obj.k = self.k
        obj.rng = self.rng
        obj.weight = self.weight
        obj.dim = self.dim
        return obj

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"FF_XOR(n={self.n}, k={self.k}, rows={self.required_rows})"


@register_pytree_node_class
class FF_XOR_Symbolic(FF_XOR):
    """Feed-forward XOR PUF whose components must be non-cascading."""

    def __init__(
        self,
        rng: SeedLike,
        n: int,
        loop_specs: Iterable[Iterable[Sequence[int] | FFLoop]],
    ) -> None:
        """Create a symbolic-safe feed-forward XOR PUF."""
        super().__init__(rng, n, loop_specs)
        for loops in self.loop_specs:
            validate_non_cascading(loops)

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"FF_XOR_Symbolic(n={self.n}, k={self.k}, rows={self.required_rows})"
