"""
Pufs/randomness.py
------------------
Deterministic PRNG and distribution helpers for PUF experiments.

JAX PRNG keys are explicit values.  This module provides a small, documented
stateful convenience layer for experiment orchestration while keeping hot
numerical kernels pure: keys are created at the boundary, then passed into
functional primitives or immutable operation states.

The most important design rule is reproducibility.  Given the same seed and
the same sequence of ``next()``/``split()`` calls, a DeterministicKeyStream
returns the same keys and therefore the same PUF weights, challenges, and
stochastic perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from Pufs.primitives import PRNGKey


@dataclass
class DeterministicKeyStream:
    """
    Explicit, reproducible stream of JAX PRNG subkeys.

    The class is intentionally stateful because it is used at experiment and
    attack orchestration boundaries.  Do not pass the stream itself into JIT
    compiled functions.  Instead, request keys at the Python boundary and pass
    the resulting ``PRNGKey`` values into JAX kernels.

    Parameters
    ----------
    seed:
        Integer seed controlling the entire stream.  Changing this value is the
        intended way to create independent experimental trials.
    """

    seed: int = 0
    _key: PRNGKey = field(init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise the stream from the configured integer seed."""
        self._key = jax.random.PRNGKey(self.seed)

    @property
    def count(self) -> int:
        """Number of subkeys consumed from this stream."""
        return self._count

    def next(self) -> PRNGKey:
        """
        Return the next deterministic subkey and advance the stream.

        Returns
        -------
        PRNGKey
            A fresh JAX PRNG key derived from the current stream state.
        """
        self._key, subkey = jax.random.split(self._key)
        self._count += 1
        return subkey

    def split(self, n: int) -> Tuple[PRNGKey, ...]:
        """
        Return ``n`` deterministic subkeys and advance the stream once.

        This is useful when a caller wants a stable group of sibling keys for a
        single conceptual phase, such as ``(puf_key, challenge_key, val_key)``.
        """
        if n < 0:
            raise ValueError("split count must be non-negative")
        if n == 0:
            return ()
        keys = jax.random.split(self.next(), n)
        return tuple(keys)

    def fork(self, salt: int = 0) -> "DeterministicKeyStream":
        """
        Create an independent child stream from the next key.

        ``salt`` lets a caller create named/offset child streams without
        disturbing reproducibility.  The child seed is derived from the parent
        stream state, so two parents at the same position create equivalent
        children.
        """
        child_seed = int(jax.random.randint(self.next(), (), 0, 2**31 - 1))
        return DeterministicKeyStream(child_seed ^ int(salt))


FallbackKeyFn = Callable[[], PRNGKey]


def next_key(
    key_stream: Optional[DeterministicKeyStream] = None,
    *,
    fallback: Optional[FallbackKeyFn] = None,
) -> PRNGKey:
    """
    Return a key from ``key_stream`` or from an explicit fallback generator.

    This helper lets legacy functions remain backward compatible: pass the
    module-level ``new_key`` function as ``fallback`` when no deterministic
    stream has been supplied.
    """
    if key_stream is not None:
        return key_stream.next()
    if fallback is None:
        raise ValueError("No key stream supplied and no fallback key generator provided.")
    return fallback()


@dataclass(frozen=True)
class DistributionSpec:
    """
    Immutable description of a simple random distribution.

    Distribution specs are intended for experiment setup and documentation, not
    for hiding random behavior inside numerical kernels.  The supported families
    cover the distributions commonly used in the PUF codebase today: Gaussian
    noise, Bernoulli bits, signed Bernoulli challenges, and integer ranges.
    """

    family: str
    loc: float = 0.0
    scale: float = 1.0
    low: int = 0
    high: int = 2
    dtype: str = "float32"

    @classmethod
    def normal(cls, loc: float = 0.0, scale: float = 1.0, dtype: str = "float32") -> "DistributionSpec":
        """Create a Gaussian distribution spec."""
        return cls(family="normal", loc=float(loc), scale=float(scale), dtype=dtype)

    @classmethod
    def bernoulli(cls, p: float = 0.5, dtype: str = "int32") -> "DistributionSpec":
        """Create a Bernoulli distribution spec returning 0/1 samples."""
        return cls(family="bernoulli", loc=float(p), dtype=dtype)

    @classmethod
    def signed_bernoulli(cls, p: float = 0.5, dtype: str = "int8") -> "DistributionSpec":
        """Create a Bernoulli distribution spec returning -1/+1 samples."""
        return cls(family="signed_bernoulli", loc=float(p), dtype=dtype)

    @classmethod
    def integers(cls, low: int, high: int, dtype: str = "int32") -> "DistributionSpec":
        """Create an integer distribution spec over ``[low, high)``."""
        return cls(family="integers", low=int(low), high=int(high), dtype=dtype)

    def sample(self, rng: PRNGKey, shape: Sequence[int]) -> jax.Array:
        """
        Draw samples from this distribution using an explicit PRNG key.

        Parameters
        ----------
        rng:
            JAX PRNG key.
        shape:
            Output sample shape.
        """
        dtype = getattr(jnp, self.dtype)
        out_shape = tuple(int(x) for x in shape)

        if self.family == "normal":
            return (jax.random.normal(rng, out_shape) * self.scale + self.loc).astype(dtype)
        if self.family == "bernoulli":
            return jax.random.bernoulli(rng, p=self.loc, shape=out_shape).astype(dtype)
        if self.family == "signed_bernoulli":
            bits = jax.random.bernoulli(rng, p=self.loc, shape=out_shape)
            return jnp.where(bits, 1, -1).astype(dtype)
        if self.family == "integers":
            return jax.random.randint(rng, out_shape, self.low, self.high, dtype=dtype)
        raise ValueError(f"Unsupported distribution family {self.family!r}.")


def sample_distribution(rng: PRNGKey, spec: DistributionSpec, shape: Sequence[int]) -> jax.Array:
    """Convenience wrapper for ``DistributionSpec.sample``."""
    return spec.sample(rng, shape)
