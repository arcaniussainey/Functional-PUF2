"""
Optional adapters for the external ``pypuf`` package.

The wrappers expose a small ``get_response(challenges)`` surface that matches
this project's PUF classes while keeping ``pypuf`` an optional dependency.  The
module imports cleanly without ``pypuf``; constructing a wrapper raises a clear
error if the dependency is unavailable.
"""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.base import BasePUF
from Pufs.primitives import Challenge, Response, Weight


class PyPUFUnavailableError(ImportError):
    """Raised when a PyPUF wrapper is instantiated without pypuf installed."""


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Load ``symbol_name`` from ``module_name`` or raise a dependency error."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PyPUFUnavailableError(
            "The optional 'pypuf' package is required for PyPUF wrappers."
        ) from exc
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise PyPUFUnavailableError(
            f"Installed pypuf does not expose {module_name}.{symbol_name}."
        ) from exc


def _normalise_response(response: Any) -> jax.Array:
    """Convert PyPUF's ``{-1, +1}`` or ``{0, 1}`` response to ``uint8`` bits."""
    array = np.asarray(response).reshape(-1)
    if np.any(array < 0):
        array = ((array + 1) / 2).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return jnp.asarray(array, dtype=jnp.uint8)


class _PyPUFBase(BasePUF):
    """Shared implementation for optional PyPUF adapters."""

    def __init__(self, n: int, seed: int = 0) -> None:
        """Initialise adapter metadata and create the external simulation."""
        self.n = int(n)
        self.seed = int(seed)
        self.rng = jax.random.PRNGKey(self.seed)
        self.dim = (1, self.n)
        self.weight = jnp.zeros(self.dim, dtype=jnp.float32)
        self._simulation = self._create_simulation()

    def _create_simulation(self) -> Any:
        """Create the concrete PyPUF simulation object."""
        raise NotImplementedError

    def _evaluate_external(self, challenge: Challenge) -> Response:
        """Evaluate the wrapped PyPUF simulation."""
        challenge_np = np.asarray(challenge, dtype=np.int8)
        if hasattr(self._simulation, "eval"):
            return _normalise_response(self._simulation.eval(challenge_np))
        if hasattr(self._simulation, "r_eval"):
            return _normalise_response(self._simulation.r_eval(challenge_np))
        if callable(self._simulation):
            return _normalise_response(self._simulation(challenge_np))
        raise PyPUFUnavailableError("The wrapped PyPUF object has no known evaluator.")

    def _compute_response(self, weight: Weight, challenge: Challenge) -> Response:
        """Evaluate via the external simulation; ``weight`` is unused metadata."""
        del weight
        return self._evaluate_external(challenge).reshape(-1, 1)

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"{type(self).__name__}(n={self.n}, seed={self.seed})"


class PyPUF_Arbiter(_PyPUFBase):
    """Adapter for PyPUF's Arbiter PUF simulation."""

    def _create_simulation(self) -> Any:
        cls = _load_symbol("pypuf.simulation.arbiter_based.arbiter", "ArbiterPUF")
        return cls(n=self.n, seed=self.seed)


class PyPUF_XOR(_PyPUFBase):
    """Adapter for PyPUF's XOR Arbiter PUF simulation."""

    def __init__(self, n: int, k: int, seed: int = 0) -> None:
        """Initialise a PyPUF XOR adapter."""
        self.k = int(k)
        super().__init__(n=n, seed=seed)
        self.dim = (self.k, self.n)
        self.weight = jnp.zeros(self.dim, dtype=jnp.float32)

    def _create_simulation(self) -> Any:
        cls = _load_symbol("pypuf.simulation.arbiter_based.xor_arbiter", "XORArbiterPUF")
        return cls(n=self.n, k=self.k, seed=self.seed)

    def get_response(self, challenge: Challenge):
        """Return placeholder individual responses and the PyPUF XOR response."""
        xor_response = self._evaluate_external(challenge)
        individual = jnp.zeros((challenge.shape[0], self.k), dtype=jnp.uint8)
        return individual, xor_response

    def __repr__(self) -> str:
        """Return a compact human-readable representation."""
        return f"PyPUF_XOR(n={self.n}, k={self.k}, seed={self.seed})"


class PyPUF_FeedForward(_PyPUFBase):
    """Adapter for PyPUF's Feed-Forward Arbiter PUF simulation."""

    def __init__(self, n: int, ff: Iterable[Sequence[int]], seed: int = 0) -> None:
        """Initialise a PyPUF feed-forward adapter."""
        self.ff = tuple((int(src), int(tgt)) for src, tgt in ff)
        super().__init__(n=n, seed=seed)

    def _create_simulation(self) -> Any:
        cls = _load_symbol("pypuf.simulation.arbiter_based.feed_forward", "FeedForwardArbiterPUF")
        return cls(n=self.n, ff=self.ff, seed=self.seed)


class PyPUF_Lightweight(_PyPUFBase):
    """Adapter for PyPUF's Lightweight Secure PUF simulation."""

    def __init__(self, n: int, k: int, seed: int = 0) -> None:
        """Initialise a PyPUF lightweight-secure adapter."""
        self.k = int(k)
        super().__init__(n=n, seed=seed)

    def _create_simulation(self) -> Any:
        cls = _load_symbol(
            "pypuf.simulation.arbiter_based.lightweight_secure",
            "LightweightSecurePUF",
        )
        return cls(n=self.n, k=self.k, seed=self.seed)


class PyPUF_Interpose(_PyPUFBase):
    """Adapter for PyPUF's Interpose PUF simulation."""

    def __init__(self, n: int, k_down: int, k_up: int, seed: int = 0) -> None:
        """Initialise a PyPUF interpose adapter."""
        self.k_down = int(k_down)
        self.k_up = int(k_up)
        super().__init__(n=n, seed=seed)

    def _create_simulation(self) -> Any:
        cls = _load_symbol("pypuf.simulation.arbiter_based.interpose", "InterposePUF")
        return cls(n=self.n, k_down=self.k_down, k_up=self.k_up, seed=self.seed)
