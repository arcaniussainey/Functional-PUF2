"""
Reusable attack-state containers for PUF attack code.

These classes are deliberately lightweight.  They keep experiment state such as
challenge samples, logging buffers, deterministic key streams, and run metadata
out of individual attack functions, without smuggling mutable global state into
JAX-transformed fitness functions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax

from Pufs.primitives import Challenge, PRNGKey
from Pufs.randomness import DeterministicKeyStream


@dataclass(frozen=True)
class AttackRunConfig:
    """
    Immutable high-level configuration for evolutionary PUF attacks.

    Existing FunAttack functions still expose their historical keyword
    arguments.  This config is a structured alternative for new orchestration
    code and examples.
    """

    pop_size: int = 32
    n_generations: int = 500
    model_name: str = "CMA_ES"
    threshold: float = 2.0
    seed: int = 0
    noise: bool = False
    alphas: Tuple[float, ...] = ()

    def attack_kwargs(self) -> Dict[str, object]:
        """Return legacy keyword arguments accepted by x1_attk/xn_attk."""
        return {
            "pop_size": self.pop_size,
            "n_generations": self.n_generations,
            "model_name": self.model_name,
            "threshold": self.threshold,
        }


@dataclass
class AttackContext:
    """
    Runtime context threaded through attack functions.

    Attributes
    ----------
    c_sample:
        Sample challenge set used to estimate delay standard deviation in
        challenge-filter fitness calculations.
    log_path:
        Destination CSV path for buffered logs.
    key_stream:
        Deterministic PRNG stream controlling stochastic attack phases.
    """

    c_sample: Challenge
    log_path: str = "Funlog/log.csv"
    key_stream: DeterministicKeyStream = field(default_factory=DeterministicKeyStream, repr=False)
    _log: List[Dict] = field(default_factory=list, repr=False)

    def new_key(self) -> PRNGKey:
        """Return the next deterministic PRNG key owned by this context."""
        return self.key_stream.next()

    def split_keys(self, n: int) -> Tuple[PRNGKey, ...]:
        """Return ``n`` sibling keys from the context stream."""
        return self.key_stream.split(n)

    def fold_in(self, data: int) -> PRNGKey:
        """
        Return a deterministic key folded with integer metadata.

        This is useful for deterministic random exploration when the caller
        wants key derivation tied to a generation number, retry number, or
        experiment index while preserving a single parent stream.
        """
        return jax.random.fold_in(self.new_key(), int(data))

    def log_data(self, data: Dict) -> None:
        """Append a structured log row to the context-local log buffer."""
        self._log.append(data)

    def flush_log(self) -> None:
        """Write buffered log entries to ``self.log_path`` and clear them."""
        if not self._log:
            return
        with open(self.log_path, "a+", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._log[0].keys())
            writer.writeheader()
            writer.writerows(self._log)
        self._log.clear()


@dataclass(frozen=True)
class AttackTiming:
    """Small immutable timing bundle for attack result dictionaries."""

    setup_time: float = 0.0
    optimization_time: float = 0.0
    validation_time: float = 0.0
    wall_time: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Return a serialisable dictionary representation."""
        return {
            "setup_time": self.setup_time,
            "optimization_time": self.optimization_time,
            "validation_time": self.validation_time,
            "wall_time": self.wall_time,
        }
