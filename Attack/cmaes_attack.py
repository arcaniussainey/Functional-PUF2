"""
CMA-ES PUF attacks.

``CMAESAttack`` is the linear Arbiter-delay variant: it searches directly in
one standard ``n + 1`` Arbiter weight vector and predicts by ``sign(w^T Phi(C))``.
It is suitable for standard Arbiter PUFs and as a simple black-box baseline.

``FeedForwardCMAESAttack`` is the feed-forward variant.  It searches the active
parameters of a feed-forward Arbiter topology: one main Arbiter weight vector
plus one prefix-arbiter weight vector for each feed-forward loop.  The fitness
function evaluates the full feed-forward recurrence, so it no longer pretends
that a longer flat vector can be multiplied by ordinary ``Phi(C)``.

Both classes use the optional ``cma`` package.  Importing this module does not
require ``cma``; fitting either CMA-ES attack raises a clear error if the package
is absent.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from Attack.base_attack import BaseAttack
from Attack.lr_attack import _phi
from Pufs.feedforward_core import LoopSpecs, normalise_loops

try:  # pragma: no cover - availability depends on the caller's environment
    import cma  # type: ignore
except ImportError:  # pragma: no cover
    cma = None  # type: ignore[assignment]

CMA_AVAILABLE = cma is not None


LoopSpec = Tuple[int, int]


def _require_cma() -> None:
    """Raise a useful error when the optional CMA-ES dependency is missing."""
    if cma is None:
        raise ImportError(
            "CMA-ES attacks require the optional 'cma' package. "
            "Install it with `pip install cma` or omit CMA-ES from the experiment."
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for pseudo-probability outputs."""
    x = np.clip(x, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _arbiter_predict(weight_flat: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute binary predictions from a standard Arbiter weight vector.

    Args:
        weight_flat (np.ndarray): shape ``(n + 1,)``.
        phi (np.ndarray): shape ``(N, n + 1)``.

    Returns:
        np.ndarray: shape ``(N,)``, dtype uint8.
    """
    delta = phi @ weight_flat
    return (delta > 0).astype(np.uint8)


def _fit_cma_es(
    n_dim: int,
    sigma0: float,
    max_fevals: int,
    popsize: Optional[int],
    random_state: Optional[int],
    fitness,
) -> np.ndarray:
    """Run CMA-ES and return the best candidate observed."""
    _require_cma()
    rng = np.random.default_rng(random_state)
    x0 = rng.normal(scale=sigma0, size=n_dim).tolist()

    opts = cma.CMAOptions()
    opts["maxfevals"] = max_fevals
    opts["verbose"] = -9
    if popsize is not None:
        opts["popsize"] = popsize
    if random_state is not None:
        opts["seed"] = int(random_state)

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_w = np.asarray(x0, dtype=np.float32)
    best_fitness = float("inf")

    while not es.stop():
        candidates = es.ask()
        fitnesses = [fitness(c) for c in candidates]
        es.tell(candidates, fitnesses)
        for cand, fit in zip(candidates, fitnesses):
            if fit < best_fitness:
                best_fitness = float(fit)
                best_w = np.asarray(cand, dtype=np.float32)

    xbest = es.result.xbest
    return np.asarray(xbest, dtype=np.float32) if xbest is not None else best_w


class CMAESAttack(BaseAttack):
    """
    CMA-ES attack for the standard linear Arbiter delay model.

    The search space is exactly one ``n + 1`` weight vector.  Feed-forward PUFs
    need ``FeedForwardCMAESAttack`` because their response depends on internal
    loop arbiters and challenge mutation, not on a single ordinary feature dot
    product.

    Args:
        sigma0 (float): initial CMA-ES step size.
        max_fevals (int): maximum number of candidate fitness evaluations.
        popsize (int, optional): CMA-ES population size.  ``None`` lets the
            ``cma`` package choose its default.
        random_state (int, optional): seed for reproducibility.
        lazy_eval_size (int, optional): number of CRPs sampled per fitness
            evaluation.  ``None`` uses the full training set.
    """

    def __init__(
        self,
        sigma0: float = 2.0,
        max_fevals: int = 50_000,
        popsize: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy_eval_size: Optional[int] = 2000,
    ) -> None:
        super().__init__()
        self.sigma0 = sigma0
        self.max_fevals = max_fevals
        self.popsize = popsize
        self.random_state = random_state
        self.lazy_eval_size = lazy_eval_size
        self._best_weight: Optional[np.ndarray] = None

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Run CMA-ES over the standard Arbiter feature hyperplane."""
        phi = _phi(challenges)
        labels = np.rint(responses).astype(np.uint8)
        n_crp = phi.shape[0]
        rng = np.random.default_rng(self.random_state)

        def fitness(candidate: Sequence[float]) -> float:
            """Fraction of mispredicted CRPs; CMA-ES minimises this value."""
            w = np.asarray(candidate, dtype=np.float32)
            if self.lazy_eval_size is not None and self.lazy_eval_size < n_crp:
                idx = rng.choice(n_crp, size=self.lazy_eval_size, replace=False)
                p = phi[idx]
                r = labels[idx]
            else:
                p = phi
                r = labels
            preds = _arbiter_predict(w, p)
            return 1.0 - float(np.mean(preds == r))

        self._best_weight = _fit_cma_es(
            n_dim=phi.shape[1],
            sigma0=self.sigma0,
            max_fevals=self.max_fevals,
            popsize=self.popsize,
            random_state=self.random_state,
            fitness=fitness,
        )

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Return binary predictions using the best-found Arbiter weight."""
        return _arbiter_predict(self._best_weight, _phi(challenges))

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """Return a sigmoid transform of the signed delay margin."""
        delta = _phi(challenges) @ self._best_weight
        return _sigmoid(delta)

    def __repr__(self) -> str:
        """Return a compact description."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return f"CMAESAttack(sigma0={self.sigma0}, max_fevals={self.max_fevals}, {status})"


class FeedForwardCMAESAttack(BaseAttack):
    """
    CMA-ES attack for a single feed-forward Arbiter PUF topology.

    The candidate vector contains only active parameters:

    ``[main weight length n+1, loop_0 prefix weight length src_0+2, ...]``

    Each loop prefix weight acts on ``Phi(C[:, :src+1])`` because the loop
    arbiter observes the prefix ending at ``src`` and therefore has one bias
    coordinate.  The loop bit is converted to ``{-1,+1}`` and multiplied into
    the target challenge coordinate before later loops and the main arbiter are
    evaluated.

    Args:
        loops (Iterable[Sequence[int]]): feed-forward loop descriptors
            ``(src, tgt)`` with ``0 <= src < tgt < n_stages``.
        sigma0 (float): initial CMA-ES step size.
        max_fevals (int): maximum number of candidate fitness evaluations.
        popsize (int, optional): CMA-ES population size.
        random_state (int, optional): seed for reproducibility.
        lazy_eval_size (int, optional): number of CRPs sampled per fitness
            evaluation.  ``None`` uses the full training set.
    """

    def __init__(
        self,
        loops: Iterable[Sequence[int]],
        sigma0: float = 2.0,
        max_fevals: int = 100_000,
        popsize: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy_eval_size: Optional[int] = 2000,
    ) -> None:
        super().__init__()
        self.loops_input = tuple((int(src), int(tgt)) for src, tgt in loops)
        self.sigma0 = sigma0
        self.max_fevals = max_fevals
        self.popsize = popsize
        self.random_state = random_state
        self.lazy_eval_size = lazy_eval_size
        self.loops: Optional[LoopSpecs] = None
        self._best_weight: Optional[np.ndarray] = None

    @staticmethod
    def active_parameter_count(n_stages: int, loops: Iterable[Sequence[int]]) -> int:
        """Return the number of active scalar parameters for this topology."""
        normalised = normalise_loops(loops, n_stages)
        return (n_stages + 1) + sum(src + 2 for src, _tgt in normalised)

    @staticmethod
    def _evaluate_margin(
        candidate: np.ndarray,
        challenges: np.ndarray,
        loops: LoopSpecs,
    ) -> np.ndarray:
        """Evaluate the feed-forward recurrence and return the final margin."""
        n_stages = challenges.shape[1]
        offset = 0
        main_w = candidate[offset : offset + n_stages + 1]
        offset += n_stages + 1

        effective = challenges.astype(np.float32, copy=True)
        for src, tgt in sorted(loops, key=lambda item: item[0]):
            width = src + 2
            loop_w = candidate[offset : offset + width]
            offset += width
            prefix_phi = _phi(effective[:, : src + 1])
            loop_delta = prefix_phi @ loop_w
            loop_bit = np.where(loop_delta > 0.0, 1.0, -1.0).astype(np.float32)
            effective[:, tgt] *= loop_bit

        return _phi(effective) @ main_w

    @classmethod
    def _predict_from_candidate(
        cls,
        candidate: np.ndarray,
        challenges: np.ndarray,
        loops: LoopSpecs,
    ) -> np.ndarray:
        """Return binary responses for one candidate feed-forward model."""
        margin = cls._evaluate_margin(candidate, challenges, loops)
        return (margin > 0.0).astype(np.uint8)

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Run CMA-ES over the active feed-forward topology parameters."""
        self.loops = normalise_loops(self.loops_input, challenges.shape[1])
        labels = np.rint(responses).astype(np.uint8)
        n_crp = challenges.shape[0]
        n_dim = self.active_parameter_count(challenges.shape[1], self.loops)
        rng = np.random.default_rng(self.random_state)

        def fitness(candidate: Sequence[float]) -> float:
            """Fraction of mispredicted CRPs for a feed-forward candidate."""
            w = np.asarray(candidate, dtype=np.float32)
            if self.lazy_eval_size is not None and self.lazy_eval_size < n_crp:
                idx = rng.choice(n_crp, size=self.lazy_eval_size, replace=False)
                c = challenges[idx]
                r = labels[idx]
            else:
                c = challenges
                r = labels
            preds = self._predict_from_candidate(w, c, self.loops)
            return 1.0 - float(np.mean(preds == r))

        self._best_weight = _fit_cma_es(
            n_dim=n_dim,
            sigma0=self.sigma0,
            max_fevals=self.max_fevals,
            popsize=self.popsize,
            random_state=self.random_state,
            fitness=fitness,
        )

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Return binary predictions using the best feed-forward candidate."""
        return self._predict_from_candidate(self._best_weight, challenges, self.loops)

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """Return a sigmoid transform of the final feed-forward margin."""
        margin = self._evaluate_margin(self._best_weight, challenges, self.loops)
        return _sigmoid(margin)

    def __repr__(self) -> str:
        """Return a compact description including loop topology."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return (
            f"FeedForwardCMAESAttack(loops={self.loops_input}, sigma0={self.sigma0}, "
            f"max_fevals={self.max_fevals}, {status})"
        )
