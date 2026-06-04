"""
Logistic-regression PUF attack.

This follows the Rührmair et al. formulation directly: the challenge vector is
first mapped to the Arbiter feature space ``Phi(C)`` (suffix-product
coordinates plus a bias term) and then a logistic classifier is trained to
separate the two response classes.  For XOR PUFs the product-of-signs
structure means the effective decision boundary is non-linear in ``Phi``
coordinates; the sklearn solver navigates this through iterated restarts when
``n_restarts > 1``.

The feature transform used here is identical to ``phi_from_challenges`` in
``Pufs.primitives``, implemented in pure NumPy so that no JAX tracing is
required during sklearn's gradient steps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from Attack.base_attack import BaseAttack


def _phi(challenges: np.ndarray) -> np.ndarray:
    """
    Compute the Arbiter-PUF feature transform in NumPy.

    Maps ``{-1, +1}`` challenge bits to suffix-product coordinates plus a
    trailing constant bias coordinate, replicating ``phi_from_challenges``.

    Args:
        challenges (np.ndarray): shape ``(N, n)``, dtype float32.

    Returns:
        np.ndarray: shape ``(N, n + 1)``, dtype float32.
    """
    variable = np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1]
    bias = np.ones((challenges.shape[0], 1), dtype=np.float32)
    return np.hstack([variable, bias])


class LRAttack(BaseAttack):
    """
    Logistic-regression attack on Arbiter and XOR Arbiter PUFs.

    The solver is L-BFGS-B (sklearn ``solver='lbfgs'``) which converges
    quickly on the convex single-Arbiter problem and also works for XOR PUFs
    (non-convex in Phi space) with restarts.

    For a standard Arbiter PUF a single restart almost always suffices.  For
    k-XOR PUFs use ``n_restarts >= 5`` and allow a larger ``max_iter``; the
    paper reports that 5-XOR 128-bit PUFs typically need hundreds of thousands
    of CRPs.

    Args:
        C (float): inverse regularisation strength.  Larger values mean less
            regularisation.  Defaults to ``1e4`` (very weak regularisation),
            matching typical attack configurations.
        max_iter (int): maximum solver iterations per restart.
        n_restarts (int): number of independent optimisation runs.  The run
            with the highest training accuracy is kept.
        random_state (int, optional): seed for sklearn's internal RNG.
    """

    def __init__(
        self,
        C: float = 1e4,
        max_iter: int = 2000,
        n_restarts: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.random_state = random_state
        self._model: Optional[LogisticRegression] = None

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Train over ``n_restarts`` independent runs and keep the best."""
        phi = _phi(challenges)
        # sklearn LogisticRegression accepts float labels but integer labels are
        # unambiguous; round to eliminate any floating-point rounding from the
        # base class float32 conversion.
        labels = np.rint(responses).astype(np.int32)
        best_model = None
        best_acc = -1.0

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_restarts):
            seed = int(rng.integers(0, 2**31))
            model = LogisticRegression(
                C=self.C,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=seed,
                fit_intercept=False,
            )
            model.fit(phi, labels)
            acc = float(np.mean(model.predict(phi) == labels))
            if acc > best_acc:
                best_acc = acc
                best_model = model

        self._model = best_model

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Return hard binary class predictions via the Phi transform."""
        return self._model.predict(_phi(challenges)).astype(np.uint8)

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """Return the probability of response 1 for each challenge."""
        proba = self._model.predict_proba(_phi(challenges))
        # classes_ may be int or float depending on sklearn version; compare by value
        classes = [int(round(float(c))) for c in self._model.classes_]
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return proba[:, -1].astype(np.float32)

    def __repr__(self) -> str:
        """Return a compact description including key hyperparameters."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return (
            f"LRAttack(C={self.C}, max_iter={self.max_iter}, "
            f"n_restarts={self.n_restarts}, {status})"
        )
