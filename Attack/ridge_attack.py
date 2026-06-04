"""
Linear Ridge-regression PUF attack.

Ridge regression treats the binary PUF response as a continuous ``{0, 1}``
target and minimises mean-squared error with an L2 weight penalty.  The
predicted score is thresholded at 0.5 to produce a binary prediction.

For a standard Arbiter PUF the response is a linear threshold function of the
Phi feature vector, so Ridge generalises well with few CRPs.  For XOR and
Feed-Forward PUFs the decision surface is non-linear and Ridge accuracy
degrades accordingly -- useful as a baseline rather than a primary attack.

The Phi feature transform is the same suffix-product mapping used by
``LRAttack``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from Attack.base_attack import BaseAttack


def _phi(challenges: np.ndarray) -> np.ndarray:
    """
    Compute the Arbiter-PUF feature transform in NumPy.

    Args:
        challenges (np.ndarray): shape ``(N, n)``, dtype float32.

    Returns:
        np.ndarray: shape ``(N, n + 1)``, dtype float32.
    """
    variable = np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1]
    bias = np.ones((challenges.shape[0], 1), dtype=np.float32)
    return np.hstack([variable, bias])


class RidgeAttack(BaseAttack):
    """
    Ridge-regression attack on Arbiter-type PUFs.

    Ridge is the fastest model in this collection and makes a good sanity-check
    baseline.  It excels on standard Arbiter PUFs where the problem is exactly
    linear in Phi space.

    Args:
        alpha (float): L2 regularisation strength.  Smaller values fit more
            aggressively.  Defaults to ``1e-3``.
        fit_intercept (bool): whether to include a separate intercept term.
            Defaults to ``False`` because the bias is already encoded in the
            last coordinate of Phi.
        max_iter (int, optional): maximum conjugate-gradient iterations.
            ``None`` means sklearn chooses automatically.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        fit_intercept: bool = False,
        max_iter: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self._model: Optional[Ridge] = None

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """Fit a Ridge regressor to the Phi-transformed challenges."""
        phi = _phi(challenges)
        self._model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
        )
        self._model.fit(phi, responses)

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Threshold the continuous Ridge score at 0.5 to produce binary labels."""
        scores = self._model.predict(_phi(challenges))
        return (scores >= 0.5).astype(np.uint8)

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """
        Return the clipped continuous score as a pseudo-probability.

        Ridge scores are not calibrated probabilities; they are clipped to
        ``[0, 1]`` so that callers expecting a probability estimate receive a
        valid value.
        """
        scores = self._model.predict(_phi(challenges))
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def __repr__(self) -> str:
        """Return a compact description including key hyperparameters."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return f"RidgeAttack(alpha={self.alpha}, {status})"
