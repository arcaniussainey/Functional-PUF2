"""
Deep-neural-network PUF attack.

A multi-layer perceptron trained via Adam on the raw ``{-1, +1}`` challenge
bits (no Phi transform applied).  Neural networks can in principle learn the
non-linear structure of XOR and Feed-Forward PUFs without hand-crafted
features, though they typically require more CRPs than logistic regression on
standard Arbiter PUFs.

The raw challenge bits are used as input rather than the Phi feature vector.
This is intentional: the Phi transform encodes the linear Arbiter model
assumption explicitly, which helps LR but removes the network's ability to
learn alternative feature representations.

Architecture defaults are chosen to be practically useful across all three PUF
types (Arbiter, XOR, FF) with 64-bit challenges.  Larger PUFs or more complex
XOR configurations will benefit from wider or deeper networks.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier

from Attack.base_attack import BaseAttack


class DNNAttack(BaseAttack):
    """
    MLP classifier attack on Arbiter, XOR Arbiter, and Feed-Forward PUFs.

    The network is trained on raw ``{-1, +1}`` challenge bits.  Unlike
    ``LRAttack`` no Phi transform is applied; the network discovers its own
    internal representations.

    Args:
        hidden_layer_sizes (Sequence[int]): number of neurons per hidden layer.
            Defaults to ``(128, 64)`` -- two hidden layers suitable for 64-bit
            PUFs with moderate XOR counts.
        activation (str): sklearn activation name, one of ``'relu'``,
            ``'tanh'``, ``'logistic'``.
        max_iter (int): maximum training epochs.
        learning_rate_init (float): initial Adam learning rate.
        random_state (int, optional): seed for reproducibility.
        early_stopping (bool): whether to hold out a validation fraction and
            stop early.  Useful when the CRP budget is large.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (128, 64),
        activation: str = "relu",
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        random_state: Optional[int] = None,
        early_stopping: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.early_stopping = early_stopping
        self._model: Optional[MLPClassifier] = None

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """
        Train the MLP on raw challenge bits.

        Responses are cast to integer labels ``{0, 1}`` because sklearn's
        MLPClassifier expects integer class labels for classification mode.
        """
        y = responses.astype(np.int32)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
        )
        self._model.fit(challenges, y)

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Return hard binary class predictions."""
        return self._model.predict(challenges).astype(np.uint8)

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """Return the probability of response 1 for each challenge."""
        proba = self._model.predict_proba(challenges)
        classes = list(self._model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return proba[:, -1].astype(np.float32)

    def __repr__(self) -> str:
        """Return a compact description including architecture."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return (
            f"DNNAttack(hidden={self.hidden_layer_sizes}, "
            f"act={self.activation}, {status})"
        )
