"""
Phi-feature deep-neural-network PUF attack.

This module is the corrected DNN variant.  The network is not fed
raw challenge bits.  Each challenge is first mapped into the canonical Arbiter
PUF feature space ``Phi(C)``: suffix products of the challenge bits plus the
constant coordinate.  This is the same physical delay feature representation
used by the linear additive Arbiter model and by the logistic-regression attack.

The original raw-bit MLP remains available in ``Attack.dnn_attack`` as a broad
black-box baseline.  This implementation is the better default when the target
is known to be Arbiter-derived because it gives the network the correct physics
coordinates instead of forcing it to rediscover suffix-product structure from
raw bits.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier

from Attack.base_attack import BaseAttack
from Attack.lr_attack import _phi


class PhiDNNAttack(BaseAttack):
    """
    MLP classifier trained on Arbiter ``Phi`` features.

    ``PhiDNNAttack`` is intended for Arbiter-derived PUFs: standard Arbiter,
    XOR Arbiter, and other constructions whose useful attacker-side features
    are the additive-delay suffix products.  The network receives ``n + 1``
    inputs for an ``n``-stage challenge, not the raw ``n`` challenge bits.

    Args:
        hidden_layer_sizes (Sequence[int]): number of neurons per hidden layer.
            Defaults to ``(128, 64)`` for 64-stage experiments.
        activation (str): sklearn activation name, typically ``'tanh'`` or
            ``'relu'``.  ``'tanh'`` is a stable default for signed Phi features.
        max_iter (int): maximum training epochs.
        learning_rate_init (float): initial Adam learning rate.
        random_state (int, optional): seed for reproducible initialisation.
        early_stopping (bool): whether sklearn should reserve validation data
            and stop when validation score stops improving.
        alpha (float): L2 penalty used by sklearn's MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (128, 64),
        activation: str = "tanh",
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        random_state: Optional[int] = None,
        early_stopping: bool = False,
        alpha: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.alpha = alpha
        self._model: Optional[MLPClassifier] = None

    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """
        Train the MLP on ``Phi(C)`` instead of raw challenge bits.

        Responses are rounded before casting so noisy or float32 response arrays
        that are numerically close to ``0`` or ``1`` remain valid class labels.
        """
        phi = _phi(challenges)
        labels = np.rint(responses).astype(np.int32)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            alpha=self.alpha,
        )
        self._model.fit(phi, labels)

    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """Return hard binary predictions from the Phi-feature MLP."""
        return self._model.predict(_phi(challenges)).astype(np.uint8)

    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """Return the probability of response 1 for each challenge."""
        proba = self._model.predict_proba(_phi(challenges))
        classes = [int(round(float(c))) for c in self._model.classes_]
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return proba[:, -1].astype(np.float32)

    def __repr__(self) -> str:
        """Return a compact description including the feature-space choice."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return (
            f"PhiDNNAttack(hidden={self.hidden_layer_sizes}, act={self.activation}, "
            f"features=Phi, {status})"
        )


# Backwards-friendly aliases for experiment files that refer to the second DNN
# attack by filename rather than by the more descriptive class name.
DNNAttack2 = PhiDNNAttack
PhiFeatureDNNAttack = PhiDNNAttack
