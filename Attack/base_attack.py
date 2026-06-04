"""
Abstract base class for all PUF attack models.

An attack model accepts a set of challenge-response pairs (CRPs) collected
from a target PUF and learns a function that predicts the PUF's response to
arbitrary unseen challenges.  The learned model can then impersonate the PUF
or serve as a reference for verifier-side distinguishing experiments.

All concrete attack implementations must subclass ``BaseAttack`` and implement
``_fit``, ``_predict_proba``, and ``_predict``.

Conventions
-----------
* Challenges are expected in ``{-1, +1}`` encoding, shape ``(N, n)``, matching
  the rest of the library.  Internally an attack may convert these to any
  feature representation it needs.
* Responses are treated as binary labels in ``{0, 1}`` (uint8), shape ``(N,)``.
  For XOR PUFs this is the XOR-reduced scalar response.
* ``fit`` / ``predict`` accept and return plain ``numpy.ndarray`` or ``jax.Array``
  interchangeably; output is always a plain ``numpy.ndarray``.
"""

from __future__ import annotations

import abc
import pickle
from pathlib import Path
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np

ArrayLike = Union[np.ndarray, jnp.ndarray]


class BaseAttack(abc.ABC):
    """
    Abstract interface shared by all PUF attack models.

    Subclasses implement ``_fit``, ``_predict_proba``, and ``_predict``.
    The public ``fit``, ``predict``, ``predict_proba``, ``save``, and ``load``
    methods handle input normalisation, state tracking, and persistence.
    """

    def __init__(self) -> None:
        """Initialise a fresh, unfitted attack model."""
        self._fitted: bool = False
        self._n_stages: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted to at least one CRP set."""
        return self._fitted

    @property
    def n_stages(self) -> Optional[int]:
        """Return the challenge bit-length seen during fitting, or None if unfitted."""
        return self._n_stages

    def fit(self, challenges: ArrayLike, responses: ArrayLike) -> "BaseAttack":
        """
        Fit the attack model to observed challenge-response pairs.

        The input challenges must be in ``{-1, +1}`` encoding, matching the
        library convention.  Responses must be binary ``{0, 1}``.

        Args:
            challenges (ArrayLike): challenge matrix, shape ``(N, n)``.
            responses (ArrayLike): binary response vector, shape ``(N,)`` or
                ``(N, 1)``.

        Returns:
            BaseAttack: ``self``, to allow method chaining.
        """
        c = np.asarray(challenges, dtype=np.float32)
        r = np.asarray(responses, dtype=np.float32).ravel()
        if c.shape[0] != r.shape[0]:
            raise ValueError(
                f"challenges and responses must have the same number of rows; "
                f"got {c.shape[0]} vs {r.shape[0]}."
            )
        self._n_stages = c.shape[1]
        self._fit(c, r)
        self._fitted = True
        return self

    def predict(self, challenges: ArrayLike) -> np.ndarray:
        """
        Predict binary responses ``{0, 1}`` for a batch of challenges.

        Args:
            challenges (ArrayLike): challenge matrix, shape ``(N, n)``.

        Returns:
            np.ndarray: predicted binary responses, shape ``(N,)``, dtype uint8.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        self._check_fitted()
        c = np.asarray(challenges, dtype=np.float32)
        return self._predict(c).astype(np.uint8)

    def predict_proba(self, challenges: ArrayLike) -> np.ndarray:
        """
        Return the estimated probability of response ``1`` for each challenge.

        Args:
            challenges (ArrayLike): challenge matrix, shape ``(N, n)``.

        Returns:
            np.ndarray: probability estimates, shape ``(N,)``, dtype float32.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        self._check_fitted()
        c = np.asarray(challenges, dtype=np.float32)
        return self._predict_proba(c).astype(np.float32)

    def accuracy(self, challenges: ArrayLike, responses: ArrayLike) -> float:
        """
        Compute prediction accuracy against ground-truth responses.

        Args:
            challenges (ArrayLike): challenge matrix, shape ``(N, n)``.
            responses (ArrayLike): ground-truth binary responses, shape ``(N,)``.

        Returns:
            float: fraction of correctly predicted responses in ``[0, 1]``.
        """
        r_true = np.asarray(responses, dtype=np.uint8).ravel()
        r_pred = self.predict(challenges)
        return float(np.mean(r_pred == r_true))

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the fitted model to disk using pickle.

        The entire attack object (including any fitted sklearn/numpy state) is
        stored so that ``load`` can reconstruct an identical predictor.

        Args:
            path (Union[str, Path]): destination file path.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        self._check_fitted()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseAttack":
        """
        Deserialise a previously saved attack model.

        The file must have been written by ``save``.  The returned object is
        ready to call ``predict`` without a second ``fit``.

        Args:
            path (Union[str, Path]): source file path.

        Returns:
            BaseAttack: the deserialised attack model.
        """
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BaseAttack):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, expected a BaseAttack subclass."
            )
        return obj

    def _check_fitted(self) -> None:
        """Raise if the model has not yet been fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit() first."
            )

    @abc.abstractmethod
    def _fit(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """
        Internal training routine.  Input arrays are already normalised numpy
        float32 with responses ravelled to shape ``(N,)``.

        Args:
            challenges (np.ndarray): shape ``(N, n)``, dtype float32.
            responses (np.ndarray): shape ``(N,)``, dtype float32, values in {0, 1}.
        """

    @abc.abstractmethod
    def _predict(self, challenges: np.ndarray) -> np.ndarray:
        """
        Return hard binary predictions for normalised challenges.

        Args:
            challenges (np.ndarray): shape ``(N, n)``, dtype float32.

        Returns:
            np.ndarray: shape ``(N,)``, binary values.
        """

    @abc.abstractmethod
    def _predict_proba(self, challenges: np.ndarray) -> np.ndarray:
        """
        Return probability-of-1 estimates for normalised challenges.

        Args:
            challenges (np.ndarray): shape ``(N, n)``, dtype float32.

        Returns:
            np.ndarray: shape ``(N,)``, values in ``[0, 1]``.
        """

    def __repr__(self) -> str:
        """Return a compact human-readable description."""
        status = f"fitted, n_stages={self._n_stages}" if self._fitted else "unfitted"
        return f"{type(self).__name__}({status})"
