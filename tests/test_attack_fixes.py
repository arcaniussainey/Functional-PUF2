"""Regression tests for attack feature-space and feed-forward CMA-ES fixes."""

from __future__ import annotations

import warnings

import jax
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from Attack.dnn_attack_2 import PhiDNNAttack
from Attack.cmaes_attack import FeedForwardCMAESAttack
from Pufs.FunctionalPuf import Arbiter, FF_Arbiter, generate_challenges


def test_phi_dnn_attack_accepts_standard_attack_contract() -> None:
    """The Phi-feature MLP should fit and predict through BaseAttack."""
    key = jax.random.PRNGKey(0)
    puf = Arbiter(key, (1, 16))
    challenges = generate_challenges(jax.random.PRNGKey(1), (64, 16))
    responses = np.asarray(puf.get_response(challenges)).ravel()

    attack = PhiDNNAttack(hidden_layer_sizes=(8,), max_iter=5, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        attack.fit(np.asarray(challenges), responses)

    predictions = attack.predict(np.asarray(challenges[:8]))
    assert predictions.shape == (8,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_feedforward_cmaes_uses_active_prefix_dimensions() -> None:
    """The FF CMA-ES model must evaluate loop prefix arbiters, not flat Phi."""
    loops = ((2, 12),)
    puf = FF_Arbiter(0, 16, loops)
    challenges = generate_challenges(jax.random.PRNGKey(2), (32, 16))
    responses = np.asarray(puf.get_response(challenges)).ravel()

    assert FeedForwardCMAESAttack.active_parameter_count(16, loops) == 21
    true_active_vector = np.concatenate([
        np.asarray(puf.weight[0]),
        np.asarray(puf.weight[1])[:4],  # src=2 observes 3 challenge bits + bias
    ])
    predicted = FeedForwardCMAESAttack._predict_from_candidate(
        true_active_vector,
        np.asarray(challenges),
        loops,
    )
    assert np.array_equal(predicted, responses)
