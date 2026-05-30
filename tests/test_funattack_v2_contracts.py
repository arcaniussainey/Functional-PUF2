"""
Focused contracts for FunAttack_v2 refactor behaviour.

These tests avoid requiring evosax by installing a tiny import-time stub; the
attack loop itself is monkeypatched so we can test refactor contracts without
running a long evolutionary optimisation.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

# FunAttack_v2 imports ``Strategies`` at module import time.  These tests do
# not execute evosax strategies, so a minimal stub is sufficient.
sys.modules.setdefault("evosax", types.SimpleNamespace(Strategies={}))

from Attack import FunAttack_v2 as fa  # pylint: disable=wrong-import-position
from Pufs.FunctionalPuf import Xor, generate_challenges  # pylint: disable=wrong-import-position


def test_deterministic_key_stream_reproducible() -> None:
    """Same seed must produce the same explicit PRNG sequence."""
    a = fa.DeterministicKeyStream(seed=17)
    b = fa.DeterministicKeyStream(seed=17)
    for _ in range(8):
        np.testing.assert_array_equal(a.next(), b.next())


def test_transform_flip_roundtrip() -> None:
    """Vectorised iPUF transform must invert exactly."""
    key = jax.random.PRNGKey(0)
    c = generate_challenges(key, (16, 8))
    r = jnp.array([0, 1] * 8, dtype=jnp.uint8)
    transformed = fa.transform_flip(c, r)
    restored = fa.reverse_transform(transformed)
    np.testing.assert_array_equal(restored, c)
    assert transformed.shape == (16, 9)


def test_attack_indv_puf_does_not_mutate_attk_args(monkeypatch) -> None:
    """Per-arbiter alpha is added to local copies only, never caller kwargs."""
    chall = generate_challenges(jax.random.PRNGKey(1), (8, 4))
    ctx = fa.AttackContext(c_sample=chall, key_stream=fa.DeterministicKeyStream(seed=2))
    attk_args = {
        "pop_size": 4,
        "n_generations": 1,
        "model_name": "stub",
        "threshold": 0.5,
    }
    original = dict(attk_args)
    seen_xn_alphas = []

    def fake_x1_attk(challenges, ctx_arg, **kwargs):
        assert challenges.shape == chall.shape
        assert ctx_arg is ctx
        assert "alpha" not in kwargs
        state = SimpleNamespace(best_fitness=jnp.float32(0.0), best_member=jnp.ones((4,)))
        return state, jnp.ones((1, 4))

    def fake_xn_attk(challenges, prev_w, ctx_arg, **kwargs):
        assert challenges.shape == chall.shape
        assert prev_w.shape[1] == 4
        assert ctx_arg is ctx
        assert "alpha" in kwargs
        seen_xn_alphas.append(kwargs["alpha"])
        state = SimpleNamespace(best_fitness=jnp.float32(0.0), best_member=jnp.ones((4,)))
        return state, jnp.ones((1, 4))

    monkeypatch.setattr(fa, "x1_attk", fake_x1_attk)
    monkeypatch.setattr(fa, "xn_attk", fake_xn_attk)

    lrn_w, best_fitness, total_time = fa.attack_indv_puf(
        chall,
        attk_args,
        noise=False,
        xor_puf=None,  # not used when noise=False
        nxor=3,
        alphas=[0.1, 0.2],
        ctx=ctx,
    )

    assert attk_args == original
    assert seen_xn_alphas == [0.1, 0.2]
    assert lrn_w.shape == (3, 4)
    assert len(best_fitness) == 3
    assert total_time >= 0


def test_initialize_challenge_filter_reproducible_with_seed() -> None:
    """Explicit key streams make challenge filtering reproducible."""
    puf_key = jax.random.PRNGKey(3)
    puf = Xor(puf_key, (2, 8))
    stream_a = fa.DeterministicKeyStream(seed=4)
    stream_b = fa.DeterministicKeyStream(seed=4)

    ca = fa.initialize_challenge_filter(8, puf, 0.1, stream_a)
    cb = fa.initialize_challenge_filter(8, puf, 0.1, stream_b)

    np.testing.assert_array_equal(ca, cb)
    assert ca.shape == (8, 8)
