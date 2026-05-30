"""Regression tests for reusable randomness and attack-state helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Pufs.attack_state import AttackContext, AttackRunConfig, AttackTiming
from Pufs.randomness import DeterministicKeyStream, DistributionSpec, sample_distribution
from Pufs.primitives import generate_challenges


def test_deterministic_stream_replays_same_keys() -> None:
    """Two streams with the same seed should produce the same key sequence."""
    a = DeterministicKeyStream(seed=123)
    b = DeterministicKeyStream(seed=123)

    for _ in range(5):
        assert jnp.array_equal(a.next(), b.next())

    assert a.count == 5
    assert b.count == 5


def test_stream_split_is_reproducible_and_advances_once() -> None:
    """split(n) should produce stable sibling keys and track consumption."""
    a = DeterministicKeyStream(seed=7)
    b = DeterministicKeyStream(seed=7)

    keys_a = a.split(3)
    keys_b = b.split(3)

    assert len(keys_a) == 3
    assert a.count == 1
    for left, right in zip(keys_a, keys_b):
        assert jnp.array_equal(left, right)


def test_distribution_spec_samples_supported_families() -> None:
    """DistributionSpec provides a stable documented sampling surface."""
    key = jax.random.PRNGKey(0)

    normal = DistributionSpec.normal(scale=0.1).sample(key, (4, 3))
    signed = DistributionSpec.signed_bernoulli().sample(key, (4, 3))
    ints = sample_distribution(key, DistributionSpec.integers(-2, 3), (4, 3))

    assert normal.shape == (4, 3)
    assert signed.shape == (4, 3)
    assert set(jnp.unique(signed).tolist()).issubset({-1, 1})
    assert ints.min() >= -2
    assert ints.max() < 3


def test_attack_context_owns_deterministic_keys() -> None:
    """AttackContext should be reusable without relying on global PRNG state."""
    stream = DeterministicKeyStream(seed=22)
    c_sample = generate_challenges(stream.next(), (8, 4))
    ctx_a = AttackContext(c_sample=c_sample, key_stream=DeterministicKeyStream(seed=99))
    ctx_b = AttackContext(c_sample=c_sample, key_stream=DeterministicKeyStream(seed=99))

    assert jnp.array_equal(ctx_a.new_key(), ctx_b.new_key())
    assert jnp.array_equal(ctx_a.fold_in(3), ctx_b.fold_in(3))


def test_attack_config_and_timing_are_serialisable_helpers() -> None:
    """Structured attack metadata should map cleanly to legacy dictionaries."""
    cfg = AttackRunConfig(pop_size=8, n_generations=10, threshold=0.75, seed=5)
    timing = AttackTiming(setup_time=1.0, optimization_time=2.0)

    assert cfg.attack_kwargs() == {
        "pop_size": 8,
        "n_generations": 10,
        "model_name": "CMA_ES",
        "threshold": 0.75,
    }
    assert timing.as_dict()["optimization_time"] == 2.0
