"""
tests/test_functional_puf.py
Run from the repo root:  pytest tests/ -v

BUG tests   -- fail on the original code, pass once the bug is fixed.
DRIFT tests -- pin correct behaviour; catch regressions during refactoring.
"""
# pylint: disable=redefined-outer-name
# Rationale: pytest injects fixtures by matching parameter names to fixture
# function names.  Pylint sees the parameter shadow the module-level fixture
# function and raises W0621, but this is intentional and correct pytest usage.
# pylint: disable=import-outside-toplevel
# Rationale: TestGlobalPrng.test_extra_call_changes_sequence deliberately
# uses a lazy import to load a fresh, isolated module instance mid-test.
# pylint: disable=too-few-public-methods
# Rationale: pytest test classes are single-responsibility by design; they
# are not general-purpose classes and do not need two public methods.

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import (
    Arbiter, Xor,
    generate_weights, generate_challenges, generate_mem_weights,
    get_response, get_delta_response, xor_get_response,
    noisy_get_response,
    n_new_keys, row_vec,
)

KEY = jax.random.PRNGKey(0)



# Shared fixtures


@pytest.fixture(scope="module")
def w1x32() -> jax.Array:
    """Single arbiter weight (1, 32)."""
    _, sk = jax.random.split(KEY)
    return generate_weights(sk, (1, 32))


@pytest.fixture(scope="module")
def w3x32() -> jax.Array:
    """Three-arbiter weight (3, 32)."""
    _, sk = jax.random.split(KEY)
    return generate_weights(sk, (3, 32))


@pytest.fixture(scope="module")
def c100x32() -> jax.Array:
    """100 challenges of width 32."""
    _, _, sk = jax.random.split(KEY, 3)
    return generate_challenges(sk, (100, 32))



# XOR correctness


class TestXorResponse:
    """XOR correctness: xor_get_response must equal per-arbiter XOR."""

    def test_xor_matches_manual(
        self,
        w3x32: jax.Array,
        c100x32: jax.Array,
    ) -> None:
        """XOR result must equal bitwise XOR of individual get_response calls."""
        _, xored = xor_get_response(w3x32, c100x32)
        r = [get_response(row_vec(w3x32[i]), c100x32).flatten() for i in range(3)]
        expected = np.bitwise_xor(
            np.bitwise_xor(np.array(r[0]), np.array(r[1])), np.array(r[2])
        )
        np.testing.assert_array_equal(np.array(xored).flatten(), expected)



# Memory weight range


class TestMemWeightsRange:
    """
    randint upper bound is ``v-1`` but JAX randint is exclusive,
    so the maximum value ``v-1`` is never produced -- actual max is ``v-2``.
    """

    @pytest.mark.parametrize("w_bits,expected_max", [(3, 3), (4, 7), (5, 15)])
    def test_max_reachable(self, w_bits: int, expected_max: int) -> None:
        """Identify the reachable range of randint for each bit-width."""
        _, sk = jax.random.split(KEY)
        weights = generate_mem_weights(sk, (1, 100_000), w_bits)
        assert int(weights.max()) == expected_max, (
            f"w={w_bits}: max={int(weights.max())}, expected {expected_max}"
        )

    @pytest.mark.parametrize("w_bits", [3, 4, 5])
    def test_min(self, w_bits: int) -> None:
        """Minimum value must equal -(2 ** (w_bits - 1))."""
        _, sk = jax.random.split(KEY)
        assert int(generate_mem_weights(sk, (1, 100_000), w_bits).min()) == -(2 ** (w_bits - 1))



# Global PRNG fragility test


class TestGlobalPrng:
    """new_key() is a stateful module-level generator -- document its fragility."""

    def test_extra_call_changes_sequence(self) -> None:
        """Demonstrates fragility: injecting an extra new_key() shifts the sequence."""
        import importlib
        from pathlib import Path

        def load_fresh() -> object:
            """Load a fresh, independent instance of FunctionalPuf."""
            name = f"_FunctionalPuf_fresh_{id(object())}"
            spec = importlib.util.spec_from_file_location(
                name, Path(__file__).parent.parent / "Pufs" / "FunctionalPuf.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod

        mod_a, mod_b = load_fresh(), load_fresh()
        k1a, k2a = mod_a.new_key(), mod_a.new_key()  # type: ignore[attr-defined]
        mod_b.new_key()  # type: ignore[attr-defined]  # deliberate extra call
        k1b, k2b = mod_b.new_key(), mod_b.new_key()  # type: ignore[attr-defined]

        # Passing this test is intentionally BAD -- it means the sequences diverged
        assert not (jnp.array_equal(k1a, k1b) and jnp.array_equal(k2a, k2b))

    def test_n_new_keys_is_deterministic(self) -> None:
        """n_new_keys must produce the same output given the same PRNG key."""
        rng = jax.random.PRNGKey(99)
        _, ka = n_new_keys(rng, 4)
        _, kb = n_new_keys(rng, 4)
        np.testing.assert_array_equal(ka, kb)



# Drift tests -- pin correct behaviour against regressions


class TestDriftChallenges:
    """Pin generate_challenges behaviour."""

    def test_shape(self) -> None:
        """Output shape must match the requested dim."""
        _, sk = jax.random.split(KEY)
        assert generate_challenges(sk, (50, 64)).shape == (50, 64)

    def test_values_pm1(self) -> None:
        """All values must be in {-1, +1}."""
        _, sk = jax.random.split(KEY)
        assert set(np.unique(np.array(generate_challenges(sk, (500, 32))))).issubset({-1, 1})

    def test_dtype(self) -> None:
        """dtype must be int8."""
        _, sk = jax.random.split(KEY)
        assert generate_challenges(sk, (10, 8)).dtype == jnp.int8

    def test_balanced(self) -> None:
        """Fraction of +1 entries must be close to 0.5."""
        _, sk = jax.random.split(KEY)
        frac = float(jnp.mean(generate_challenges(sk, (5000, 64)) == 1))
        assert 0.45 < frac < 0.55

    def test_deterministic(self) -> None:
        """Same key must produce the same challenges."""
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            generate_challenges(sk, (20, 16)),
            generate_challenges(sk, (20, 16)),
        )


class TestDriftWeights:
    """Pin generate_weights behaviour."""

    def test_shape(self) -> None:
        """Output shape must match (k, n)."""
        _, sk = jax.random.split(KEY)
        assert generate_weights(sk, (3, 64)).shape == (3, 64)

    def test_dtype(self) -> None:
        """dtype must be float32."""
        _, sk = jax.random.split(KEY)
        assert generate_weights(sk, (2, 32)).dtype == jnp.float32

    def test_rows_independent(self) -> None:
        """Each arbiter row must be distinct."""
        _, sk = jax.random.split(KEY)
        w_mat = generate_weights(sk, (4, 32))
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(w_mat[i], w_mat[j])

    def test_deterministic(self) -> None:
        """Same key must produce the same weights."""
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            generate_weights(sk, (3, 32)),
            generate_weights(sk, (3, 32)),
        )


class TestDriftGetResponse:
    """Pin get_response behaviour."""

    def test_shape_single(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Single-arbiter response must have shape (N, 1)."""
        assert get_response(w1x32, c100x32).shape == (100, 1)

    def test_shape_multi(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """Multi-arbiter response must have shape (N, k)."""
        assert get_response(w3x32, c100x32).shape == (100, 3)

    def test_binary(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """All values must be in {0, 1}."""
        assert set(np.unique(np.array(get_response(w3x32, c100x32)))).issubset({0, 1})

    def test_dtype(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """dtype must be uint8."""
        assert get_response(w1x32, c100x32).dtype == jnp.uint8

    def test_deterministic(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """Same inputs must produce the same responses."""
        np.testing.assert_array_equal(
            get_response(w3x32, c100x32),
            get_response(w3x32, c100x32),
        )

    def test_formula(self) -> None:
        """Response formula: sign(w·c) > 0 must match integer dot-product check."""
        sk1, sk2 = jax.random.split(KEY, 2)
        w_mat = generate_weights(sk1, (2, 8))
        chall = generate_challenges(sk2, (5, 8)).astype(jnp.float32)
        resp = get_response(w_mat, chall)
        for i in range(5):
            for j in range(2):
                assert int(resp[i, j]) == int(np.dot(np.array(w_mat[j]), np.array(chall[i])) > 0)

    def test_balanced(self, w1x32: jax.Array) -> None:
        """Response fraction should be near 0.5 for a typical weight."""
        _, sk = jax.random.split(KEY)
        chall = generate_challenges(sk, (2000, 32))
        assert 0.42 < float(jnp.mean(get_response(w1x32, chall))) < 0.58


class TestDriftDeltaResponse:
    """Pin get_delta_response behaviour."""

    def test_shape(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """Delta response shape must be (N, k)."""
        assert get_delta_response(w3x32, c100x32).shape == (100, 3)

    def test_dtype_float(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Delta must have a floating-point dtype."""
        assert jnp.issubdtype(get_delta_response(w1x32, c100x32).dtype, jnp.floating)

    def test_sign_consistent(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """delta > 0.5 must agree with binary response."""
        delta = get_delta_response(w3x32, c100x32)
        resp  = get_response(w3x32, c100x32)
        np.testing.assert_array_equal(np.array(delta > 0.5), np.array(resp))


class TestDriftXorGetResponse:
    """Pin xor_get_response behaviour."""

    def test_returns_two_tuple(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """Must return a 2-tuple."""
        result = xor_get_response(w3x32, c100x32)
        assert isinstance(result, tuple) and len(result) == 2

    def test_xor_binary(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """XOR values must be in {0, 1}."""
        _, xored = xor_get_response(w3x32, c100x32)
        assert set(np.unique(np.array(xored))).issubset({0, 1})

    def test_xor_size(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """XOR vector must have one entry per challenge."""
        _, xored = xor_get_response(w3x32, c100x32)
        assert xored.size == 100

    def test_xor_deterministic(self, w3x32: jax.Array, c100x32: jax.Array) -> None:
        """Same inputs must produce the same XOR output."""
        _, x1 = xor_get_response(w3x32, c100x32)
        _, x2 = xor_get_response(w3x32, c100x32)
        np.testing.assert_array_equal(x1, x2)


class TestDriftNoisyResponse:
    """Pin noisy_get_response behaviour."""

    def test_shape(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Noisy response shape must be (N, 1)."""
        sk, _ = jax.random.split(KEY)
        _, noisy_resp = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.5))
        assert noisy_resp.shape == (100, 1)

    def test_binary(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Noisy response values must be in {0, 1}."""
        sk, _ = jax.random.split(KEY)
        _, noisy_resp = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.5))
        assert set(np.unique(np.array(noisy_resp))).issubset({0, 1})

    def test_zero_sigma_matches_clean(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Zero noise must produce the same result as the noiseless response."""
        sk, _ = jax.random.split(KEY)
        _, noisy_resp = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.0))
        np.testing.assert_array_equal(np.array(noisy_resp), np.array(get_response(w1x32, c100x32)))

    def test_high_noise_degrades(self, w1x32: jax.Array) -> None:
        """Very high noise must cause significant accuracy degradation."""
        sk1, sk2 = jax.random.split(KEY, 2)
        chall = generate_challenges(sk1, (2000, 32))
        clean = get_response(w1x32, chall).flatten()
        _, noisy_resp = noisy_get_response(sk2, w1x32, chall, jnp.float32(100.0))
        assert float(jnp.mean(jnp.equal(noisy_resp.flatten(), clean))) < 0.65

    def test_returns_tuple(self, w1x32: jax.Array, c100x32: jax.Array) -> None:
        """Must return a tuple."""
        sk, _ = jax.random.split(KEY)
        assert isinstance(noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.1)), tuple)


class TestDriftArbiter:
    """Pin Arbiter class behaviour."""

    def test_weight_shape(self) -> None:
        """Weight must have shape (1, 64)."""
        _, sk = jax.random.split(KEY)
        assert Arbiter(sk, (1, 64)).weight.shape == (1, 64)

    def test_call_equals_get_response(self, c100x32: jax.Array) -> None:
        """__call__ must equal get_response."""
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        np.testing.assert_array_equal(arb(c100x32), arb.get_response(c100x32))

    def test_same_rng_same_weights(self) -> None:
        """Same RNG must always produce the same weights."""
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            Arbiter(sk, (1, 32)).weight,
            Arbiter(sk, (1, 32)).weight,
        )

    def test_clone(self) -> None:
        """clone() must produce an independent object with equal weights."""
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        clone = arb.clone()
        assert arb is not clone
        np.testing.assert_array_equal(arb.weight, clone.weight)

    def test_pytree_roundtrip(self) -> None:
        """Pytree flatten/unflatten must preserve weights."""
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        children, aux = arb.tree_flatten()
        np.testing.assert_array_equal(
            arb.weight, Arbiter.tree_unflatten(aux, children).weight
        )


class TestDriftXor:
    """Pin Xor class behaviour."""

    def test_weight_shape(self) -> None:
        """Weight must have shape (3, 64)."""
        _, sk = jax.random.split(KEY)
        assert Xor(sk, (3, 64)).weight.shape == (3, 64)

    def test_response_is_tuple(self, c100x32: jax.Array) -> None:
        """get_response must return a 2-tuple."""
        _, sk = jax.random.split(KEY)
        result = Xor(sk, (3, 32)).get_response(c100x32)
        assert isinstance(result, tuple) and len(result) == 2

    def test_call_equals_get_response(self, c100x32: jax.Array) -> None:
        """__call__ must equal get_response for the XOR result."""
        _, sk = jax.random.split(KEY)
        xor_puf = Xor(sk, (3, 32))
        np.testing.assert_array_equal(xor_puf(c100x32)[1], xor_puf.get_response(c100x32)[1])

    def test_get_weight_shape(self) -> None:
        """get_weight(i) must return shape (1, n)."""
        _, sk = jax.random.split(KEY)
        xor_puf = Xor(sk, (3, 32))
        for i in range(3):
            assert xor_puf.get_weight(i).shape == (1, 32)

    def test_pytree_roundtrip(self) -> None:
        """Pytree flatten/unflatten must preserve weights."""
        _, sk = jax.random.split(KEY)
        xor_puf = Xor(sk, (3, 32))
        children, aux = xor_puf.tree_flatten()
        np.testing.assert_array_equal(xor_puf.weight, Xor.tree_unflatten(aux, children).weight)


class TestDriftNNewKeys:
    """Pin n_new_keys behaviour."""

    def test_shape(self) -> None:
        """Subkeys array must have shape (n, 2)."""
        _, subkeys = n_new_keys(KEY, 4)
        assert subkeys.shape == (4, 2)

    def test_deterministic(self) -> None:
        """Same PRNG key must always yield the same subkeys."""
        _, ka = n_new_keys(KEY, 4)
        _, kb = n_new_keys(KEY, 4)
        np.testing.assert_array_equal(ka, kb)

    def test_subkeys_distinct(self) -> None:
        """Each subkey must be unique."""
        _, subkeys = n_new_keys(KEY, 5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.array_equal(subkeys[i], subkeys[j])


class TestDriftIntegration:
    """End-to-end integration drift tests."""

    def test_crp_stable(self) -> None:
        """Repeated get_response calls must be identical (no hidden state)."""
        sk1, sk2 = jax.random.split(KEY, 2)
        arb = Arbiter(sk1, (1, 64))
        chall = generate_challenges(sk2, (100, 64))
        np.testing.assert_array_equal(arb.get_response(chall), arb.get_response(chall))

    def test_distinct_pufs_differ(self) -> None:
        """Two PUFs with different keys must produce substantially different responses."""
        sk1, sk2, sk_c = jax.random.split(KEY, 3)
        chall = generate_challenges(sk_c, (200, 64))
        r1 = Arbiter(sk1, (1, 64)).get_response(chall).flatten()
        r2 = Arbiter(sk2, (1, 64)).get_response(chall).flatten()
        assert float(jnp.mean(r1 != r2)) > 0.1

    def test_xor_nonlinear(self) -> None:
        """XOR output must differ substantially from any individual arbiter."""
        sk1, sk2 = jax.random.split(KEY, 2)
        xor_puf = Xor(sk1, (3, 32))
        chall = generate_challenges(sk2, (500, 32))
        _, xor_r = xor_puf.get_response(chall)
        for i in range(3):
            single = get_response(row_vec(xor_puf.weight[i]), chall).flatten()
            assert float(jnp.mean(xor_r.flatten() == single)) < 0.95
