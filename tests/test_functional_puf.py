"""
tests/test_functional_puf.py
Run from the repo root:  pytest tests/ -v

BUG tests   — fail on the original code, pass once the bug is fixed.
DRIFT tests — pin correct behaviour; catch regressions during refactoring.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import (
    Arbiter, Xor,
    generate_weights, generate_challenges, generate_mem_weights,
    get_response, get_delta_response, xor_get_response,
    noisy_get_response, noisy_generate_weights,
    n_new_keys, row_vec,
)

KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def w1x32():
    _, sk = jax.random.split(KEY)
    return generate_weights(sk, (1, 32))


@pytest.fixture(scope="module")
def w3x32():
    _, sk = jax.random.split(KEY)
    return generate_weights(sk, (3, 32))


@pytest.fixture(scope="module")
def c100x32():
    _, _, sk = jax.random.split(KEY, 3)
    return generate_challenges(sk, (100, 32))


# Tests

class XORResponseTest:
    """XOR correctness is tested here. It should be noted that the shape of the return on the xor response shape is weird, but the code seems to expect this? """

    # Could add shape test here

    def test_xor_matches_manual(self, w3x32, c100x32):
        """XOR result must equal bitwise XOR of individual get_response calls."""
        _, xored = xor_get_response(w3x32, c100x32)
        r = [get_response(row_vec(w3x32[i]), c100x32).flatten() for i in range(3)]
        expected = np.bitwise_xor(np.bitwise_xor(np.array(r[0]), np.array(r[1])), np.array(r[2]))
        np.testing.assert_array_equal(np.array(xored).flatten(), expected)


class MemWeightsRange:
    """randint upper bound is v-1 but JAX randint is exclusive,
    so the maximum value v-1 is never produced. This means it's actually v-2"""

    @pytest.mark.parametrize("w_bits,expected_max", [(3, 3), (4, 7), (5, 15)])
    def test_max_reachable(self, w_bits, expected_max):
        """Identify range of randint"""
        _, sk = jax.random.split(KEY)
        weights = generate_mem_weights(sk, (1, 100_000), w_bits)
        assert int(weights.max()) == expected_max, \
            f"w={w_bits}: max={int(weights.max())}, expected {expected_max}"

    @pytest.mark.parametrize("w_bits", [3, 4, 5])
    def test_min(self, w_bits):
        _, sk = jax.random.split(KEY)
        assert int(generate_mem_weights(sk, (1, 100_000), w_bits).min()) == -(2 ** (w_bits - 1))


class GlobalPrng:
    """new_key() is a stateful module-level generator"""

    def test_extra_call_changes_sequence(self):
        """Demonstrates the fragility. Must use n_new_keys() everywhere reproducibility is wanted."""
        import importlib
        from pathlib import Path

        def load_fresh():
            name = f"_FunctionalPuf_fresh_{id(object())}"
            spec = importlib.util.spec_from_file_location(
                name, Path(__file__).parent.parent / "Pufs" / "FunctionalPuf.py"
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m

        ma, mb = load_fresh(), load_fresh()
        k1a, k2a = ma.new_key(), ma.new_key()
        mb.new_key()
        k1b, k2b = mb.new_key(), mb.new_key()

        #This test passing is likely BAD
        assert not (jnp.array_equal(k1a, k1b) and jnp.array_equal(k2a, k2b))

    def test_n_new_keys_is_deterministic(self):
        rng = jax.random.PRNGKey(99)
        _, ka = n_new_keys(rng, 4)
        _, kb = n_new_keys(rng, 4)
        np.testing.assert_array_equal(ka, kb)


# Identifying feature drift
class TestDrift_Challenges:

    def test_shape(self):
        _, sk = jax.random.split(KEY)
        assert generate_challenges(sk, (50, 64)).shape == (50, 64)

    def test_values_pm1(self):
        _, sk = jax.random.split(KEY)
        assert set(np.unique(np.array(generate_challenges(sk, (500, 32))))).issubset({-1, 1})

    def test_dtype(self):
        _, sk = jax.random.split(KEY)
        assert generate_challenges(sk, (10, 8)).dtype == jnp.int8

    def test_balanced(self):
        _, sk = jax.random.split(KEY)
        frac = float(jnp.mean(generate_challenges(sk, (5000, 64)) == 1))
        assert 0.45 < frac < 0.55

    def test_deterministic(self):
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            generate_challenges(sk, (20, 16)),
            generate_challenges(sk, (20, 16)),
        )


class TestDrift_Weights:

    def test_shape(self):
        _, sk = jax.random.split(KEY)
        assert generate_weights(sk, (3, 64)).shape == (3, 64)

    def test_dtype(self):
        _, sk = jax.random.split(KEY)
        assert generate_weights(sk, (2, 32)).dtype == jnp.float32

    def test_rows_independent(self):
        _, sk = jax.random.split(KEY)
        W = generate_weights(sk, (4, 32))
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(W[i], W[j])

    def test_deterministic(self):
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            generate_weights(sk, (3, 32)),
            generate_weights(sk, (3, 32)),
        )


class TestDrift_GetResponse:

    def test_shape_single(self, w1x32, c100x32):
        assert get_response(w1x32, c100x32).shape == (100, 1)

    def test_shape_multi(self, w3x32, c100x32):
        assert get_response(w3x32, c100x32).shape == (100, 3)

    def test_binary(self, w3x32, c100x32):
        assert set(np.unique(np.array(get_response(w3x32, c100x32)))).issubset({0, 1})

    def test_dtype(self, w1x32, c100x32):
        assert get_response(w1x32, c100x32).dtype == jnp.uint8

    def test_deterministic(self, w3x32, c100x32):
        np.testing.assert_array_equal(
            get_response(w3x32, c100x32),
            get_response(w3x32, c100x32),
        )

    def test_formula(self):
        sk1, sk2 = jax.random.split(KEY, 2)
        W = generate_weights(sk1, (2, 8))
        C = generate_challenges(sk2, (5, 8)).astype(jnp.float32)
        R = get_response(W, C)
        for i in range(5):
            for j in range(2):
                assert int(R[i, j]) == int(np.dot(np.array(W[j]), np.array(C[i])) > 0)

    def test_balanced(self, w1x32):
        _, sk = jax.random.split(KEY)
        C = generate_challenges(sk, (2000, 32))
        assert 0.42 < float(jnp.mean(get_response(w1x32, C))) < 0.58


class TestDrift_DeltaResponse:

    def test_shape(self, w3x32, c100x32):
        assert get_delta_response(w3x32, c100x32).shape == (100, 3)

    def test_dtype_float(self, w1x32, c100x32):
        assert jnp.issubdtype(get_delta_response(w1x32, c100x32).dtype, jnp.floating)

    def test_sign_consistent(self, w3x32, c100x32):
        D = get_delta_response(w3x32, c100x32)
        R = get_response(w3x32, c100x32)
        np.testing.assert_array_equal(np.array(D > 0.5), np.array(R))


class TestDrift_XorGetResponse:

    def test_returns_two_tuple(self, w3x32, c100x32):
        result = xor_get_response(w3x32, c100x32)
        assert isinstance(result, tuple) and len(result) == 2

    def test_xor_binary(self, w3x32, c100x32):
        _, xored = xor_get_response(w3x32, c100x32)
        assert set(np.unique(np.array(xored))).issubset({0, 1})

    def test_xor_size(self, w3x32, c100x32):
        _, xored = xor_get_response(w3x32, c100x32)
        assert xored.size == 100

    def test_xor_deterministic(self, w3x32, c100x32):
        _, x1 = xor_get_response(w3x32, c100x32)
        _, x2 = xor_get_response(w3x32, c100x32)
        np.testing.assert_array_equal(x1, x2)


class TestDrift_NoisyResponse:

    def test_shape(self, w1x32, c100x32):
        sk, _ = jax.random.split(KEY)
        _, NR = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.5))
        assert NR.shape == (100, 1)

    def test_binary(self, w1x32, c100x32):
        sk, _ = jax.random.split(KEY)
        _, NR = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.5))
        assert set(np.unique(np.array(NR))).issubset({0, 1})

    def test_zero_sigma_matches_clean(self, w1x32, c100x32):
        sk, _ = jax.random.split(KEY)
        _, NR = noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.0))
        np.testing.assert_array_equal(np.array(NR), np.array(get_response(w1x32, c100x32)))

    def test_high_noise_degrades(self, w1x32):
        sk1, sk2 = jax.random.split(KEY, 2)
        C = generate_challenges(sk1, (2000, 32))
        clean = get_response(w1x32, C).flatten()
        _, NR = noisy_get_response(sk2, w1x32, C, jnp.float32(100.0))
        assert float(jnp.mean(jnp.equal(NR.flatten(), clean))) < 0.65

    def test_returns_tuple(self, w1x32, c100x32):
        sk, _ = jax.random.split(KEY)
        assert isinstance(noisy_get_response(sk, w1x32, c100x32, jnp.float32(0.1)), tuple)


class TestDrift_Arbiter:

    def test_weight_shape(self):
        _, sk = jax.random.split(KEY)
        assert Arbiter(sk, (1, 64)).weight.shape == (1, 64)

    def test_call_equals_get_response(self, c100x32):
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        np.testing.assert_array_equal(arb(c100x32), arb.get_response(c100x32))

    def test_same_rng_same_weights(self):
        _, sk = jax.random.split(KEY)
        np.testing.assert_array_equal(
            Arbiter(sk, (1, 32)).weight,
            Arbiter(sk, (1, 32)).weight,
        )

    def test_clone(self):
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        clone = arb.clone()
        assert arb is not clone
        np.testing.assert_array_equal(arb.weight, clone.weight)

    def test_pytree_roundtrip(self):
        _, sk = jax.random.split(KEY)
        arb = Arbiter(sk, (1, 32))
        children, aux = arb.tree_flatten()
        np.testing.assert_array_equal(
            arb.weight, Arbiter.tree_unflatten(aux, children).weight
        )


class TestDrift_Xor:

    def test_weight_shape(self):
        _, sk = jax.random.split(KEY)
        assert Xor(sk, (3, 64)).weight.shape == (3, 64)

    def test_response_is_tuple(self, c100x32):
        _, sk = jax.random.split(KEY)
        result = Xor(sk, (3, 32)).get_response(c100x32)
        assert isinstance(result, tuple) and len(result) == 2

    def test_call_equals_get_response(self, c100x32):
        _, sk = jax.random.split(KEY)
        x = Xor(sk, (3, 32))
        np.testing.assert_array_equal(x(c100x32)[1], x.get_response(c100x32)[1])

    def test_get_weight_shape(self):
        _, sk = jax.random.split(KEY)
        x = Xor(sk, (3, 32))
        for i in range(3):
            assert x.get_weight(i).shape == (1, 32)

    def test_pytree_roundtrip(self):
        _, sk = jax.random.split(KEY)
        x = Xor(sk, (3, 32))
        children, aux = x.tree_flatten()
        np.testing.assert_array_equal(x.weight, Xor.tree_unflatten(aux, children).weight)


class TestDrift_NNewKeys:

    def test_shape(self):
        _, subkeys = n_new_keys(KEY, 4)
        assert subkeys.shape == (4, 2)

    def test_deterministic(self):
        _, ka = n_new_keys(KEY, 4)
        _, kb = n_new_keys(KEY, 4)
        np.testing.assert_array_equal(ka, kb)

    def test_subkeys_distinct(self):
        _, subkeys = n_new_keys(KEY, 5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.array_equal(subkeys[i], subkeys[j])


class TestDrift_Integration:

    def test_crp_stable(self):
        sk1, sk2 = jax.random.split(KEY, 2)
        arb = Arbiter(sk1, (1, 64))
        C = generate_challenges(sk2, (100, 64))
        np.testing.assert_array_equal(arb.get_response(C), arb.get_response(C))

    def test_distinct_pufs_differ(self):
        sk1, sk2, sk_c = jax.random.split(KEY, 3)
        C = generate_challenges(sk_c, (200, 64))
        R1 = Arbiter(sk1, (1, 64)).get_response(C).flatten()
        R2 = Arbiter(sk2, (1, 64)).get_response(C).flatten()
        assert float(jnp.mean(R1 != R2)) > 0.1

    def test_xor_nonlinear(self):
        sk1, sk2 = jax.random.split(KEY, 2)
        x = Xor(sk1, (3, 32))
        C = generate_challenges(sk2, (500, 32))
        _, xor_r = x.get_response(C)
        for i in range(3):
            single = get_response(row_vec(x.weight[i]), C).flatten()
            assert float(jnp.mean(xor_r.flatten() == single)) < 0.95