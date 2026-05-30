"""
Pufs/FunctionalPuf.py
---------------------
Public re-export surface and concrete PUF class definitions.

Contains the core PUF I/O primitives (re-exported from Pufs.primitives)
and the Arbiter / Xor wrapper classes.

Assumptions
-----------
* If you see ``rng`` as a parameter it expects a fresh JAX PRNG key.
* Everything is a row vector: a single weight with 64 stages has shape (1, 64).
* For challenges, a single row is one challenge set.

PRNG key generation is done through two functions:

  1. ``rng = new_key(seed)``   -- stateful module-level generator
  2. ``rng, subkeys = n_new_keys(rng, n)``  -- pure, JIT-friendly

``new_key()`` is an infinite generator and contains state; it cannot be
used inside jitted functions.  ``n_new_keys()`` can be.  Given the same
PRNG key, repeated function calls produce the same output across devices.

Classes
-------
``Arbiter(rng, dim)`` and ``Xor(rng, dim)`` are thin wrappers around
their functional counterparts.  Both register with JAX as internal pytree
nodes, allowing them to be used as parameters to jitted / vmapped functions
(with care).

Example::

    @partial(jax.jit, static_argnums=(0,))
    def puf_in_jitted_fn(puf, chall):
        return puf(chall)

Parameter order convention::

    1. PRNG key
    2. Weights
    3. Challenges
    4. *args / **kwargs

Noisy versions of I/O functions shadow their non-noisy counterparts by
prepending ``noisy_`` to the function name and require an additional
``sigma_error`` argument (the standard deviation of a N(0, sigma_error^2)
distribution from which the noise is sampled).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

# All JAX primitives and type aliases live in primitives.py.
# FunAttack.py imports from here, so keeping this the public surface
# avoids changing any import in the attack code.
from Pufs.primitives import (
    PRNGKey, Weight, Challenge, Response, Delta,
    row_vec,
    col_vec,
    n_new_keys,
    generate_challenges,
    generate_1weight,
    generate_weights,
    generate_mem_weights,
    get_response,
    get_delta_response,
    xor_get_response,
    noisy_generate_weights,
    noisy_get_response,
    noisy_xor_get_response,
    noisy_get_delta_response,
    noisy_xor_get_delta_response,
    target_error,
    get_sigma_error,
    similar_weight,
)

from Pufs.base import BasePUF

from Pufs.randomness import DeterministicKeyStream, DistributionSpec, sample_distribution
from Pufs.attack_state import AttackContext, AttackRunConfig, AttackTiming

from Pufs.operations import (
    OperationSpec,
    OperationState,
    OperationPipeline,
    op_pipeline,
    lower_operations,
    register_operation,
    registered_operations,
    op_generate_weights,
    op_generate_challenges,
    op_add_gaussian_noise,
    op_age,
    op_evaluate_response,
    op_evaluate_xor_response,
)



# Public API -- everything listed here is intentionally re-exported so that
# callers can do ``from Pufs.FunctionalPuf import <name>`` without touching
# Pufs.primitives or Pufs.base directly.

__all__ = [
    # Type aliases
    "PRNGKey", "Weight", "Challenge", "Response", "Delta",
    # Array helpers
    "row_vec", "col_vec",
    # PRNG helpers
    "n_new_keys",
    # Challenge generation
    "generate_challenges",
    # Weight generation
    "generate_1weight", "generate_weights", "generate_mem_weights",
    # Response functions
    "get_response", "get_delta_response", "xor_get_response",
    # Noise functions
    "noisy_generate_weights",
    "noisy_get_response", "noisy_xor_get_response",
    "noisy_get_delta_response", "noisy_xor_get_delta_response",
    # Reliability / sigma search
    "target_error", "get_sigma_error", "similar_weight",
    # Reproducibility / attack orchestration helpers
    "DeterministicKeyStream", "DistributionSpec", "sample_distribution",
    "AttackContext", "AttackRunConfig", "AttackTiming",
    # Operation-spec DSL
    "OperationSpec", "OperationState", "OperationPipeline",
    "op_pipeline", "lower_operations", "register_operation", "registered_operations",
    "op_generate_weights", "op_generate_challenges", "op_add_gaussian_noise",
    "op_age", "op_evaluate_response", "op_evaluate_xor_response",
    # Base class
    "BasePUF",
    # Concrete classes
    "Arbiter", "Xor",
    # Module-level PRNG helpers
    "RANDOM", "key_gen", "new_key",
]


RANDOM: bool = True  # initialise new_key(seed) with a random seed?


# Stateful PRNG key generator


def key_gen(seed: int = 0):
    """
    Infinite PRNG key generator.

    Args:
        seed (int, optional): PRNG starting seed. Defaults to 0.

    Yields:
        jax.Array: PRNG key
    """
    _key_state = jax.random.PRNGKey(seed)
    _key_state, subkey = jax.random.split(_key_state)
    while True:
        yield subkey
        _key_state, subkey = jax.random.split(_key_state)


_key = (
    key_gen(np.random.randint(0, 1001)) if RANDOM else key_gen()
)  # module-level generator instance


def new_key() -> PRNGKey:
    """
    Return a new PRNG key by advancing the module-level generator.

    Note: this is stateful and cannot be used inside jitted functions.
    Use n_new_keys() for pure, reproducible key generation.

    Returns:
        jax.Array: PRNG Key
    """
    return next(_key)



# Concrete PUF classes


@register_pytree_node_class
class Arbiter(BasePUF):
    """
    Single-arbiter PUF.

    The rng key provided to the constructor is used to generate weights;
    this process is deterministic -- given the same rng you will get the
    same weights.

    Note: tree_unflatten reconstructs using the RNG. Manually changing
    weights is not recommended.

    The dim init param should be a tuple describing the shape of the
    weights matrix, e.g. (1, 64).
    """

    def __init__(self, rng: PRNGKey, dim: Tuple[int, int] = (1, 64)) -> None:
        """Initialise an Arbiter PUF with the given RNG key and dimension."""
        super().__init__(rng, dim)

    def _compute_response(self, weight: Weight, challenge: Challenge) -> Response:
        """Compute the binary arbiter response for a batch of challenges."""
        return get_response(weight, challenge)

    def noisy_get_response(
        self,
        rng: PRNGKey,
        challenge: Challenge,
        sigma_error: jax.Array,
    ) -> Tuple[PRNGKey, Response]:
        """
        Evaluate with Gaussian noise on the weight vector.

        Prefer ``puf.run_pipeline(rng, pipeline)`` with an
        ``add_gaussian_noise`` step for new code.  This method is
        retained for backwards compatibility.

        Args:
            rng (PRNGKey): PRNG key
            challenge (Challenge): challenge matrix, shape (N, n)
            sigma_error (jax.Array): noise std deviation

        Returns:
            Tuple[PRNGKey, Response]: (new_rng, response)
        """
        rng, subkey = jax.random.split(rng)
        return noisy_get_response(subkey, self.weight, challenge, sigma_error)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"Arbiter(weight={self.weight})"


@register_pytree_node_class
class Xor(BasePUF):
    """
    XOR PUF -- k independent arbiters whose outputs are XOR-combined.

    The rng key provided to the constructor is used to generate weights;
    this process is deterministic -- given the same rng you will get the
    same weights.

    Note: tree_unflatten reconstructs using the RNG. Manually changing
    weights is not recommended.

    The dim init param should be a tuple describing the shape of the
    weights matrix, e.g. (3, 64) for a 3-XOR PUF with 64 stages.
    """

    def __init__(self, rng: PRNGKey, dim: Tuple[int, int] = (3, 64)) -> None:
        """Initialise a XOR PUF with the given RNG key and dimension."""
        super().__init__(rng, dim)

    def _compute_response(
        self, weight: Weight, challenge: Challenge
    ) -> Tuple[Response, Response]:
        """
        Return (individual, xor_r) matching xor_get_response conventions.

        individual: shape (N, k) -- per-arbiter binary responses
        xor_r:      shape (N,)   -- XOR across all k arbiters
        """
        return xor_get_response(weight, challenge)

    def get_noisy_response(
        self,
        rng: PRNGKey,
        challenge: Challenge,
        sigma_error: jax.Array,
    ) -> Tuple[PRNGKey, Response]:
        """
        Evaluate with per-arbiter Gaussian noise.

        Prefer ``puf.run_pipeline(rng, pipeline)`` with an
        ``add_gaussian_noise`` step for new code.  This method is
        retained for backwards compatibility.

        Args:
            rng (PRNGKey): PRNG key
            challenge (Challenge): challenge matrix, shape (N, n)
            sigma_error (jax.Array): shape (k,) -- one std deviation per arbiter

        Returns:
            Tuple[PRNGKey, Response]: (new_rng, xor_response of shape (N, 1))
        """
        rng, subkey = jax.random.split(rng)
        return noisy_xor_get_response(subkey, self.weight, challenge, sigma_error)

    def get_weight(self, i: int) -> Weight:
        """
        Return arbiter i's weight vector (row i of the weight matrix).

        Args:
            i (int): arbiter row index

        Returns:
            Weight: arbiter i's weight, shape (1, n)
        """
        return row_vec(self.weight[i, :])

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"Xor(weight={self.weight})"



# Module-level smoke-test


if __name__ == "__main__":
    TARGET_ERR = 0.05

    _rng = jax.random.PRNGKey(seed=0)
    _rng, _sk0, _sk1, _sk2 = jax.random.split(_rng, 4)

    _puf = Xor(_sk0, (3, 128))
    _chall = generate_challenges(_sk1, (1000, _puf.dim[1]))

    _rng, _noisy_weights, _sigma_error = similar_weight(
        _sk2, _puf.weight, target=TARGET_ERR
    )

    print(f"Target={TARGET_ERR}")

    for _i in range(_puf.dim[0]):
        _noisy_response = get_response(row_vec(_noisy_weights[_i]), _chall).flatten()
        _true_response  = get_response(row_vec(_puf.weight[_i]), _chall).flatten()
        _accuracy       = jnp.equal(_noisy_response, _true_response).mean()
        print(f"weight {_i} % of responses equal: {_accuracy}")
