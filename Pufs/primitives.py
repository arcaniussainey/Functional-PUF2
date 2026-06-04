"""
Pure functional JAX primitives shared by all PUF implementations.

No classes, no circular imports.  Every function here is either
@jax.jit or explicitly documented as to why it is not.

Conventions (unchanged from FunctionalPuf.py):
  * <rng> always expects a fresh JAX PRNG key
  * everything is a row vector: shape (k, n + 1) where k=arbiters and n=challenge stages
  * challenges are elements of {-1, +1}, shape (N, n)
  * responses are elements of {0, 1},   shape (N, k)
  * the parameter order is: rng, weights, challenges, *args
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
from jax import lax
import jax.numpy as jnp



# Type aliases (structural -- JAX arrays are all jax.Array at runtime)


PRNGKey   = jax.Array   # shape (2,)
Weight    = jax.Array   # shape (k, n + 1)  float32
Challenge = jax.Array   # shape (N, n)      int8 in {-1, +1}
Response  = jax.Array   # shape (N, k)  uint8 in {0, 1}
Delta     = jax.Array   # shape (N, k)  float32



# Utility shapes


def row_vec(x: jax.Array) -> jax.Array:
    """Reshape any array to a single row vector (1, -1)."""
    return x.reshape((1, -1))


def col_vec(x: jax.Array) -> jax.Array:
    """Reshape any array to a single column vector (-1, 1)."""
    return x.reshape((-1, 1))



# PRNG helpers


def n_new_keys(rng: PRNGKey, n: int) -> Tuple[PRNGKey, jax.Array]:
    """
    Generate n new keys from rng.
    The subkeys returned are vstacked (i.e. subkeys.shape == (n, 2)).

    Args:
        rng (PRNGKey): PRNG to split
        n (int): number of keys needed

    Returns:
        Tuple[PRNGKey, jax.Array]: (rng, subkeys)
    """
    rng, *subkeys = jax.random.split(rng, n + 1)
    subkeys = jnp.vstack(subkeys)
    return rng, subkeys



# Challenge generation


@partial(jax.jit, static_argnums=(1,))
def generate_challenges(rng: PRNGKey, dim: Tuple[int, int]) -> Challenge:
    """
    Generate challenges in {-1, +1}.

    Args:
        rng (PRNGKey): PRNG key
        dim (Tuple[int, int]): challenge dimension, e.g. (10, 128)

    Returns:
        Challenge: dim-shaped array of challenges in {-1, +1}
    """
    c = jax.random.randint(rng, dim, 0, 2, dtype=jnp.int8)
    c = (c * 2) - 1
    return c



# Arbiter feature transform


@jax.jit
def phi_from_challenges(challenge: Challenge) -> jax.Array:
    """
    Convert ``{-1, +1}`` challenges to Arbiter-PUF feature vectors.

    A k-bit Arbiter challenge maps to k suffix-product coordinates plus a
    trailing constant bias coordinate.  With repository challenge bits
    represented directly as ``{-1, +1}``, the variable coordinates are the
    right-to-left cumulative products of each row.

    Args:
        challenge (Challenge): challenge matrix, shape (N, n)

    Returns:
        jax.Array: feature matrix, shape (N, n + 1), dtype float32
    """
    challenge_f = challenge.astype(jnp.float32)
    variable = jnp.cumprod(challenge_f[:, ::-1], axis=1)[:, ::-1]
    bias = jnp.ones((challenge.shape[0], 1), dtype=jnp.float32)
    return jnp.hstack([variable, bias])


@jax.jit
def linear_get_response(weight: Weight, feature: jax.Array) -> Response:
    """
    Return binary responses for an explicit feature matrix.

    This is the raw linear threshold primitive.  Use ``get_response`` for a
    normal Arbiter PUF because it applies ``phi_from_challenges`` first.
    """
    delta = weight @ feature.T
    return (delta.T > 0).astype(jnp.uint8)


@jax.jit
def linear_get_delta_response(weight: Weight, feature: jax.Array) -> Delta:
    """Return raw linear delay deltas for an explicit feature matrix."""
    return (weight @ feature.T).T.astype(jnp.float32)


# Weight generation


@partial(jax.jit, static_argnums=(1,))
def generate_1weight(rng: PRNGKey, dim: int) -> Weight:
    """
    Generate a single paper-correct Arbiter-PUF weight vector.

    ``dim`` is the number of physical challenge stages.  The returned weight
    has one additional coordinate for the constant term in ``Phi(C)``.

    Args:
        rng (PRNGKey): PRNG Key
        dim (int): number of challenge stages

    Returns:
        Weight: single weight as row vector, shape (1, dim + 1)
    """
    delays = (jax.random.normal(rng, shape=(4, dim)) + 500) * 4
    wv0 = delays[0, :] - delays[1, :]
    wv1 = delays[2, :] - delays[3, :]
    shiftr = jnp.hstack([jnp.array(0.0, dtype=jnp.float32), (wv0 + wv1) / 2])
    sub = jnp.hstack([(wv0 - wv1) / 2, jnp.array(0.0, dtype=jnp.float32)])
    weight = shiftr + sub
    weight = weight.at[0].set((wv0[0] - wv1[0]) / 2)
    return row_vec(weight.astype(jnp.float32))


def generate_weights(rng: PRNGKey, dim: Tuple[int, int] = (1, 64)) -> Weight:
    """
    Generate one or more paper-correct arbiter weight vectors.
    Each row is a PUF weight vector.
    E.g. (3, 64) generates 3 arbiters for 64 challenge stages.

    Args:
        rng (PRNGKey): PRNG Key
        dim (Tuple[int, int]): (k, n) -- k arbiters, n challenge stages

    Returns:
        Weight: weight matrix of shape (k, n + 1)
    """
    subkeys = jax.random.split(rng, dim[0])
    weights = jnp.vstack([generate_1weight(sk, dim[1]) for sk in subkeys])
    return weights


@partial(jax.jit, static_argnums=(1, 2))
def generate_mem_weights(rng: PRNGKey, dim: Tuple[int, int], w: int = 4) -> jax.Array:
    """
    Generate memory PUF weights given dimension and bit-width.

    Memory PUF is a digital PUF that uses fixed integers as weights instead
    of floats.  It avoids the reliability issues of regular PUFs caused by
    circuit noise.  Two's complement is used to create the interval of PUF
    weights.

    Args:
        rng (PRNGKey): PRNG key
        dim (Tuple[int, int]): dimension of generated weights
        w (int): bit-width to use, determines the range of numbers to pick
                 from when generating weights; usually 3, 4, or 5

    Returns:
        jax.Array: weight matrix, dtype int8
    """
    v = 2 ** (w - 1)
    return jax.random.randint(rng, dim, -v, v, dtype=jnp.int8)



# Core response functions


@jax.jit
def get_response(weight: Weight, challenge: Challenge) -> Response:
    """
    Canonical paper-correct Arbiter PUF response.

    Output layout: responses[i, j] is arbiter j's response to challenge i.

    | r_c0_w0 | r_c0_w1 | ...
    | r_c1_w0 | r_c1_w1 | ...

    A challenge with ``n`` bits is first mapped to ``Phi(C)`` with ``n + 1``
    coordinates; the final coordinate is the constant bias term.

    Args:
        weight (Weight): weight matrix, shape (k, n + 1)
        challenge (Challenge): challenge matrix, shape (N, n)

    Returns:
        Response: shape (N, k), dtype uint8, values in {0, 1}
    """
    return linear_get_response(weight, phi_from_challenges(challenge))


@jax.jit
def get_delta_response(weight: Weight, challenge: Challenge) -> Delta:
    """
    Raw paper-correct Arbiter delay difference for each challenge.

    Equivalent to get_response without the thresholding step.

    Args:
        weight (Weight): weight matrix, shape (k, n + 1)
        challenge (Challenge): challenge matrix, shape (N, n)

    Returns:
        Delta: raw time delay values, shape (N, k), dtype float32
    """
    return linear_get_delta_response(weight, phi_from_challenges(challenge))


@jax.jit
def xor_get_response(weight: Weight, challenge: Challenge) -> Tuple[Response, Response]:
    """
    Canonical XOR PUF response.

    If you have 3 arbiters with 64 stages, weight.shape == (3, 64).
    Returns both the per-arbiter individual responses and the XOR result.

    Args:
        weight (Weight): weight matrix, shape (k, n)
        challenge (Challenge): challenge matrix, shape (N, n)

    Returns:
        Tuple[Response, Response]:
            individual: shape (N, k) -- per-arbiter binary responses;
                        individual[i, j] is arbiter j's response to
                        challenge i.  Access arbiter j across all
                        challenges as individual[:, j].
            xor_r:      shape (N,)   -- XOR of all k arbiters per challenge.
    """
    individual = get_response(weight, challenge)
    xor_r = lax.reduce(individual, jnp.uint8(0), lax.bitwise_xor, (1,))
    return individual, xor_r



# Noise


@jax.jit
def noisy_generate_weights(
    rng: PRNGKey,
    weight: Weight,
    sigma_error: jax.Array,
) -> Tuple[PRNGKey, Weight]:
    """
    Add Gaussian noise to a weight matrix.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): true weight matrix
        sigma_error (jax.Array): scalar or broadcastable noise std deviation

    Returns:
        Tuple[PRNGKey, Weight]: (new_rng, noisy_weight)
    """
    rng, subkey = jax.random.split(rng)
    noisy_w = weight + (jax.random.normal(subkey, weight.shape) * sigma_error)
    return rng, noisy_w


@jax.jit
def noisy_get_response(
    rng: PRNGKey,
    weight: Weight,
    challenge: Challenge,
    sigma_error: jax.Array,
) -> Tuple[PRNGKey, Response]:
    """
    Noisy version of get_response.

    Applies a fresh per-challenge noise draw to the weight vector.
    Each challenge is evaluated against an independently perturbed weight,
    modelling the physical reality that each measurement is a separate event.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): true weight matrix, shape (k, n)
        challenge (Challenge): challenge matrix, shape (N, n)
        sigma_error (jax.Array): noise std deviation (scalar)

    Returns:
        Tuple[PRNGKey, Response]: (new_rng, response of shape (N, 1))
    """
    rng, subkey = jax.random.split(rng)
    noisy_weight = jnp.repeat(weight, challenge.shape[0], axis=0)
    rng, noisy_weight = noisy_generate_weights(subkey, noisy_weight, sigma_error)
    phi = phi_from_challenges(challenge)
    delta = jnp.sum(noisy_weight * phi, axis=1)
    return rng, col_vec((delta > 0).astype(jnp.uint8))


@jax.jit
def noisy_xor_get_response(
    rng: PRNGKey,
    weight: Weight,
    challenge: Challenge,
    sigma_error: jax.Array,
) -> Tuple[PRNGKey, Response]:
    """
    Noisy version of xor_get_response.

    Each of the k arbiters gets independently noisy weights.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): weight matrix, shape (k, n)
        challenge (Challenge): challenge matrix, shape (N, n)
        sigma_error (jax.Array): shape (k,) -- one std deviation per arbiter

    Returns:
        Tuple[PRNGKey, Response]: (new_rng of shape (2,), xor_response of shape (N, 1))
    """
    rng, subkeys = n_new_keys(rng, weight.shape[0])
    _per_arbiter_rng, noisy_response = jax.vmap(
        lambda rng_k, w, sigma: noisy_get_response(
            rng_k, row_vec(w), challenge, row_vec(sigma)
        )
    )(subkeys, weight, sigma_error)
    # vmap shape is (k, N, 1); squeeze the trailing 1 and transpose to (N, k)
    noisy_response = jnp.squeeze(noisy_response, axis=2).T
    noisy_response = lax.reduce(noisy_response, jnp.uint8(0), lax.bitwise_xor, (1,))
    return rng, col_vec(noisy_response)


@jax.jit
def noisy_get_delta_response(
    rng: PRNGKey,
    weight: Weight,
    challenge: Challenge,
    sigma_error: jax.Array,
) -> Tuple[PRNGKey, Delta]:
    """
    Noisy version of get_delta_response.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): weight matrix, shape (k, n)
        challenge (Challenge): challenge matrix, shape (N, n)
        sigma_error (jax.Array): noise std deviation (scalar)

    Returns:
        Tuple[PRNGKey, Delta]: (new_rng, delta of shape (N, 1))
    """
    rng, subkey = jax.random.split(rng)
    noisy_weight = jnp.repeat(weight, challenge.shape[0], axis=0)
    rng, noisy_weight = noisy_generate_weights(subkey, noisy_weight, sigma_error)
    phi = phi_from_challenges(challenge)
    noisy_delta = jnp.sum(noisy_weight * phi, axis=1)
    return rng, col_vec(noisy_delta.astype(jnp.float32))


@jax.jit
def noisy_xor_get_delta_response(
    rng: PRNGKey,
    weight: Weight,
    challenge: Challenge,
    sigma_error: jax.Array,
) -> Tuple[PRNGKey, Delta]:
    """
    Noisy version of xor_get_delta_response.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): weight matrix, shape (k, n)
        challenge (Challenge): challenge matrix, shape (N, n)
        sigma_error (jax.Array): shape (k,) -- one std deviation per arbiter

    Returns:
        Tuple[PRNGKey, Delta]: (new_rng, delta of shape (N, k))
    """
    rng, subkeys = n_new_keys(rng, weight.shape[0])
    noisy_delta = jax.vmap(
        lambda rng_k, w, sigma: noisy_get_delta_response(
            rng_k, row_vec(w), challenge, sigma
        )[1]
    )(subkeys, weight, sigma_error)
    # vmap shape is (k, N, 1); squeeze trailing 1 and transpose to (N, k)
    noisy_delta = jnp.squeeze(noisy_delta, axis=2).T
    return rng, noisy_delta



# Sigma / reliability search


def target_error(
    rng: PRNGKey,
    weight: Weight,
    chall: Challenge,
    target: float = 0.95,
    start: int = 0,
    stop: int = 10,
    nsamples: int = 2_500,
) -> jax.Array:
    """
    Find the noise std deviation that produces an accuracy closest to *target*.

    Given a PRNG key, true weights, and a challenge matrix, *nsamples*
    linearly spaced candidates in [start, stop] are evaluated as the
    std deviation for a N(0, sigma^2) perturbation.  The candidate
    whose accuracy (fraction of responses matching the noiseless output)
    is nearest to *target* is returned.

    Args:
        rng (PRNGKey): PRNG Key
        weight (Weight): weight matrix
        chall (Challenge): challenge matrix
        target (float, optional): desired accuracy rate. Defaults to 0.95.
        start (int, optional): lower bound of search. Defaults to 0.
        stop (int, optional): upper bound of search. Defaults to 10.
        nsamples (int, optional): search resolution. Defaults to 2_500.

    Returns:
        jax.Array: the std deviation yielding accuracy closest to *target*,
                   shape (k, 1) -- one sigma per arbiter row in the weight matrix.
    """
    true_r = get_response(weight, chall).flatten()
    sigma_error = jnp.linspace(start, stop, nsamples, axis=0)
    rng, subkeys = n_new_keys(rng, sigma_error.shape[0])
    calc_err = jax.vmap(
        lambda rng_k, sigma: jnp.equal(
            get_response(noisy_generate_weights(rng_k, weight, sigma)[1], chall).flatten(),
            true_r,
        ).mean()
    )
    err = calc_err(subkeys, sigma_error)
    sigma_error = sigma_error[jnp.abs(err - target).flatten().argmin()]
    return col_vec(sigma_error)


def get_sigma_error(
    rng: PRNGKey,
    weight: Weight,
    target: float = 0.95,
    start: int = 0,
    stop: int = 10,
    nsamples: int = 2_500,
    nchall: int = 5_000,
) -> jax.Array:
    """
    Like target_error but generates its own challenge set internally.
    Vmaps over each row of the weight matrix independently.

    Args:
        rng (PRNGKey): PRNG key
        weight (Weight): weight matrix, shape (k, n)
        target (float): desired accuracy. Defaults to 0.95.
        start (int): sigma search lower bound. Defaults to 0.
        stop (int): sigma search upper bound. Defaults to 10.
        nsamples (int): search resolution. Defaults to 2_500.
        nchall (int): number of internally generated challenges. Defaults to 5_000.

    Returns:
        jax.Array: shape (k, 1) -- one sigma per arbiter
    """
    rng, *subkeys = jax.random.split(rng, 3)
    challenge = generate_challenges(subkeys[0], (nchall, weight.shape[1]))
    calc_sigma_error = jax.vmap(
        lambda w: target_error(
            subkeys[1], row_vec(w), challenge,
            target=target, start=start, stop=stop, nsamples=nsamples,
        )
    )
    sigma_error = calc_sigma_error(weight)
    return col_vec(sigma_error)


def similar_weight(
    rng: PRNGKey,
    weight: Weight,
    target: float = 0.95,
    start: int = 0,
    stop: int = 10,
    nsamples: int = 2_500,
    nchall: int = 50_000,
) -> Tuple[PRNGKey, Weight, jax.Array]:
    """
    Yield weights and the std deviation that will on average produce a
    *target* deviance from the true response.

    Args:
        rng (PRNGKey): PRNG Key
        weight (Weight): weight matrix
        target (float, optional): desired error rate. Defaults to 0.95.
        start (int, optional): lower bound of search. Defaults to 0.
        stop (int, optional): upper bound of search. Defaults to 10.
        nsamples (int, optional): search resolution. Defaults to 2_500.
        nchall (int, optional): challenges used for error calculation. Defaults to 50_000.

    Returns:
        Tuple[PRNGKey, Weight, jax.Array]: (rng, noisy_weights, sigma_error)
    """
    rng, subkey = jax.random.split(rng)
    sigma_error = get_sigma_error(
        subkey, weight,
        target=target, start=start, stop=stop, nsamples=nsamples, nchall=nchall,
    )
    rng, subkeys = n_new_keys(rng, sigma_error.shape[0])
    _, noisy_weights = jax.vmap(noisy_generate_weights)(subkeys, weight, sigma_error)
    return rng, noisy_weights, sigma_error
