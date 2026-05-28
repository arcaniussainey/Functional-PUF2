import jax.numpy as jnp
import jax
from functools import partial
import jax.lax as lax
import numpy as np
from functools import partial, wraps
from typing import Tuple
from jax.tree_util import register_pytree_node_class
from Attack.FunAttack import *
from Pufs.FunctionalPuf import *


"""
------------ Functional Puf ------------

    Contains Core Puf Primitive I/O functions 

    Assume:
    * if you see <rng> as a param it expects  a fresh PRNG key
    * everything is a row vector

    For challenges that means a single row is one set of challenges
    and a single weight with 64 stages would have shape (1, 64)
    
    PRNG key generations is done through two functions:
        1. rng = new_key(seed)
        2. rng, subkeys = n_new_keys(rng, n)

    <new_key()> is an infinite generator and contains state.
    as a result it cannot be used in jitted functions. 
    <n_new_keys()> however can be. 
    Given the same PRNG key repeated function calls
    should produce the same output across devices. 

    Classes:
     
    The `Arbiter(rng, dim)` and `Xor(rng, dim)` classes
    are thin wrappers around calls to their functional
    alternatives. They both register with jax as internal
    pytree nodes allowing them be used as parameters
    to jitted/vmapped functions, with caution. 

    EG 

    partial(jax.jit, static_argnums=(0,))
    def puf_in_jitted_fn(puf, chall):
        return puf(chall)
    
    In general the order of parameters is
    1. PRNG key
    2. Weights
    3. Challenges
    4. *args / *kwargs

    noisy versions of I/O functions lexically shadow their non-noisy
    versions by prepending "noisy_" to the function name.
    and require an additional arg "sigma_error" or the 
    standard deviation of a N(0, sigma_error^2) distribution
    from which the noise is sampled. Noisy functions typically
    generate noisy weights and then call their non-noisy counterparts
    returning the split PRNG and the result of the non-noisy 
    function call. 

    An Example using noise can be found under the main guard.
 
"""


RANDOM = True  # initialize new_key(seed) with random seed?


def key_gen(seed=0):
    """
    Infinite PRNG key generator

    Args:
        seed int, optional: PRNG starting seed. Defaults to 0.

    Yields:
        jnp.array: PRNG key
    """
    global seedval
    # print(seedval)
    # print(seed)
    seedval = seed
    _KEY = jax.random.PRNGKey(seed)
    _KEY, subkey = jax.random.split(_KEY)
    while True:
        yield subkey
        _KEY, subkey = jax.random.split(_KEY)


_key = (
    key_gen(np.random.randint(0, 1001)) if RANDOM else key_gen()
)  # instance of the generator


def new_key():
    """
    get a new PRNG key by calling next() on an a
    generator produced by <key_gen()?

    Returns:
        jnp.array: PRNG Key
    """
    # print('seedval is', seedval)
    return next(_key)


def n_new_keys(rng, n: int) -> Tuple[jax.Array, jax.Array]:
    """
    generate n new keys from rng
    the subkeys returned are vstacked (IE subkeys.shape==(n, 2))
    Args:
        rng (jnp.array): PRNG to split
        n (int): number of keys needed

    Returns:
        Tuple[jnp.array, jnp.array]: (rng, subkeys)
    """
    rng, *subkeys = jax.random.split(rng, n + 1)
    subkeys = jnp.vstack(subkeys)
    return rng, subkeys


def row_vec(x):
    return x.reshape((1, -1))


def col_vec(x):
    return x.reshape((-1, 1))


# canonical IO and clones
@register_pytree_node_class
class Arbiter(object):
    """
    Arbiter Primitive

    The rng key provided to the constuctor is used to generate
    weights this process is deterministic meaning given the
    same rng you will get the same weights.

    Note: tree unflatten reconstructs using the RNG
    if you manually change weights do not use it inside jitted
    functions

    Note: tree unflatten reconstructs using the RNG. Manually
    changing weights is not reccomended.

    The dim init param should be a tuple describing the shape
    of the weights matrix EG (1, 64)
    """

    def __init__(self, rng, dim=(1, 64)):
        self.rng = rng
        self.dim = dim
        self.weight = generate_weights(rng, dim)

    def __repr__(self):
        return "Arbiter(weight={})".format(self.weight)

    def tree_flatten(self):
        children = self.weight
        aux_data = (self.rng, self.dim)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    def get_response(self, challenge):
        return get_response(self.weight, challenge)

    def noisy_get_response(self, rng, challenge, sigma_error):
        rng, subkey = jax.random.split(rng)
        return noisy_get_response(subkey, self.weight, challenge, sigma_error)

    def get_delta_response(self, challenge):
        return get_delta_response(self.weight, challenge)

    def __call__(self, challenge):
        return get_response(self.weight, challenge)

    def clone(self):
        return Arbiter(self.rng, self.dim)


@register_pytree_node_class
class Xor(object):
    """
    Xor Primitive

    The rng key provided to the constuctor is used to generate
    weights this process is deterministic meaning given the
    same rng you will get the same weights.

    Note: tree unflatten reconstructs using the RNG. Manually
    changing weights is not reccomended.

    The dim init param should be a tuple describing the shape
    of the weights matrix EG (3, 64)
    """

    def __init__(self, rng, dim=(3, 64)):
        self.rng = rng
        self.dim = dim
        self.weight = generate_weights(rng, dim)

    def __repr__(self):
        return "Xor(weight={})".format(self.weight)

    def tree_flatten(self):
        children = self.weight
        aux_data = (self.rng, self.dim)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    def get_response(self, challenge):
        return xor_get_response(self.weight, challenge)

    def get_noisy_response(self, rng, challenge, sigma_error):
        rng, subkey = jax.random.split(rng)
        return noisy_xor_get_response(subkey, self.weight, challenge, sigma_error)

    def get_delta_response(self, challenge):
        return get_delta_response(self.weight, challenge)

    def __call__(self, challenge):
        return xor_get_response(self.weight, challenge)

    def clone(self):
        return Xor(self.rng, self.dim)

    def get_weight(self, i):
        """
        get base puf i's weight (IE row i of weight matrix)

        Args:
            i jnp.array: base puf row index

        Returns:
            jnp.array: base puf i's weight
        """
        return row_vec(self.weight[i, :])


@partial(jax.jit, static_argnums=(1,))
def generate_challenges(rng, dim):
    """
    generate challenges
    challenges are elements of {-1,1}

    Args:
        rng jnp.array: PRNG key
        dim tuple(int, int) challenge dimension EG (10, 128)

    Returns:
        jnp.array: dim shaped array of challenges
    """
    c = jax.random.randint(rng, dim, 0, 2, dtype=jnp.int8)
    c = (c * 2) - 1
    return c


@partial(jax.jit, static_argnums=(1,))
def generate_1weight(rng, dim):
    """
    Generate a single weight

    Args:
        rng jnp.array: PRNG Key
        dim tuple(int, int): dimension of generated weights

    Returns:
        jnp.array: single weight as row vector
    """
    delays = (jax.random.normal(rng, shape=(4, dim-1)) + 500) * 4
    wv0 = delays[0, :] - delays[1, :]
    wv1 = delays[2, :] - delays[3, :]
    shiftr = jnp.hstack([jnp.array(0), (wv0 + wv1) / 2])
    sub = jnp.hstack([(wv0 - wv1) / 2, jnp.array(0)])
    weight = shiftr + sub

    weight = weight.at[0].set((wv0[0] - wv1[0]) / 2)
    return row_vec(weight)


def generate_weights(rng, dim=(1, 64)):
    """
    generate 1 or more weights
    each row is a puf weight vector
    EG (3,64) would generate 3 weights with 64 stages

    Args:
        rng jnp.array: PRNG Key
        dim tuple(int, int): dimension of generated weights

    Returns:
        jnp.array: weight matrix
    """
    subkeys = jax.random.split(rng, dim[0])
    weights = jnp.vstack([generate_1weight(sk, dim[1]) for sk in subkeys])
    return weights


@partial(jax.jit, static_argnums=(1, 2))
def generate_mem_weights(rng, dim, w=4):
    """
    Generate memory puf weight given dimension and bit-width
    Memory puf is a digital puf that uses fixed numbers as weights instead of floats
    It avoids the reliability issues of regular pufs caused by circuit noise
    It uses two's complement to create the interval of puf weights

    Parameters:
        rng (jnp.array): PRNG key
        dim (tuple(int, int)): dimension of generated weights
        w (int): bit-width to use, determines the range of numbers to pick from when generating weights, usually 3, 4 or 5

    Returns:
        jnp.array: weight matrix
    """
    v = 2 ** (w - 1)
    return jax.random.randint(rng, dim, -v, v, dtype=jnp.int8)


@jax.jit
def get_response(weight, challenge):
    """
    canonical I/O

    output format

    | r1c1 | r2c1 | ...
    | r1c2 | r2c2 | ...

    IE responses for a particular challenge against multiple
    weights are rows.

    Args:
        weight jnp.array: weight
        challenge jnp.array: challenge
    Returns:
        response (jnp.array):
    """
    r = (jnp.sign((weight @ challenge.T)) + 1) / 2
    return r.T.astype(jnp.uint8)


@jax.jit
def get_delta_response(weight, challenge):
    """
    Delta response is the raw time delay value
    IE get response without the sign.

    Args:
        weight jnp.array: weight matrix
        challenge jnp.array: challenge matrix

    Returns:
        jnp.array: raw time delay values
    """
    delta = ((weight @ challenge.T) + 1) / 2
    return delta.T


@jax.jit
def xor_get_response(weight, challenge):
    """
    Canonical I/O for Xor Puf
    if you have 3 arbiters with 64 stages as your base pufs the weight
    parameter would have shape (3, 64)

    Args:
        weight jnp.array: weight matrix
        challenge jnp.array: challenge matrix

    Returns:
        jnp.array: xor response
    """
    r = get_response(weight, challenge)
    t_r=r.T.astype(jnp.uint8)

    new_r = lax.reduce(r, jnp.uint8(0), lax.bitwise_xor, (1,)) #This line calculates the XOR response. It applies the lax.reduce function, which reduces the r matrix along axis 1. The lax.bitwise_xor operation is applied to each row, and the initial value is set to jnp.uint8(0). This operation computes the XOR of the bits within each row, effectively generating a single XOR response for each challenge.
    return row_vec(t_r),row_vec(new_r)



def target_error(rng, weight, chall, target=0.95, start=0, stop=10, nsamples=2_500):
    """
    given a PRNG key, true weights, and a challenge matrix
    <nsamples> linearly spaced samples the range [start, stop]
    are used as a proposed sigma_error. The sampled sigma_error
    is then used as the standard deviation for a N(0, sigma_error^2) sample.
    The value that results in an error rate closest to <target>
    is returned.

    Args:
        rng jnp.array: PRNG Key
        weight jnp.array: weight matrix
        chall jnp.array: challenge matrix
        target float, optional: desired error rate. Defaults to 0.95.
        start int, optional: lower bound of search. Defaults to 0.
        stop int, optional: upper bound of search. Defaults to 10.
        nsamples int, optional: number of samples to use in search. Defaults to 1_000.

    Returns:
        jnp.array: the std deviation values that results in error closest to target
    """
    true_r = get_response(weight, chall).flatten()
    sigma_error = jnp.linspace(start, stop, nsamples, axis=0)
    rng, subkeys = n_new_keys(rng, sigma_error.shape[0])
    calc_err = jax.vmap(
        lambda rng, sigma: jnp.equal(
            get_response(
                noisy_generate_weights(rng, weight, sigma)[1], chall
            ).flatten(),
            true_r,
        ).mean()
    )

    err = calc_err(subkeys, sigma_error)
    sigma_error = sigma_error[jnp.abs(err - target).flatten().argmin()]
    return col_vec(sigma_error)


def get_sigma_error(
    rng, weight, target=0.95, start=0, stop=10, nsamples=2_500, nchall=5_000
):
    rng, *subkeys = jax.random.split(rng, 3)
    challenge = generate_challenges(subkeys[0], (nchall, weight.shape[1]))
    calc_sigma_error = jax.vmap(
        lambda w: target_error(
            subkeys[1],
            w,
            challenge,
            target=target,
            start=start,
            stop=stop,
            nsamples=nsamples,
        )
    )
    sigma_error = calc_sigma_error(weight)
    return col_vec(sigma_error)


def similar_weight(
    rng, weight, target=0.95, start=0, stop=10, nsamples=2_500, nchall=50_000
):
    """
    yeilds the weights and the std deviation that will on average
    result in a <target> deviance from true response on average

    Args:
        rng (jnp.array): PRNG Key
        weight (jnp.array): weight matrix
        challenge (jnp.array): challenge matrix
        target (float, optional): desired error rate. Defaults to 0.95.
        start (int, optional): lower bound of search. Defaults to 0.
        stop (int, optional): upper bound of search. Defaults to 10.
        nsamples (int, optional): number of samples to use in search. Defaults to 2_500.
        nchall (int, optional): number of challenges to use in error calculation

    Returns:
        tuple(jnp.array, jnp.array, jnp.array): (rng, best weight, sigma_error)
    """
    rng, subkey = jax.random.split(rng)
    sigma_error = get_sigma_error(
        subkey,
        weight,
        target=target,
        start=start,
        stop=stop,
        nsamples=nsamples,
        nchall=nchall,
    )
    rng, subkeys = n_new_keys(rng, sigma_error.shape[0])
    _, noisy_weights = jax.vmap(noisy_generate_weights)(subkeys, weight, sigma_error)
    return rng, noisy_weights, sigma_error


@jax.jit
def noisy_generate_weights(rng, weight, sigma_error):
    rng, subkey = jax.random.split(rng)
    noisy_w = weight + (jax.random.normal(subkey, weight.shape) * sigma_error)
    return rng, noisy_w


@jax.jit
def noisy_get_response(rng, weight, challenge, sigma_error):
    rng, subkey = jax.random.split(rng)
    noisy_weight = jnp.repeat(weight, challenge.shape[0], axis=0)
    rng, noisy_weight = noisy_generate_weights(subkey, noisy_weight, sigma_error)
    noisy_response = jax.vmap(get_response)(noisy_weight, challenge)
    return rng, col_vec(noisy_response)


@jax.jit
def noisy_xor_get_response(rng, weight, challenge, sigma_error):
    rng, subkeys = n_new_keys(rng, weight.shape[0])
    rng, noisy_response = jax.vmap(
        lambda rng, w, sigma: noisy_get_response(
            rng, row_vec(w), challenge, row_vec(sigma)
        )
    )(subkeys, weight, sigma_error)
    noisy_response = jnp.squeeze(
        noisy_response, axis=2
    ).T  # vmap shape is (weight.shape[0], challenge.shape[0], 1)
    noisy_response = lax.reduce(noisy_response, jnp.uint8(0), lax.bitwise_xor, (1,))
    return rng, col_vec(noisy_response)


@jax.jit
def noisy_get_delta_response(rng, weight, challenge, sigma_error):
    rng, subkey = jax.random.split(rng)
    noisy_weight = jnp.repeat(weight, challenge.shape[0], axis=0)
    rng, noisy_weight = noisy_generate_weights(subkey, noisy_weight, sigma_error)
    noisy_delta = jax.vmap(get_delta_response)(noisy_weight, challenge)
    return rng, col_vec(noisy_delta)


@jax.jit
def noisy_xor_get_delta_response(rng, weight, challenge, sigma_error):
    rng, subkeys = n_new_keys(rng, weight.shape[0])
    noisy_delta = jax.vmap(
        lambda rng, w, sigma: noisy_get_delta_response(
            rng, row_vec(w), challenge, sigma
        )[1]
    )(subkeys, weight, sigma_error)
    noisy_delta = jnp.squeeze(
        noisy_delta, axis=2
    ).T  # vmap shape is (weight.shape[0], challenge.shape[0], 1)
    return rng, noisy_delta


if __name__ == "__main__":
    target_err = 0.05

    rng = jax.random.PRNGKey(seed=0)
    rng, subkey0, subkey1, subkey2 = jax.random.split(rng, 4)

    puf = Xor(
        subkey0, (3, 128)
    )  # puf.weight[0, :5] = [0.9725952 2.7405396 7.4138794 3.347351  4.2404175]

    chall = generate_challenges(
        subkey1, (1000, puf.dim[1])
    )  # chall[0, :5] = [ 1 -1  1  1  1]

    rng, noisy_weights, sigma_error = similar_weight(
        subkey2, puf.weight, target=target_err
    )

    print(f"Target={target_err}")

    for i in range(puf.dim[0]):
        noisy_response = get_response(row_vec(noisy_weights[i]), chall).flatten()

        true_response = get_response(row_vec(puf.weight[i]), chall).flatten()
        accuracy = jnp.equal(noisy_response, true_response).mean()
        print(f"weight {i} % of responses equal: {accuracy}")