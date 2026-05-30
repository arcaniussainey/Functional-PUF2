from __future__ import annotations

import csv
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
try:
    from evosax import Strategies  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - depends on optional attack dependency
    Strategies = None  # type: ignore[assignment]

from Pufs.FunctionalPuf import (
    PRNGKey, Weight, Challenge, Response,
    Arbiter, Xor,
    row_vec,
    generate_challenges,
    get_response,
    get_delta_response,
    xor_get_response,
    noisy_generate_weights,
    noisy_xor_get_response,
    noisy_get_response,
    get_sigma_error,
    target_error,
    new_key,
)
from Pufs.attack_state import AttackContext
from Pufs.randomness import DeterministicKeyStream, next_key as _stream_next_key


def _next_key(key_stream: Optional[DeterministicKeyStream] = None) -> PRNGKey:
    """Return a deterministic stream key when supplied, else legacy new_key()."""
    return _stream_next_key(key_stream, fallback=new_key)


def _make_strategy(model_name: str, pop_size: int, num_dims: int) -> object:
    """Create an evosax strategy or raise a clear optional-dependency error."""
    if Strategies is None:
        raise ImportError(
            "FunAttack_v2 evolutionary attack functions require the optional "
            "dependency `evosax`. Install evosax to run x1_attk, xn_attk, "
            "arbiter_direct_attack, xor_attk, or ipuf_attk."
        )
    return Strategies[model_name](popsize=pop_size, num_dims=num_dims)



# Standalone logging (kept for backwards-compatible call sites)


_GLOBAL_LOG: List[Dict] = []


def log_data(data: Dict, log_path: str = "Funlog/log.csv") -> None:
    """
    Append *data* to the module-level log list.

    Prefer ``AttackContext.log_data()`` for new code so that log state is
    not global.

    Args:
        data (Dict): dictionary of results to record.
        log_path (str): unused in this overload; kept for API compatibility.
    """
    _ = log_path
    _GLOBAL_LOG.append(data)



# Hashing / identity helpers


def hash_w(w: Weight) -> int:
    """
    Compute a hash over a weight matrix for identity checking.

    Args:
        w (Weight): weight matrix.

    Returns:
        int: hash of the flattened, stringified weight values.
    """
    w_str = "".join([str(x) for x in w.flatten().tolist()])
    return hash(w_str)



# Evolution-strategy loops


def run_es_loop(
    rng: PRNGKey,
    num_steps: int,
    fit_fn: object,
    model: object,
) -> Tuple[jax.Array, object]:
    """
    Run a standard (noise-free) evolution-strategy loop.

    Scan through *num_steps* evolution rollouts.  Your ``fit_fn`` should
    accept ``(rng, w)`` as its first two parameters and must not read or
    write external state (required for ``jax.lax.scan``).

    See https://jax.readthedocs.io/en/latest/jax-101/07-state.html
    and the evosax README ("Scan Through Evolution Rollouts").

    Args:
        rng (PRNGKey): PRNG key.
        num_steps (int): number of evolution steps.
        fit_fn: fitness function ``(rng, candidates) -> fitness_array``.
        model: evosax strategy instance.

    Returns:
        Tuple[jax.Array, object]: (mean_fitness_over_scan, final_es_state)
    """
    es_params = model.default_params  # type: ignore[attr-defined]
    state = model.initialize(rng, es_params)  # type: ignore[attr-defined]

    def es_step(state_input: list, _tmp: jax.Array) -> Tuple[list, jax.Array]:
        """Single evolution step: ask -> evaluate -> tell."""
        rng_inner, state_inner = state_input
        rng_inner, rng_iter = jax.random.split(rng_inner)
        x, state_inner = model.ask(rng_iter, state_inner, es_params)  # type: ignore[attr-defined]
        fitness = fit_fn(rng_iter, x).flatten()  # type: ignore[operator]
        state_inner = model.tell(x, fitness, state_inner, es_params)  # type: ignore[attr-defined]
        return [rng_inner, state_inner], fitness[jnp.argmin(fitness)]

    state, scan_out = jax.lax.scan(es_step, [rng, state], [jnp.zeros(num_steps)])
    return jnp.mean(scan_out), state


def noisy_run_es_loop(
    rng: PRNGKey,
    num_steps: int,
    fit_fn: object,
    model: object,
    sigma_error: jax.Array,
) -> Tuple[jax.Array, object]:
    """
    Run a noisy evolution-strategy loop.

    Like ``run_es_loop`` but perturbs candidate weights with Gaussian noise
    before fitness evaluation, simulating a noisy PUF environment.

    Args:
        rng (PRNGKey): PRNG key.
        num_steps (int): number of evolution steps.
        fit_fn: fitness function ``(rng, candidates) -> fitness_array``.
        model: evosax strategy instance.
        sigma_error (jax.Array): noise std deviation applied to candidates.

    Returns:
        Tuple[jax.Array, object]: (mean_fitness_over_scan, final_es_state)
    """
    es_params = model.default_params  # type: ignore[attr-defined]
    state = model.initialize(rng, es_params)  # type: ignore[attr-defined]

    def es_step(state_input: list, _tmp: jax.Array) -> Tuple[list, jax.Array]:
        """Single noisy evolution step: ask -> perturb -> evaluate -> tell."""
        rng_inner, state_inner = state_input
        rng_inner, *subkey = jax.random.split(rng_inner, 4)
        x, state_inner = model.ask(subkey[0], state_inner, es_params)  # type: ignore[attr-defined]
        _rng_unused, noisy_weight = noisy_generate_weights(subkey[1], x, sigma_error)
        fitness = fit_fn(subkey[2], noisy_weight)  # type: ignore[operator]
        state_inner = model.tell(x, fitness.flatten(), state_inner, es_params)  # type: ignore[attr-defined]
        return [rng_inner, state_inner], fitness[jnp.argmin(fitness)]

    state, scan_out = jax.lax.scan(es_step, [rng, state], [jnp.zeros(num_steps)])
    return jnp.mean(scan_out), state



# Sorting helper


@jax.jit
def sort_on(
    arr: jax.Array,
    on: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Sort *arr* by the values in *on*.

    Args:
        arr (jax.Array): array to sort.
        on (jax.Array): sort key array (same length as *arr*).

    Returns:
        Tuple[jax.Array, jax.Array]: (sorted_arr, sorted_on)
    """
    sorted_on, sort_idxs = jax.lax.sort_key_val(on, jnp.arange(0, on.shape[0]))
    sorted_arr = arr[sort_idxs]
    return sorted_arr, sorted_on



# Fitness functions


def fitness_direct_comparison(
    rng: PRNGKey,
    w: Weight,
    w2: Weight,
    c: Challenge,
) -> jax.Array:
    """
    Fitness based on direct comparison of two weight arrays' responses.

    Args:
        rng (PRNGKey): PRNG key (unused; kept for uniform fitness signature).
        w (Weight): candidate weight array.
        w2 (Weight): reference weight array.
        c (Challenge): challenge matrix.

    Returns:
        jax.Array: mean XOR mismatch rate (lower is better).
    """
    _ = rng
    r1 = get_response(w, c).flatten()
    r2 = get_response(w2, c).flatten()
    fitness = jnp.bitwise_xor(r1, r2)
    return fitness.mean()


def filter_challenges(
    puf: Xor,
    c: Challenge,
    threshold: float,
) -> Challenge:
    """
    Filter a challenge matrix, keeping only challenges above the threshold.

    Uses the PUF's delta response: a challenge is kept only when *all*
    of its per-arbiter absolute delays exceed ``sigma * threshold``.

    Args:
        puf (Xor): PUF whose delta response is used for filtering.
        c (Challenge): candidate challenge matrix.
        threshold (float): acceptance threshold multiplier on delta std.

    Returns:
        Challenge: filtered challenge matrix containing only accepted rows.
    """
    delta = get_delta_response(puf.weight, c)
    sigma = delta.std(axis=0)
    accept = jax.vmap(
        lambda d: jnp.greater_equal(jnp.abs(d), sigma * threshold).sum(),
        in_axes=(0,),
    )
    accept_idx = jnp.equal(accept(delta), delta.shape[1])
    return c[accept_idx, :]


def fitness_chall_filter(
    rng: PRNGKey,
    w: Weight,
    c: Challenge,
    threshold: float,
    ctx: AttackContext,
) -> jax.Array:
    """
    Fitness based on filtered challenges and a given threshold.

    The fraction of challenges that survive the acceptance filter is the
    fitness score (lower means more challenges pass, i.e. better).

    Args:
        rng (PRNGKey): PRNG key (unused; kept for uniform fitness signature).
        w (Weight): candidate weight array.
        c (Challenge): challenge matrix.
        threshold (float): acceptance threshold.
        ctx (AttackContext): attack context supplying ``c_sample``.

    Returns:
        jax.Array: scalar float32 fitness value.
    """
    _ = rng
    sigma_hat = get_delta_response(w, ctx.c_sample).std(axis=0).reshape((1, -1))
    delta = get_delta_response(w, c)
    accept = jax.vmap(
        lambda _delta: jnp.greater_equal(jnp.abs(_delta), sigma_hat * threshold).sum(),
        in_axes=(0,),
    )
    fitness = jnp.not_equal(accept(delta), w.shape[0])
    return fitness.mean().astype(jnp.float32)


def xorn_fitness(
    rng: PRNGKey,
    w: Weight,
    prev_ws: Weight,
    valid_chall: Challenge,
    threshold: float,
    alpha: float,
    ctx: AttackContext,
) -> jax.Array:
    """
    Combined fitness: challenge-filter score plus a diversity penalty.

    The penalty penalises candidate weights that are too similar to
    previously learned weights.

    Args:
        rng (PRNGKey): PRNG key.
        w (Weight): candidate weight vector.
        prev_ws (Weight): matrix of previously learned weights.
        valid_chall (Challenge): validation challenge set.
        threshold (float): challenge-filter threshold.
        alpha (float): penalty scaling coefficient.
        ctx (AttackContext): attack context supplying ``c_sample``.

    Returns:
        jax.Array: scalar fitness value.
    """
    filter_score = fitness_chall_filter(rng, row_vec(w), valid_chall, threshold, ctx)
    prev_eql = jax.vmap(
        lambda _prev_weight: fitness_direct_comparison(
            rng, row_vec(w), row_vec(_prev_weight), valid_chall
        )
    )
    pct_eql = prev_eql(prev_ws)
    penalty = (jnp.where(pct_eql > 1 - pct_eql, pct_eql, 1 - pct_eql) * 2).mean()
    penalty = penalty * alpha
    fitness = (filter_score + penalty) / 2
    return fitness



@jax.jit
def _batch_fitness_chall_filter(
    candidates: Weight,
    chall: Challenge,
    threshold: jax.Array,
    c_sample: Challenge,
) -> jax.Array:
    """Vectorised challenge-filter fitness for an ES candidate population.

    Kept at module scope so JAX can cache the compiled program by input shape
    instead of seeing a fresh Python closure on every retry.
    """

    def _score(w: Weight) -> jax.Array:
        sigma_hat = get_delta_response(row_vec(w), c_sample).std(axis=0).reshape((1, -1))
        delta = get_delta_response(row_vec(w), chall)
        accept = jnp.greater_equal(jnp.abs(delta), sigma_hat * threshold).sum(axis=1)
        return jnp.not_equal(accept, 1).mean().astype(jnp.float32)

    return jax.vmap(_score)(candidates)


@jax.jit
def _batch_xorn_fitness(
    candidates: Weight,
    prev_ws: Weight,
    chall: Challenge,
    threshold: jax.Array,
    alpha: jax.Array,
    c_sample: Challenge,
) -> jax.Array:
    """Vectorised subsequent-arbiter fitness for an ES population."""

    def _score(w: Weight) -> jax.Array:
        w_row = row_vec(w)
        sigma_hat = get_delta_response(w_row, c_sample).std(axis=0).reshape((1, -1))
        delta = get_delta_response(w_row, chall)
        accept = jnp.greater_equal(jnp.abs(delta), sigma_hat * threshold).sum(axis=1)
        filter_score = jnp.not_equal(accept, 1).mean().astype(jnp.float32)

        response = get_response(w_row, chall).flatten()

        def _prev_similarity(prev_weight: Weight) -> jax.Array:
            prev_response = get_response(row_vec(prev_weight), chall).flatten()
            return jnp.bitwise_xor(response, prev_response).mean()

        pct_eql = jax.vmap(_prev_similarity)(prev_ws)
        penalty = (jnp.where(pct_eql > 1 - pct_eql, pct_eql, 1 - pct_eql) * 2).mean()
        return (filter_score + (penalty * alpha)) / 2

    return jax.vmap(_score)(candidates)



# Direct arbiter attack


def arbiter_direct_attack(
    chall: Challenge,
    pop_size: int = 32,
    n_generations: int = 500,
    model_name: str = "CMA_ES",
) -> Tuple[object, Weight]:
    """
    Evolutionary attack on a single Arbiter PUF.

    Args:
        chall (Challenge): challenge matrix used for fitness evaluation.
        pop_size (int): ES population size.
        n_generations (int): number of ES generations.
        model_name (str): evosax strategy name.

    Returns:
        Tuple[object, Weight]: (final_es_state, learned_weight_row_vector)
    """
    _nchall, dim = chall.shape
    arb = Arbiter(new_key(), (1, dim))

    arb_direct_fitness = jax.jit(
        jax.vmap(
            lambda rng, w: fitness_direct_comparison(rng, row_vec(w), arb.weight, chall),
            in_axes=(None, 0),
        )
    )

    model = _make_strategy(model_name, pop_size, dim)
    _, state = run_es_loop(new_key(), n_generations, arb_direct_fitness, model)
    _, state = state

    lrn_w = row_vec(state.best_member)
    return state, lrn_w



# Single-arbiter XOR attack


def x1_attk(
    chall: Challenge,
    ctx: AttackContext,
    pop_size: int = 32,
    n_generations: int = 500,
    model_name: str = "CMA_ES",
    threshold: float = 2.0,
    sigma_error: Optional[jax.Array] = None,
) -> Tuple[object, Weight]:
    """
    Evolutionary attack targeting the first arbiter of a XOR PUF.

    Args:
        chall (Challenge): filtered challenge matrix.
        ctx (AttackContext): attack context (supplies ``c_sample``).
        pop_size (int): ES population size.
        n_generations (int): number of ES generations.
        model_name (str): evosax strategy name.
        threshold (float): challenge-filter threshold.
        sigma_error (jax.Array | None): noise std deviation; if supplied,
            runs ``noisy_run_es_loop`` instead of ``run_es_loop``.

    Returns:
        Tuple[object, Weight]: (final_es_state, learned_weight_matrix)
    """
    _, dim = chall.shape
    model = _make_strategy(model_name, pop_size, dim)

    threshold_arr = jnp.float32(threshold)

    def x1_fit(rng: PRNGKey, candidates: Weight) -> jax.Array:
        """Population fitness wrapper; rng kept for run_es_loop signature."""
        _ = rng
        return _batch_fitness_chall_filter(candidates, chall, threshold_arr, ctx.c_sample)

    if sigma_error is not None:
        _, state = noisy_run_es_loop(ctx.new_key(), n_generations, x1_fit, model, sigma_error)
    else:
        _, state = run_es_loop(ctx.new_key(), n_generations, x1_fit, model)

    _, state = state
    lrn_w = row_vec(state.best_member)
    return state, jnp.vstack(lrn_w)



# Subsequent-arbiter XOR attack


def xn_attk(
    chall: Challenge,
    prev_w: Weight,
    ctx: AttackContext,
    pop_size: int = 32,
    n_generations: int = 500,
    model_name: str = "CMA_ES",
    threshold: float = 2.0,
    alpha: float = 0.43,
    sigma_error: Optional[jax.Array] = None,
) -> Tuple[object, Weight]:
    """
    Evolutionary attack targeting subsequent arbiters of a XOR PUF.

    Unlike ``x1_attk``, this takes already-learned weights (``prev_w``) and
    adds a diversity penalty so the new candidate is not isomorphic to them.

    Args:
        chall (Challenge): filtered challenge matrix.
        prev_w (Weight): previously learned weight rows.
        ctx (AttackContext): attack context.
        pop_size (int): ES population size.
        n_generations (int): number of ES generations.
        model_name (str): evosax strategy name.
        threshold (float): challenge-filter threshold.
        alpha (float): diversity-penalty coefficient.
        sigma_error (jax.Array | None): noise std deviation.

    Returns:
        Tuple[object, Weight]: (final_es_state, learned_weight_row_vector)
    """
    _, dim = chall.shape
    model = _make_strategy(model_name, pop_size, dim)

    threshold_arr = jnp.float32(threshold)
    alpha_arr = jnp.float32(alpha)

    def xn_fit(rng: PRNGKey, candidates: Weight) -> jax.Array:
        """Population fitness wrapper; rng kept for run_es_loop signature."""
        _ = rng
        return _batch_xorn_fitness(candidates, prev_w, chall, threshold_arr, alpha_arr, ctx.c_sample)

    if sigma_error is not None:
        _, state = noisy_run_es_loop(ctx.new_key(), n_generations, xn_fit, model, sigma_error)
    else:
        _, state = run_es_loop(ctx.new_key(), n_generations, xn_fit, model)

    _, state = state
    lrn_w = row_vec(state.best_member)
    return state, lrn_w



# Challenge-filter data collection


def xor_cfilt_data(n_trials: int) -> List[Dict]:
    """
    Collect challenge-filter statistics across dimensions, counts, and thresholds.

    Iterates over all combinations of (threshold, dim, nchall, nxor) and
    records how many challenges survive the filter for each configuration.
    Results are written to ``data/xor_challenge_filter_data.csv``.

    Args:
        n_trials (int): number of independent trials per configuration.

    Returns:
        List[Dict]: collected experiment records.
    """
    dims = [64, 128]
    nchalls = [10_000, 20_000, 50_000, 100_000]
    thresholds = [1, 1.5, 2, 2.5, 3]
    num_xors = [2, 3, 4, 5, 6]
    data: List[Dict] = []

    for threshold in thresholds:
        for d in dims:
            for n in nchalls:
                for nxor in num_xors:
                    for _ in range(n_trials):
                        c = generate_challenges(new_key(), (n, d))
                        xor_puf = Xor(new_key(), (nxor, d))
                        filtered_c = filter_challenges(xor_puf, c, threshold)
                        num_kept = filtered_c.shape[0]
                        pct_kept = num_kept / n
                        data.append(dict(
                            nchall=n,
                            dim=d,
                            threshold=threshold,
                            num_kept=num_kept,
                            pct_kept=pct_kept,
                            nxor=nxor,
                        ))

    with open("data/xor_challenge_filter_data.csv", "a+", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    return data



# Similarity utility


def similarity(
    w: Weight,
    compar_w: Weight,
    key_stream: Optional[DeterministicKeyStream] = None,
) -> jax.Array:
    """
    Compute the fraction of matching responses between *w* and each row of *compar_w*.

    Args:
        w (Weight): reference weight, shape (1, n).
        compar_w (Weight): comparison weight matrix, shape (m, n).

    Returns:
        jax.Array: similarity values, shape (m,).
    """
    c = generate_challenges(_next_key(key_stream), (25_000, w.shape[1]))

    sim = jax.vmap(
        lambda w2: jnp.equal(
            get_response(row_vec(w), c).flatten(),
            get_response(row_vec(w2), c).flatten(),
        ).mean()
    )
    return sim(compar_w)



# Challenge filter initialisation


def initialize_challenge_filter(
    nchall: int,
    puf: Xor,
    threshold: float,
    key_stream: Optional[DeterministicKeyStream] = None,
) -> Challenge:
    """
    Generate and filter challenges, doubling the candidate pool until enough pass.

    Args:
        nchall (int): desired number of post-filter challenges.
        puf (Xor): PUF used for filtering.
        threshold (float): challenge-filter threshold.

    Returns:
        Challenge: shape (nchall, n_stages) of accepted challenges.
    """
    chall = generate_challenges(_next_key(key_stream), (nchall, puf.dim[1]))
    filt_chall = filter_challenges(puf, chall, threshold)

    factor = 2
    while filt_chall.shape[0] < nchall:
        factor *= 2
        chall = generate_challenges(_next_key(key_stream), (int(nchall * factor), puf.dim[1]))
        filt_chall = filter_challenges(puf, chall, threshold)

    return filt_chall[:nchall, :]



# Individual PUF attack


def attack_indv_puf(
    challenges: Challenge,
    attk_args: Dict,
    noise: bool,
    xor_puf: Xor,
    nxor: int,
    alphas: List[float],
    ctx: AttackContext,
) -> Tuple[Weight, List[jax.Array], float]:
    """
    Run the full sequential arbiter-learning attack on one XOR PUF.

    Learns each arbiter in turn: ``x1_attk`` for the first, then
    ``xn_attk`` for each subsequent one, accumulating the learned
    weight rows into a growing matrix.

    Args:
        challenges (Challenge): filtered challenge set.
        attk_args (Dict): read-only common keyword arguments forwarded to
            x1_attk / xn_attk.  This function intentionally does not mutate
            the caller-owned dictionary; per-arbiter alpha values are added
            to local copies only.
        noise (bool): whether to use a noisy ES loop.
        xor_puf (Xor): the target XOR PUF (used to obtain sigma_error).
        nxor (int): number of arbiters.
        alphas (List[float]): diversity-penalty alpha values for each subsequent arbiter.
        ctx (AttackContext): attack context.

    Returns:
        Tuple[Weight, List[jax.Array], float]:
            (learned_weight_matrix, best_fitness_per_step, total_time_seconds)
    """
    best_fitness: List[jax.Array] = []
    start_time = time.monotonic()

    # Compute sigma_error once if noise is requested
    sigma_error_vec: Optional[jax.Array] = None
    if noise:
        sigma_error_vec = get_sigma_error(ctx.new_key(), xor_puf.weight)

    # Learn first arbiter
    if noise and sigma_error_vec is not None:
        state1, lrn_w1 = x1_attk(challenges, ctx, **attk_args, sigma_error=sigma_error_vec[0])
    else:
        state1, lrn_w1 = x1_attk(challenges, ctx, **attk_args)

    best_fitness.append(state1.best_fitness)
    lrn_w = row_vec(lrn_w1)
    states = [state1]
    sigma_idx = 1

    # Learn remaining arbiters
    for alpha in alphas[: nxor - 1]:
        xn_args = {**attk_args, "alpha": alpha}
        if noise and sigma_error_vec is not None:
            state, lw = xn_attk(
                challenges, lrn_w, ctx,
                **xn_args,
                sigma_error=sigma_error_vec[sigma_idx],
            )
            sigma_idx += 1
        else:
            state, lw = xn_attk(challenges, lrn_w, ctx, **xn_args)

        best_fitness.append(state.best_fitness)
        states.append(state)
        lrn_w = jnp.vstack([lrn_w, row_vec(lw)])

    total_time = time.monotonic() - start_time
    return lrn_w, best_fitness, total_time



# Challenge transformation helpers (iPUF)


def transform_flip(c1: Challenge, r1: jax.Array) -> Challenge:
    """
    Interpose challenge transformation: insert the left-PUF response at mid-index.

    If ``r1[i] == 1``, insert +1 and flip the bits before mid-index.
    Otherwise, insert -1 and leave the original challenge unchanged.

    This implementation is vectorised to avoid Python loops over challenge
    rows; it preserves the original transformation semantics.
    """
    r_pm = jnp.where(jnp.asarray(r1).reshape((-1,)) == 1, 1, -1).astype(c1.dtype)
    mid_index = c1.shape[1] // 2
    left = c1[:, :mid_index]
    right = c1[:, mid_index:]
    transformed_left = jnp.where(r_pm[:, None] == 1, -left, left)
    return jnp.concatenate((transformed_left, r_pm[:, None], right), axis=1)


def reverse_transform(c2: Challenge) -> Challenge:
    """
    Reverse the interpose transformation to recover the original challenges.

    Vectorised inverse of ``transform_flip``.
    """
    mid_index = c2.shape[1] // 2
    marker = c2[:, mid_index]
    left = c2[:, :mid_index]
    right = c2[:, mid_index + 1:]
    restored_left = jnp.where(marker[:, None] == 1, -left, left)
    return jnp.concatenate((restored_left, right), axis=1)



# iPUF challenge initialisation


def initialize_ipuf_challenge_filter(
    nchall: int,
    lxor: Xor,
    rxor: Xor,
    threshold: float,
    key_stream: Optional[DeterministicKeyStream] = None,
) -> Tuple[Challenge, Challenge, Response]:
    """
    Generate and filter challenges for an Interpose PUF (IPUF).

    Iteratively generates candidates until *nchall* challenges survive
    filtering by both the left and right XOR PUFs.

    Args:
        nchall (int): desired number of challenges.
        lxor (Xor): left XOR PUF.
        rxor (Xor): right XOR PUF.
        threshold (float): filter threshold.

    Returns:
        Tuple[Challenge, Challenge, Response]:
            (c_filtered, c2, c2_responses)
            c_filtered -- challenges filtered by both PUFs, shape (nchall, n)
            c2         -- transformed challenges before filtering by right PUF
            c2_responses -- right PUF responses to c2
    """
    c2: jax.Array = jnp.empty((0, lxor.dim[1] + 1), dtype=jnp.int8)
    factor = 500

    while c2.shape[0] < nchall:
        factor *= 100
        candidate_challenges = generate_challenges(
            _next_key(key_stream), (int(nchall * factor), lxor.dim[1])
        )
        c1 = filter_challenges(lxor, candidate_challenges, threshold)

        _individual, true_r = lxor(c1)  # type: ignore[misc]
        r1 = true_r.flatten()

        c2 = transform_flip(c1, r1)
        c2 = filter_challenges(rxor, c2, threshold)

    c2 = c2[:nchall, :]

    # Get right XOR PUF responses to the filtered c2.
    _c2_individual, c2_r = rxor(c2)  # type: ignore[misc]

    c_filtered = reverse_transform(c2)[:nchall, :]
    return c_filtered, c2, c2_r



# Full XOR PUF attack


_DEFAULT_ALPHAS = [0.37, 0.55, 0.16, 0.39, 0.12, 0.23, 0.31, 0.47, 0.29, 0.34, 0.18, 0.26]


def xor_attk(  # pylint: disable=too-many-locals
    dim: Tuple[int, int] = (3, 128),
    nchall: int = 10_000,
    threshold: float = 0.5,
    n_generations: int = 1_000,
    model_name: str = "CMA_ES",
    pop_size: int = 32,
    alphas: Optional[List[float]] = None,
    noise: bool = False,
    rxor: int = 0,
    seed: int = 0,
) -> Dict:
    """
    Perform an evolutionary attack on a XOR PUF (or iPUF left half).

    Repeats the full attack loop until the validation accuracy exceeds 98 %.

    Args:
        dim (Tuple[int, int]): (nxor, n_stages); default (3, 128).
        nchall (int): number of challenge-response pairs; default 10 000.
        threshold (float): challenge-filter threshold; default 0.5.
        n_generations (int): ES generations; default 1 000.
        model_name (str): evosax strategy; default "CMA_ES".
        pop_size (int): ES population size; default 32.
        alphas (List[float] | None): per-arbiter diversity penalty; uses
            ``_DEFAULT_ALPHAS`` if None.
        noise (bool): enable noisy ES loop; default False.
        rxor (int | Xor): right XOR PUF for iPUF mode, or 0 for plain XOR.
        seed (int): deterministic seed controlling all v2 stochastic choices.

    Returns:
        Dict: attack results including accuracy, learned weights, timings, etc.
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    nxor, stages = dim
    accuracies: List[jax.Array] = []
    key_stream = DeterministicKeyStream(seed)
    wall_start = time.monotonic()
    setup_start = wall_start

    # Build target XOR PUF via the class.
    xor_puf = Xor(key_stream.next(), (nxor, stages))
    print("The original weights", xor_puf)

    # Initialise challenge set.
    if rxor == 0:
        c_filtered = initialize_challenge_filter(nchall, xor_puf, threshold, key_stream)
        c2 = c2_r = 0
    else:
        c_filtered, c2, c2_r = initialize_ipuf_challenge_filter(
            nchall, xor_puf, rxor, threshold, key_stream  # type: ignore[arg-type]
        )

    c_sample = generate_challenges(key_stream.next(), (5_000, c_filtered.shape[1]))
    ctx = AttackContext(c_sample=c_sample, key_stream=key_stream)

    # Validation challenges (filtered).
    val_raw = generate_challenges(ctx.new_key(), (25_000, stages))
    val_chall = filter_challenges(xor_puf, val_raw, threshold)
    setup_time = time.monotonic() - setup_start

    # Build base attack args.  Per-arbiter alpha values are added to local
    # copies inside attack_indv_puf; this dictionary remains read-only.
    attk_args: Dict = dict(
        pop_size=pop_size,
        n_generations=n_generations,
        model_name=model_name,
        threshold=threshold,
    )

    overall_acc = 0.45
    lrn_w: Optional[Weight] = None
    best_fitness: List[jax.Array] = []
    total_time: float = 0.0

    retry_count = 0
    validation_time = 0.0

    while overall_acc < 0.98:
        retry_count += 1
        lrn_w, best_fitness, total_time = attack_indv_puf(
            c_filtered, dict(attk_args), noise, xor_puf, nxor, alphas, ctx
        )

        val_start = time.monotonic()
        _val_indv, val_old_xor_r = xor_puf(val_chall)  # type: ignore[misc]
        val_old_xor_r = val_old_xor_r.flatten()

        lrn_individual, lrn_r = xor_get_response(lrn_w, val_chall)
        lrn_r = lrn_r.flatten()

        acc = jnp.equal(val_old_xor_r, lrn_r).mean()
        overall_acc = float(jnp.array([acc, 1 - acc]).max())
        validation_time += time.monotonic() - val_start

    assert lrn_w is not None  # guaranteed by loop above

    # Final learned responses on the filtered challenges (for iPUF chaining)
    lrn_r_actual_chal_individual, lrn_r_actual_chal = xor_get_response(lrn_w, c_filtered)
    lrn_r_actual_chal = lrn_r_actual_chal.flatten()

    acc = jnp.equal(val_old_xor_r, lrn_r).mean()  # type: ignore[possibly-undefined]
    print("Accuracy", acc)
    overall_acc_arr = jnp.array([acc, 1 - acc]).max()
    accuracies.append(overall_acc_arr)

    # Per-arbiter accuracy matrix
    indv_acc: List[jax.Array] = []
    for i in range(nxor):
        lrn_r_subset = lrn_individual[:, i]  # type: ignore[possibly-undefined]
        for j in range(nxor):
            r_subset = _val_indv[:, j]  # type: ignore[possibly-undefined]
            this_acc = jnp.equal(r_subset, lrn_r_subset).mean()
            print(f"Prediction acc wv_{i + 1} vs wv_{j + 1}", this_acc)
            indv_acc.append(this_acc)

    print(f"nchall={nchall}: {overall_acc_arr.mean()}")

    final_fitness = float(fitness_chall_filter(
        ctx.new_key(), lrn_w, c_filtered, threshold, ctx
    ))
    wall_time = time.monotonic() - wall_start

    data = dict(
        overall_acc=overall_acc_arr,
        model_name=model_name,
        dim=stages,
        pop_size=pop_size,
        n_generations=n_generations,
        threshold=threshold,
        nchall=nchall,
        alphas=alphas,
        final_fitness=final_fitness,
        noise=noise,
        puf_type=f"xor {xor_puf.dim}",
        weight_hash=hash_w(xor_puf.weight),
        nxors=nxor,
        best_fitness=best_fitness,
        total_time=total_time,
        setup_time=setup_time,
        validation_time=validation_time,
        wall_time=wall_time,
        retry_count=retry_count,
        seed=seed,
        indv_acc=indv_acc,
        xor_filt_chall=c_filtered,
        lrn_r=lrn_r_actual_chal,
        c2_r=c2_r,
        c2=c2,
    )

    for i, alpha in enumerate(alphas):
        data[f"alpha{i + 1}"] = alpha
    for i in range(len(alphas), 5):
        data[f"alpha{i + 1}"] = None

    print("TOTAL TIME IS", total_time)
    print(data)
    return data



# Full iPUF attack


def ipuf_attk(  # pylint: disable=too-many-locals
    xor1_dim: Tuple[int, int] = (3, 128),
    dim: Tuple[int, int] = (3, 129),
    nchall: int = 5_000,
    threshold: float = 0.75,
    n_generations: int = 1_000,
    model_name: str = "CMA_ES",
    pop_size: int = 32,
    alphas: Optional[List[float]] = None,
    noise: bool = False,
    seed: int = 0,
) -> Dict:
    """
    Perform an evolutionary attack on an Interpose PUF (IPUF).

    First attacks the left XOR PUF, then uses the learned responses to
    construct transformed challenges for attacking the right XOR PUF.

    Args:
        xor1_dim (Tuple[int, int]): dimensions of the left XOR PUF.
        dim (Tuple[int, int]): IPUF dimensions (one stage higher than left).
        nchall (int): number of challenge-response pairs.
        threshold (float): challenge-filter threshold.
        n_generations (int): ES generations.
        model_name (str): evosax strategy.
        pop_size (int): ES population size.
        alphas (List[float] | None): per-arbiter diversity penalty.
        noise (bool): enable noisy ES loop.
        seed (int): deterministic seed controlling all v2 stochastic choices.

    Returns:
        Dict: attack results.
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    nxor, stages = dim
    accuracies: List[jax.Array] = []
    key_stream = DeterministicKeyStream(seed)

    xor_puf = Xor(key_stream.next(), (nxor, stages))
    print("The original weights of iPUF", xor_puf)

    # Attack the left XOR PUF
    left_data = xor_attk(
        xor1_dim, nchall, threshold, n_generations, model_name, pop_size,
        alphas, noise, xor_puf, seed=seed + 1
    )

    c_filtered   = jnp.array(left_data["xor_filt_chall"], dtype=jnp.int32)
    lrn_r        = jnp.array(left_data["lrn_r"],          dtype=jnp.int32)

    # Build transformed challenges for the right PUF
    new_challenges = transform_flip(c_filtered, lrn_r)
    new_filtered   = new_challenges

    # Build context for the right-PUF attack.
    c_sample = generate_challenges(key_stream.next(), (5_000, new_challenges.shape[1]))
    ctx = AttackContext(c_sample=c_sample, key_stream=key_stream)

    # True right-PUF responses on the filtered challenges.
    _true_indv, true_r = xor_puf(new_filtered)  # type: ignore[misc]
    true_r = true_r.flatten()

    # Validation challenges.
    val_raw = generate_challenges(ctx.new_key(), (25_000, stages))
    val_chall = filter_challenges(xor_puf, val_raw, threshold)

    attk_args: Dict = dict(
        pop_size=pop_size,
        n_generations=n_generations,
        model_name=model_name,
        threshold=threshold,
    )

    overall_acc = 0.5
    lrn_w: Optional[Weight] = None
    best_fitness: List[jax.Array] = []
    total_time: float = 0.0
    lrn_individual: Optional[jax.Array] = None
    _val_indv: Optional[jax.Array] = None
    val_old_xor_r: Optional[jax.Array] = None
    lrn_r_val: Optional[jax.Array] = None

    while overall_acc < 0.98:
        lrn_w, best_fitness, total_time = attack_indv_puf(
            new_filtered, dict(attk_args), noise, xor_puf, nxor, alphas, ctx
        )

        _val_indv, val_old_xor_r = xor_puf(val_chall)  # type: ignore[misc]
        val_old_xor_r = val_old_xor_r.flatten()

        lrn_individual, lrn_r_val = xor_get_response(lrn_w, val_chall)
        lrn_r_val = lrn_r_val.flatten()

        acc = jnp.equal(val_old_xor_r, lrn_r_val).mean()
        print("Accuracy", acc)
        overall_acc = float(jnp.array([acc, 1 - acc]).max())

    assert lrn_w is not None

    indv_acc: List[jax.Array] = []
    for i in range(nxor):
        lrn_r_subset = lrn_individual[:, i]  # type: ignore[index]
        for j in range(nxor):
            r_subset = _val_indv[:, j]  # type: ignore[index]
            this_acc = jnp.equal(r_subset, lrn_r_subset).mean()
            print(f"Prediction acc wv_{i + 1} vs wv_{j + 1}", this_acc)
            indv_acc.append(this_acc)

    print(f"nchall={nchall}: {jnp.array([acc, 1 - acc]).max().mean()}")  # type: ignore[possibly-undefined]

    final_fitness = float(fitness_chall_filter(
        ctx.new_key(), lrn_w, new_filtered, threshold, ctx
    ))

    data = dict(
        overall_acc=jnp.array([acc, 1 - acc]).max(),  # type: ignore[possibly-undefined]
        model_name=model_name,
        dim=stages,
        pop_size=pop_size,
        n_generations=n_generations,
        threshold=threshold,
        nchall=nchall,
        alphas=alphas,
        final_fitness=final_fitness,
        noise=noise,
        puf_type=f"xor {xor_puf.dim}",
        weight_hash=hash_w(xor_puf.weight),
        nxors=nxor,
        best_fitness=best_fitness,
        total_time=total_time,
        seed=seed,
        indv_acc=indv_acc,
    )

    for i, alpha in enumerate(alphas):
        data[f"alpha{i + 1}"] = alpha
    for i in range(len(alphas), 5):
        data[f"alpha{i + 1}"] = None

    print("TOTAL TIME IS", total_time)
    return data



# Noisy challenge-filter experiment


def noisy_challenge_filter_test(
    dims: Optional[List[Tuple[int, int]]] = None,
    targets: Optional[List[float]] = None,
    nsamples: int = 25,
    threshold: float = 1.0,
    data_path: str = "data/noise_filter.csv",
    seed: int = 0,
) -> List[Dict]:
    """
    Evaluate XOR PUF accuracy under various noise levels and dimensions.

    Iterates over all combinations of (dim, target_accuracy) and records
    how well a noisy XOR PUF matches the true response.

    Args:
        dims (List[Tuple[int, int]] | None): list of (nxor, n_stages) tuples.
        targets (List[float] | None): target accuracy values to test.
        nsamples (int): trials per configuration.
        threshold (float): filter threshold (currently unused in the inner loop).
        data_path (str): CSV output path.
        seed (int): deterministic seed for this experiment.

    Returns:
        List[Dict]: collected experiment records.
    """
    if dims is None:
        dims = [(3, 128), (4, 128), (5, 128), (6, 128)]
    if targets is None:
        targets = [0.8, 0.85, 0.9, 0.95, 0.99]

    _ = threshold  # parameter reserved for future filtering inside the loop

    data: List[Dict] = []
    trial = 1
    key_stream = DeterministicKeyStream(seed)

    for dim in dims:
        print(f"trial {trial}")
        trial += 1

        for target in targets:
            for _ in range(nsamples):
                puf = Xor(key_stream.next(), dim)

                c = generate_challenges(key_stream.next(), (50_000, puf.dim[1]))

                sigma_keys = jax.random.split(key_stream.next(), puf.weight.shape[0])
                get_factor = jax.vmap(
                    lambda rng_k, w: target_error(rng_k, w, c, target=target, nsamples=1_000),
                    in_axes=(0, 0),
                )
                sigma_error = get_factor(sigma_keys, puf.weight).flatten()

                val_chall = generate_challenges(key_stream.next(), (50_000, puf.dim[1]))

                # Noisy XOR response
                _, noisy_r = noisy_xor_get_response(key_stream.next(), puf.weight, val_chall, sigma_error)
                noisy_r = noisy_r.flatten()

                # True XOR response via class
                _puf_indv, puf_r = puf(val_chall)  # type: ignore[misc]
                xor_accuracy = float((noisy_r == puf_r.flatten()).mean())

                run_data: Dict = dict(
                    dim=dim[1],
                    nxor=dim[0],
                    xor_accuracy=xor_accuracy,
                    threshold=threshold,
                    target=target,
                )

                for i in range(3, 7):
                    run_data[f"p{i + 1}_accuracy"] = None
                    run_data[f"sigma{i + 1}"] = None

                for i in range(dim[0]):
                    arb_w = puf.get_weight(i)
                    arb_resp = get_response(arb_w, c).flatten()
                    _, noise_resp = noisy_get_response(
                        key_stream.next(), arb_w, c, sigma_error[dim[0] - i]
                    )
                    run_data[f"p{i + 1}_accuracy"] = float((arb_resp == noise_resp.flatten()).mean())
                    run_data[f"sigma{i + 1}"] = float(sigma_error[i])

                data.append(run_data)

    with open(data_path, "a+", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    return data
