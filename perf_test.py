"""
perf_bench.py  —  performance benchmark for Functional-PUF
Run from the repo root:  python perf_bench.py [options]

Options
-------
--quick              2 warmup, 8 timed reps
--nchall   N         challenge count        (default 50 000)
--nweights N         max k for XOR tests    (default 32)
--stages   N         stages per arbiter     (default 128)
--seed     N         PRNG seed              (default 0)
--only     NAME      one group: weights | response | xor | noisy | arbiter | xor_class
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import (
    generate_weights,
    generate_challenges,
    generate_mem_weights,
    get_response,
    get_delta_response,
    xor_get_response,
    noisy_get_response,
    noisy_generate_weights,
    Arbiter,
    Xor,
    n_new_keys,
    row_vec,
)

# Timing

def _block(x):
    if isinstance(x, (list, tuple)):
        for v in x:
            _block(v)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()


def timeit(fn, reps, warmup):
    for _ in range(warmup):
        _block(fn())
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _block(fn())
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return float(a.mean()), float(a.std())


# Display funcs

SEP = "-" * 68

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label, mean_ms, std_ms):
    print(f"  {label:<38s}  {mean_ms:8.3f} ms  ±{std_ms:.3f}")


# Benchmark

def bench_weights(cfg):
    header("generate_weights(rng, dim)")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    for nw in [1, 4, 16, cfg.nweights]:
        rng, sk = jax.random.split(rng)
        m, s = timeit(lambda sk=sk, nw=nw: generate_weights(sk, (nw, cfg.stages)),
                      cfg.reps, cfg.warmup)
        row(f"k={nw}, stages={cfg.stages}", m, s)


def bench_response(cfg):
    header(f"get_response(weight, challenge)  nc={cfg.nchall}")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, sk1, sk2 = jax.random.split(rng, 3)
    W_all = generate_weights(sk1, (cfg.nweights, cfg.stages))
    C     = generate_challenges(sk2, (cfg.nchall, cfg.stages))
    for nw in [1, 4, cfg.nweights]:
        W = W_all[:nw]
        m, s = timeit(lambda W=W: get_response(W, C), cfg.reps, cfg.warmup)
        row(f"k={nw}", m, s)


def bench_xor(cfg):
    header(f"xor_get_response(weight, challenge)  nc={cfg.nchall}")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, sk1, sk2 = jax.random.split(rng, 3)
    W_all = generate_weights(sk1, (cfg.nweights, cfg.stages))
    C     = generate_challenges(sk2, (cfg.nchall, cfg.stages))
    for k in [2, 3, 4, min(cfg.nweights, 6)]:
        W = W_all[:k]
        m, s = timeit(lambda W=W: xor_get_response(W, C), cfg.reps, cfg.warmup)
        row(f"k={k}", m, s)


def bench_noisy(cfg):
    header(f"noisy_get_response(rng, weight, challenge, sigma)  nc={cfg.nchall}")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, sk1, sk2, sk3 = jax.random.split(rng, 4)
    W = generate_weights(sk1, (1, cfg.stages))
    C = generate_challenges(sk2, (cfg.nchall, cfg.stages))
    for sigma in [0.1, 0.5, 2.0]:
        sig = jnp.float32(sigma)
        m, s = timeit(lambda sk=sk3, sig=sig: noisy_get_response(sk, W, C, sig),
                      cfg.reps, cfg.warmup)
        row(f"sigma={sigma}", m, s)


def bench_arbiter(cfg):
    header("Arbiter construction + get_response")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, sk1, sk2 = jax.random.split(rng, 3)
    C = generate_challenges(sk2, (cfg.nchall, cfg.stages))
    for stages in [32, 64, cfg.stages]:
        rng, sk = jax.random.split(rng)
        m, s = timeit(lambda sk=sk, st=stages: Arbiter(sk, (1, st)).get_response(
                          generate_challenges(sk, (cfg.nchall, st))),
                      cfg.reps, cfg.warmup)
        row(f"stages={stages}", m, s)


def bench_xor_class(cfg):
    header("Xor construction  (calls generate_weights internally)")
    print(f"  {'label':<38s}  {'mean':>8}      std")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)
    for k in [1, 3, 8, cfg.nweights]:
        rng, sk = jax.random.split(rng)
        dim = (k, cfg.stages)
        m, s = timeit(lambda sk=sk, dim=dim: Xor(sk, dim), cfg.reps, cfg.warmup)
        row(f"k={k}, stages={cfg.stages}", m, s)


# Main

GROUPS = {
    "weights":   bench_weights,
    "response":  bench_response,
    "xor":       bench_xor,
    "noisy":     bench_noisy,
    "arbiter":   bench_arbiter,
    "xor_class": bench_xor_class,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",    action="store_true")
    p.add_argument("--nchall",   type=int, default=50_000)
    p.add_argument("--nweights", type=int, default=32)
    p.add_argument("--stages",   type=int, default=128)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--only",     choices=list(GROUPS), default=None)
    args = p.parse_args()

    class Cfg:
        nchall   = args.nchall
        nweights = args.nweights
        stages   = args.stages
        seed     = args.seed
        reps     = 8  if args.quick else 30
        warmup   = 2  if args.quick else 4

    print(f"\n{'='*68}")
    print(f"  Functional-PUF  :  performance benchmark")
    print(f"{'='*68}")
    print(f"  nchall={Cfg.nchall:,}  nweights={Cfg.nweights}  stages={Cfg.stages}"
          f"  reps={Cfg.reps}  backend={jax.default_backend()}")

    for name in ([args.only] if args.only else list(GROUPS)):
        GROUPS[name](Cfg)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
