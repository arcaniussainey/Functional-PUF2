"""
perf_test.py  --  performance benchmark for Functional-PUF.

Run from the repo root::

    python perf_test.py [options]

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
import time

import jax
import jax.numpy as jnp
import numpy as np

from Pufs.FunctionalPuf import (
    generate_weights,
    generate_challenges,
    get_response,
    xor_get_response,
    noisy_get_response,
    Arbiter,
    Xor,
)



# Timing helpers


def _block(x: object) -> None:
    """Block until all pending JAX computation in *x* is complete."""
    if isinstance(x, (list, tuple)):
        for v in x:
            _block(v)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()  # type: ignore[union-attr]


def timeit(fn: object, reps: int, warmup: int) -> tuple[float, float, float]:
    """
    Time *fn* as both first-call and steady-state execution.

    The first call includes any JAX tracing/compilation and object setup costs
    incurred by that callable.  The steady-state mean/std are measured after
    additional warm-up calls and use block_until_ready() so asynchronous JAX
    dispatch does not under-report runtime.

    Returns:
        tuple[float, float, float]: (first_call_ms, steady_mean_ms, steady_std_ms)
    """
    t0 = time.perf_counter()
    _block(fn())  # type: ignore[operator]
    first_call_ms = (time.perf_counter() - t0) * 1000

    for _ in range(max(0, warmup - 1)):
        _block(fn())  # type: ignore[operator]

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _block(fn())  # type: ignore[operator]
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return first_call_ms, float(arr.mean()), float(arr.std())



# Display helpers


SEP = "-" * 68


def header(title: str) -> None:
    """Print a section header."""
    print(f"\n{SEP}\n  {title}\n{SEP}")


def row(label: str, first_ms: float, mean_ms: float, std_ms: float) -> None:
    """Print a single benchmark result row."""
    print(f"  {label:<38s}  first={first_ms:8.3f} ms  steady={mean_ms:8.3f} ms  ±{std_ms:.3f}")



# Benchmark groups


def bench_weights(cfg: object) -> None:
    """Benchmark generate_weights across various arbiter counts."""
    header("generate_weights(rng, dim)")
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    for nw in [1, 4, 16, cfg.nweights]:  # type: ignore[attr-defined]
        rng, sk = jax.random.split(rng)
        first, m, s = timeit(
            lambda sk=sk, nw=nw: generate_weights(  # type: ignore[attr-defined]
                sk, (nw, cfg.stages)
            ),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"k={nw}, stages={cfg.stages}", first, m, s)  # type: ignore[attr-defined]


def bench_response(cfg: object) -> None:
    """Benchmark get_response across various arbiter counts."""
    header(f"get_response(weight, challenge)  nc={cfg.nchall}")  # type: ignore[attr-defined]
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    rng, sk1, sk2 = jax.random.split(rng, 3)
    w_all = generate_weights(sk1, (cfg.nweights, cfg.stages))  # type: ignore[attr-defined]
    challenges = generate_challenges(sk2, (cfg.nchall, cfg.stages))  # type: ignore[attr-defined]
    for nw in [1, 4, cfg.nweights]:  # type: ignore[attr-defined]
        w_slice = w_all[:nw]
        first, m, s = timeit(
            lambda w=w_slice: get_response(w, challenges),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"k={nw}", first, m, s)


def bench_xor(cfg: object) -> None:
    """Benchmark xor_get_response across various arbiter counts."""
    header(f"xor_get_response(weight, challenge)  nc={cfg.nchall}")  # type: ignore[attr-defined]
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    rng, sk1, sk2 = jax.random.split(rng, 3)
    w_all = generate_weights(sk1, (cfg.nweights, cfg.stages))  # type: ignore[attr-defined]
    challenges = generate_challenges(sk2, (cfg.nchall, cfg.stages))  # type: ignore[attr-defined]
    for k in [2, 3, 4, min(cfg.nweights, 6)]:  # type: ignore[attr-defined]
        w_slice = w_all[:k]
        first, m, s = timeit(
            lambda w=w_slice: xor_get_response(w, challenges),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"k={k}", first, m, s)


def bench_noisy(cfg: object) -> None:
    """Benchmark noisy_get_response across sigma values."""
    nc = cfg.nchall  # type: ignore[attr-defined]
    header(f"noisy_get_response(rng, weight, challenge, sigma)  nc={nc}")
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    rng, sk1, sk2, sk3 = jax.random.split(rng, 4)
    weight = generate_weights(sk1, (1, cfg.stages))  # type: ignore[attr-defined]
    challenges = generate_challenges(sk2, (cfg.nchall, cfg.stages))  # type: ignore[attr-defined]
    for sigma in [0.1, 0.5, 2.0]:
        sig = jnp.float32(sigma)
        first, m, s = timeit(
            lambda sk=sk3, sig=sig: noisy_get_response(sk, weight, challenges, sig),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"sigma={sigma}", first, m, s)


def bench_arbiter(cfg: object) -> None:
    """Benchmark Arbiter construction + get_response."""
    header("Arbiter construction + get_response")
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    rng, _sk1, sk2 = jax.random.split(rng, 3)  # sk1 is unused intentionally
    challenges = generate_challenges(sk2, (cfg.nchall, cfg.stages))  # type: ignore[attr-defined]
    _ = challenges  # referenced inside lambda
    for stages in [32, 64, cfg.stages]:  # type: ignore[attr-defined]
        rng, sk = jax.random.split(rng)
        first, m, s = timeit(
            lambda sk=sk, st=stages: Arbiter(sk, (1, st)).get_response(
                generate_challenges(sk, (cfg.nchall, st))  # type: ignore[attr-defined]
            ),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"stages={stages}", first, m, s)


def bench_xor_class(cfg: object) -> None:
    """Benchmark Xor construction across various arbiter counts."""
    header("Xor construction  (calls generate_weights internally)")
    print(f"  {'label':<38s}  {'first':>15}  {'steady mean':>17}      std")
    print(f"  {'-'*38}  {'-'*15}  {'-'*17}  {'-'*6}")
    rng = jax.random.PRNGKey(cfg.seed)  # type: ignore[attr-defined]
    for k in [1, 3, 8, cfg.nweights]:  # type: ignore[attr-defined]
        rng, sk = jax.random.split(rng)
        dim = (k, cfg.stages)  # type: ignore[attr-defined]
        first, m, s = timeit(
            lambda sk=sk, dim=dim: Xor(sk, dim),
            cfg.reps, cfg.warmup,  # type: ignore[attr-defined]
        )
        row(f"k={k}, stages={cfg.stages}", first, m, s)  # type: ignore[attr-defined]



# Main


GROUPS = {
    "weights":   bench_weights,
    "response":  bench_response,
    "xor":       bench_xor,
    "noisy":     bench_noisy,
    "arbiter":   bench_arbiter,
    "xor_class": bench_xor_class,
}


def main() -> None:
    """Parse CLI arguments and run the requested benchmark groups."""
    p = argparse.ArgumentParser()
    p.add_argument("--quick",    action="store_true")
    p.add_argument("--nchall",   type=int, default=50_000)
    p.add_argument("--nweights", type=int, default=32)
    p.add_argument("--stages",   type=int, default=128)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--only",     choices=list(GROUPS), default=None)
    args = p.parse_args()

    class Cfg:  # pylint: disable=too-few-public-methods
        """Benchmark configuration (data-holder; not a general-purpose class)."""

        nchall   = args.nchall
        nweights = args.nweights
        stages   = args.stages
        seed     = args.seed
        reps     = 8  if args.quick else 30
        warmup   = 2  if args.quick else 4

    print(f"\n{'='*68}")
    print("  Functional-PUF  :  performance benchmark")
    print(f"{'='*68}")
    print(
        f"  nchall={Cfg.nchall:,}  nweights={Cfg.nweights}  stages={Cfg.stages}"
        f"  reps={Cfg.reps}  backend={jax.default_backend()}"
    )

    for name in ([args.only] if args.only else list(GROUPS)):
        GROUPS[name](Cfg)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
