"""
Pipeline composition examples for Arbiter, XOR, and feed-forward PUFs.

Run from the repository root:
    python examples/06_pipeline_composition.py
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from Pufs.feedforward_core import required_xor_weight_rows
from Pufs.pipeline import (
    PipelineState,
    add_gaussian_noise,
    age,
    apply,
    compose,
    evaluate_feedforward_xor_response,
    evaluate_response,
    evaluate_xor_response,
    generate_challenges,
    generate_weights,
    use_challenges,
)
from Pufs.primitives import generate_challenges as generate_fixed_challenges
from Pufs.statistics import reliability, uniformity


def _log_weight_norm(state: PipelineState) -> PipelineState:
    """Example custom Python step that logs the current weight norm."""
    print(f"  [debug] weight L2 norm={float(jnp.linalg.norm(state.weight)):.3f}")
    return state


def main() -> None:
    """Run several increasingly complex immutable pipeline examples."""
    rng = jax.random.PRNGKey(0)

    rng, run_key = jax.random.split(rng)
    minimal = compose(
        generate_weights(n_stages=64),
        generate_challenges(n_challenges=1_000),
        evaluate_response(),
    ).run(run_key)
    print("1. Minimal Arbiter PUF")
    print(f"  response shape={minimal.response.shape}")
    print(f"  uniformity={uniformity(minimal.response):.3f}")

    rng, run_key = jax.random.split(rng)
    noisy = compose(
        generate_weights(n_stages=64),
        add_gaussian_noise(sigma=0.5),
        generate_challenges(n_challenges=1_000),
        evaluate_response(),
    ).run(run_key)
    print("\n2. Noisy Arbiter PUF")
    print(f"  response shape={noisy.response.shape}")

    rng, run_key = jax.random.split(rng)
    xor_state = compose(
        generate_weights(n_stages=64, k=3),
        age(n_steps=200, sigma_per_step=0.01),
        generate_challenges(n_challenges=2_000),
        evaluate_xor_response(),
    ).run(run_key)
    individual, xor_response = xor_state.response
    print("\n3. Aged 3-XOR PUF")
    print(f"  individual shape={individual.shape}")
    print(f"  xor shape={xor_response.shape}")
    print(f"  xor uniformity={uniformity(xor_response):.3f}")

    rng, challenge_key, puf_key_a, puf_key_b = jax.random.split(rng, 4)
    fixed_challenges = generate_fixed_challenges(challenge_key, (1_000, 32))
    first = compose(
        generate_weights(n_stages=32),
        use_challenges(fixed_challenges),
        evaluate_response(),
    ).run(puf_key_a)
    second = compose(
        generate_weights(n_stages=32),
        use_challenges(fixed_challenges),
        evaluate_response(),
    ).run(puf_key_b)
    print("\n4. Two devices on the same challenges")
    print(f"  agreement={reliability(first.response, second.response):.3f}; ideal near 0.5")

    rng, run_key = jax.random.split(rng)
    print("\n5. Custom logging step")
    compose(
        generate_weights(n_stages=32),
        apply(_log_weight_norm, label="log_weight_norm"),
        add_gaussian_noise(sigma=0.3),
        apply(_log_weight_norm, label="log_weight_norm_after_noise"),
        generate_challenges(n_challenges=100),
        evaluate_response(),
    ).run(run_key)

    rng, run_key = jax.random.split(rng)
    loop_specs = (((2, 20),), ((4, 24), (8, 28)), ((1, 18),))
    rows = required_xor_weight_rows(loop_specs)
    ff_xor = compose(
        generate_weights(n_stages=32, k=rows),
        generate_challenges(n_challenges=500),
        evaluate_feedforward_xor_response(loop_specs),
    ).run(run_key)
    ff_individual, ff_xor_response = ff_xor.response
    print("\n6. Feed-forward XOR PUF through the pipeline")
    print(f"  compact weight rows={rows}")
    print(f"  individual shape={ff_individual.shape}")
    print(f"  xor uniformity={uniformity(ff_xor_response):.3f}")


if __name__ == "__main__":
    main()
