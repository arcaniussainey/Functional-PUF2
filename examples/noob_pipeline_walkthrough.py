"""
Line-by-line beginner walkthrough for PUFs and the pipeline DSL.

Run from the repository root:
    python examples/05_noob_pipeline_walkthrough.py
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

from Pufs.pipeline import compose, evaluate_response, generate_challenges, generate_weights
from Pufs.statistics import uniformity


def main() -> None:
    """Build and evaluate the smallest useful Arbiter PUF experiment."""
    # A PRNG key is JAX's explicit source of randomness.  Reusing the same key
    # gives the same experiment, so your experiment is reproducible.
    rng = jax.random.PRNGKey(123)

    # A pipeline is a readable recipe.  This one says:
    #   1. create one 16-stage Arbiter PUF weight vector,
    #   2. create eight random challenges,
    #   3. evaluate the binary response bits.
    pipeline = compose(
        generate_weights(n_stages=16, k=1),
        generate_challenges(n_challenges=8), # For a good PUF, increasing # challenges should get us CLOSER to the ideal (like coin flips)
        evaluate_response(),
    )

    # Running the pipeline returns an immutable state object carrying the final
    # weight matrix, challenge matrix, response matrix, and advanced PRNG key.
    state = pipeline.run(rng)

    print("Pipeline recipe:")
    print(f"  {pipeline}")

    print("\nGenerated arrays:")
    print(f"  weights:    shape={state.weight.shape}; one row per Arbiter PUF")
    print(f"  challenges: shape={state.challenge.shape}; one row per challenge")
    print(f"  responses:  shape={state.response.shape}; values are 0/1 bits")

    print("\nActual challenge-response pairs:")
    for index, (challenge, response) in enumerate(zip(state.challenge, state.response)):
        challenge_text = " ".join(f"{int(bit):+d}" for bit in challenge)
        print(f"  CRP {index:02d}: [{challenge_text}] -> {int(response[0])}")

    # Uniformity is the simplest PUF quality metric: a good unbiased PUF should
    # produce about half zeros and half ones over many random challenges.
    response_uniformity = uniformity(state.response)
    print("\nMetric:")
    print(f"  uniformity={response_uniformity:.3f}; ideal is approximately 0.5")

    # JAX arrays are immutable.  This line demonstrates that response values can
    # be used with ordinary array operations without changing the pipeline state.
    ones = int(jnp.sum(state.response))
    print(f"  observed one-bits={ones} out of {state.response.size}")


if __name__ == "__main__":
    main()
