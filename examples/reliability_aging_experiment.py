"""Measure response stability after simulated aging/noise.

The experiment uses the same challenge set against the original PUF and an
aged copy produced through the operation-spec pipeline.  This is a compact
pattern for reliability studies: deterministic setup, explicit perturbation,
and scalar reproducibility metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from Pufs.FunctionalPuf import (
    DeterministicKeyStream,
    OperationState,
    Xor,
    generate_challenges,
    op_age,
    op_evaluate_xor_response,
    op_pipeline,
    xor_get_response,
)


def main() -> None:
    """Run a small aging reliability experiment."""
    stream = DeterministicKeyStream(seed=77)
    puf = Xor(stream.next(), dim=(4, 64))
    challenges = generate_challenges(stream.next(), (5_000, puf.dim[1]))

    _, original_response = xor_get_response(puf.weight, challenges)

    aging_pipeline = op_pipeline(
        op_age(n_steps=20, sigma_per_step=0.005, model="additive"),
        op_evaluate_xor_response(),
    )
    aged_state = aging_pipeline.run(
        OperationState(rng=stream.next(), weight=puf.weight, challenge=challenges),
        jit=True,
    )
    _, aged_response = aged_state.response

    reliability = jnp.equal(original_response.flatten(), aged_response.flatten()).mean()
    print("original weight shape:", puf.weight.shape)
    print("aged weight shape:", aged_state.weight.shape)
    print("reliability after aging:", float(reliability))


if __name__ == "__main__":
    main()
