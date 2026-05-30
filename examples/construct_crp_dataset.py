"""Build a deterministic challenge-response dataset with OperationPipeline.

This is the smallest useful scientific experiment: create a PUF, generate a
fixed number of challenges, evaluate responses, and compute a response-bias
summary.  The same seed reproduces the same PUF, challenges, and responses.
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
    Xor,
    op_evaluate_xor_response,
    op_generate_challenges,
    op_pipeline,
)


def main() -> None:
    """Construct and summarise a deterministic CRP dataset."""
    stream = DeterministicKeyStream(seed=2026)
    puf = Xor(stream.next(), dim=(3, 64))

    pipeline = op_pipeline(
        op_generate_challenges(n_challenges=100_000),
        op_evaluate_xor_response(),
    )

    state = puf.run_operations(stream.next(), pipeline, jit=True)
    individual, xor_response = state.response

    print("challenge shape:", state.challenge.shape)
    print("individual response shape:", individual.shape)
    print("xor response shape:", xor_response.shape)
    print("xor response bias P(response=1):", float(jnp.mean(xor_response)))


if __name__ == "__main__":
    main()
