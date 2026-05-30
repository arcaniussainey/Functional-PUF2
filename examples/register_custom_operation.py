"""Register and use a custom operation-spec handler.

Future PUF variants such as feed-forward transforms, reliability masks, and
new aging/noise models should be added as immutable specs plus pure handlers.
This example registers a simple response mask operation without changing the
pipeline executor.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace

import jax.numpy as jnp

from Pufs.FunctionalPuf import (
    DeterministicKeyStream,
    OperationSpec,
    Xor,
    op_evaluate_xor_response,
    op_generate_challenges,
    op_pipeline,
    register_operation,
)


def _apply_mask_xor_response(state, spec):
    """Pure operation handler that masks the XOR response by a static period."""
    period = int(spec.get("period"))
    if state.response is None:
        raise ValueError("mask_xor_response requires a response to already exist.")

    individual, xor_response = state.response
    idx = jnp.arange(xor_response.shape[0])
    mask = (idx % period) == 0
    masked_xor = jnp.where(mask, xor_response, 0)
    return replace(state, response=(individual, masked_xor))


register_operation(
    "mask_xor_response",
    _apply_mask_xor_response,
    doc="Zero all XOR responses except every Nth challenge.",
    replace_existing=True,
)


def op_mask_xor_response(period: int) -> OperationSpec:
    """Factory for the custom operation spec."""
    return OperationSpec.create("mask_xor_response", period=int(period))


def main() -> None:
    """Run a pipeline containing a custom operation."""
    stream = DeterministicKeyStream(seed=505)
    puf = Xor(stream.next(), dim=(3, 32))

    pipeline = op_pipeline(
        op_generate_challenges(128),
        op_evaluate_xor_response(),
        op_mask_xor_response(period=4),
    )
    state = puf.run_operations(stream.next(), pipeline, jit=True)
    _, masked_response = state.response

    print("masked response shape:", masked_response.shape)
    print("non-zero response count:", int(jnp.count_nonzero(masked_response)))


if __name__ == "__main__":
    main()
