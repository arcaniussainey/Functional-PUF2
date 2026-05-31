"""
Feed-forward PUF examples.

Run from the repository root:
    python examples/08_feedforward_puf.py
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

from Pufs.feedforward import FF_Arbiter, FF_Arbiter_Expression, FF_Arbiter_Symbolic, FF_XOR
from Pufs.primitives import generate_challenges
from Pufs.statistics import uniformity


def main() -> None:
    """Compare feed-forward implementations and XOR composition."""
    challenge_key = jax.random.PRNGKey(99)
    challenges = generate_challenges(challenge_key, (1_000, 16))
    loops = [(2, 12)]

    classical = FF_Arbiter(0, 16, loops)
    symbolic = FF_Arbiter_Symbolic(0, 16, loops)
    expression = FF_Arbiter_Expression(0, 16, loops)

    classical_response = np.asarray(classical.get_response(challenges)).reshape(-1)
    symbolic_response = np.asarray(symbolic.get_response(challenges)).reshape(-1)
    expression_response = np.asarray(expression.get_response(challenges)).reshape(-1)

    print("1. Single-loop feed-forward Arbiter")
    print(f"  classical == symbolic:   {np.array_equal(classical_response, symbolic_response)}")
    print(f"  classical == expression: {np.array_equal(classical_response, expression_response)}")
    print(f"  uniformity={uniformity(classical_response):.3f}")

    print("\n2. Cascading topology validation")
    cascading = [(1, 3), (5, 9)]
    FF_Arbiter(0, 16, cascading)
    try:
        FF_Arbiter_Symbolic(0, 16, cascading)
        print("  symbolic accepted cascading topology unexpectedly")
    except ValueError:
        print("  symbolic correctly rejected cascading topology")

    loop_specs = [[(2, 12)], [(1, 10)], [(3, 14)]]
    xor_puf = FF_XOR(0, 16, loop_specs)
    individual, xor_response = xor_puf.get_response(challenges)
    print("\n3. Feed-forward XOR PUF")
    print(f"  individual shape={individual.shape}")
    print(f"  xor shape={xor_response.shape}")
    print(f"  xor uniformity={uniformity(xor_response):.3f}")

    tiny_challenges = challenges[:200]
    start = time.perf_counter()
    _ = classical.get_response(tiny_challenges)
    elapsed = time.perf_counter() - start
    print("\n4. Small performance smoke test")
    print(f"  200 challenge evaluations took {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()
