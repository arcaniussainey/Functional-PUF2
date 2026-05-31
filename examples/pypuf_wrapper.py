"""
Optional PyPUF wrapper example.

Run from the repository root after installing pypuf:
    python examples/10_pypuf_wrapper.py
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

from Pufs.primitives import generate_challenges
from Pufs.pypuf_wrapper import PyPUF_Arbiter, PyPUF_XOR, PyPUFUnavailableError
from Pufs.statistics import uniformity


def main() -> None:
    """Demonstrate optional PyPUF adapters when the dependency is installed."""
    challenges = generate_challenges(jax.random.PRNGKey(0), (1_000, 64))
    try:
        arbiter = PyPUF_Arbiter(n=64, seed=42)
        xor_puf = PyPUF_XOR(n=64, k=4, seed=42)
    except PyPUFUnavailableError as exc:
        print(exc)
        print("Install pypuf to run this optional compatibility example.")
        return

    arbiter_response = arbiter.get_response(challenges)
    _individual, xor_response = xor_puf.get_response(challenges)
    print("1. PyPUF Arbiter")
    print(f"  response shape={arbiter_response.shape}")
    print(f"  uniformity={uniformity(np.asarray(arbiter_response).reshape(-1)):.3f}")
    print("\n2. PyPUF XOR")
    print(f"  response shape={xor_response.shape}")
    print(f"  uniformity={uniformity(np.asarray(xor_response).reshape(-1)):.3f}")


if __name__ == "__main__":
    main()
