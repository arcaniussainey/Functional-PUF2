"""
Statistics example for a population of PUF devices.

Run from the repository root:
    python examples/09_statistics.py
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

from Pufs.FunctionalPuf import Arbiter
from Pufs.primitives import generate_challenges, noisy_get_response
from Pufs.statistics import (
    bit_aliasing,
    diffuseness,
    inter_distance,
    reliability,
    summary,
    uniformity,
)


def main() -> None:
    """Compute core PUF statistics for a reproducible device population."""
    rng = jax.random.PRNGKey(0)
    device_count = 30
    challenge_count = 500
    stage_count = 64

    split_keys = jax.random.split(rng, device_count + 2)
    challenge_key = split_keys[0]
    noise_key = split_keys[1]
    device_keys = split_keys[2:]
    challenges = generate_challenges(challenge_key, (challenge_count, stage_count))

    responses = np.stack(
        [
            np.asarray(Arbiter(key, (1, stage_count)).get_response(challenges)).reshape(-1)
            for key in device_keys
        ]
    )

    arbiter = Arbiter(device_keys[0], (1, stage_count))
    clean = np.asarray(arbiter.get_response(challenges)).reshape(-1)
    _next_key, noisy = noisy_get_response(noise_key, arbiter.weight, challenges, 0.5)
    flipped = np.asarray(arbiter.get_response(challenges.at[:, 0].multiply(-1))).reshape(-1)

    print("=== Scalar metrics ===")
    print(f"uniformity(device 0):     {uniformity(responses[0]):.4f}")
    print(f"inter-device distance:    {inter_distance(responses):.4f}")
    print(f"bit-aliasing mean:        {float(np.mean(bit_aliasing(responses))):.4f}")
    print(f"reliability(sigma=0.5):   {reliability(noisy, clean):.4f}")
    print(f"diffuseness(flip bit 0):  {diffuseness(clean, flipped):.4f}")
    print("\nsummary:")
    print(summary(responses))


if __name__ == "__main__":
    main()
