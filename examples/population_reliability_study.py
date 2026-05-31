"""
More complex population reliability study using aging histories.

Run from the repository root:
    python examples/11_population_reliability_study.py
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

from Pufs.aging import ArbiterPUF_Aging
from Pufs.primitives import generate_challenges
from Pufs.statistics import inter_distance, reliability, summary


def main() -> None:
    """Simulate a small population and measure uniqueness versus aging."""
    rng = jax.random.PRNGKey(2026)
    device_count = 12
    stage_count = 64
    challenge_count = 1_000
    age_steps = 80

    keys = jax.random.split(rng, device_count * 2 + 1)
    challenge_key = keys[0]
    puf_keys = keys[1 : device_count + 1]
    aging_keys = keys[device_count + 1 :]
    challenges = generate_challenges(challenge_key, (challenge_count, stage_count))

    fresh_rows = []
    aged_rows = []
    for puf_key, aging_key in zip(puf_keys, aging_keys):
        puf = ArbiterPUF_Aging(puf_key, stages=stage_count)
        puf.add_aging(aging_key, n_steps=age_steps, sigma_per_step=0.015)
        fresh_rows.append(np.asarray(puf.get_response(challenges, age_idx=0)).reshape(-1))
        aged_rows.append(np.asarray(puf.get_response(challenges, age_idx=-1)).reshape(-1))

    fresh = np.stack(fresh_rows)
    aged = np.stack(aged_rows)
    per_device_reliability = [
        reliability(aged[index], fresh[index]) for index in range(device_count)
    ]

    print("Population reliability study")
    print(f"  devices={device_count}")
    print(f"  challenges={challenge_count}")
    print(f"  aging steps={age_steps}")
    print(f"  fresh inter-distance={inter_distance(fresh):.3f}")
    print(f"  aged inter-distance={inter_distance(aged):.3f}")
    print(f"  mean fresh/aged reliability={float(np.mean(per_device_reliability)):.3f}")
    print("\nFresh population summary:")
    print(summary(fresh))
    print("\nAged population summary:")
    print(summary(aged))


if __name__ == "__main__":
    main()
