"""
Aging-aware PUF examples.

Run from the repository root:
    python examples/07_aging_puf.py
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

from Pufs.aging import ArbiterPUF_Aging, XorPUF_Aging
from Pufs.pipeline import age, compose, evaluate_response, use_challenges
from Pufs.primitives import generate_challenges
from Pufs.FunctionalPuf import Arbiter
from Pufs.statistics import reliability


def main() -> None:
    """Demonstrate explicit aging histories and pipeline aging."""
    rng = jax.random.PRNGKey(7)
    rng, puf_key, challenge_key, age_key_a, age_key_b = jax.random.split(rng, 5)

    challenges = generate_challenges(challenge_key, (1_000, 64))
    arbiter = ArbiterPUF_Aging(puf_key, stages=64)
    arbiter.add_aging(age_key_a, n_steps=100, sigma_per_step=0.01)
    arbiter.add_aging(age_key_b, n_steps=100, model="exponential", base_sigma=0.05)

    fresh = arbiter.ref(0, label="fresh")
    latest = arbiter.ref(-1, label="latest")
    fresh_response = fresh.get_response(challenges).reshape(-1)
    aged_response = latest.get_response(challenges).reshape(-1)

    print("1. Arbiter PUF aging history")
    print(f"  {arbiter}")
    print(f"  {fresh}")
    print(f"  {latest}")
    print(f"  fresh/latest reliability={reliability(aged_response, fresh_response):.3f}")

    all_responses = arbiter.response_all_ages(challenges[:200])
    agreements = jnp.mean(all_responses == all_responses[0:1], axis=(1, 2))
    print("\n2. Selected aging curve samples")
    for index in (0, 50, 100, 150, 200):
        print(f"  age[{index:03d}] agreement with fresh={float(agreements[index]):.3f}")

    rng, xor_key, xor_age_key = jax.random.split(rng, 3)
    xor_puf = XorPUF_Aging(xor_key, k=3, stages=64)
    xor_puf.add_aging(xor_age_key, n_steps=150, sigma_per_step=0.015)
    fresh_individual, fresh_xor = xor_puf.get_response(challenges, age_idx=0)
    aged_individual, aged_xor = xor_puf.get_response(challenges, age_idx=-1)

    print("\n3. XOR PUF with aging")
    print(f"  individual shape={fresh_individual.shape}")
    print(f"  xor shape={fresh_xor.shape}")
    arbiter_zero_reliability = reliability(aged_individual[:, 0], fresh_individual[:, 0])
    print(f"  arbiter-0 reliability={arbiter_zero_reliability:.3f}")
    print(f"  xor reliability={reliability(aged_xor, fresh_xor):.3f}")

    rng, standard_key, pipeline_key = jax.random.split(rng, 3)
    standard = Arbiter(standard_key, (1, 64))
    pipeline_state = standard.run_pipeline(
        pipeline_key,
        compose(
            age(n_steps=500, sigma_per_step=0.01),
            use_challenges(challenges),
            evaluate_response(),
        ),
    )
    clean = standard.get_response(challenges)
    print("\n4. Pipeline aging on a standard Arbiter")
    print(f"  clean/aged reliability={reliability(pipeline_state.response, clean):.3f}")


if __name__ == "__main__":
    main()
