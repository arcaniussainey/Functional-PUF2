"""
Complex feed-forward XOR experiment built entirely with the pipeline DSL.

Run from the repository root:
    python examples/12_feedforward_aging_pipeline_experiment.py
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax

from Pufs.feedforward_core import required_xor_weight_rows
from Pufs.pipeline import (
    add_gaussian_noise,
    age,
    compose,
    evaluate_feedforward_xor_response,
    generate_challenges,
    generate_weights,
)
from Pufs.statistics import reliability, uniformity


def _build_pipeline(stage_count: int, challenge_count: int, loop_specs):
    """Create one reusable feed-forward XOR pipeline recipe."""
    required_rows = required_xor_weight_rows(loop_specs)
    return compose(
        generate_weights(n_stages=stage_count, k=required_rows),
        age(n_steps=40, sigma_per_step=0.01),
        add_gaussian_noise(sigma=0.05),
        generate_challenges(n_challenges=challenge_count),
        evaluate_feedforward_xor_response(loop_specs),
    )


def main() -> None:
    """Run deterministic feed-forward XOR pipelines under two seeds."""
    stage_count = 32
    challenge_count = 2_000
    loop_specs = (((2, 20),), ((4, 24), (8, 28)), ((1, 18),))
    pipeline = _build_pipeline(stage_count, challenge_count, loop_specs)

    first = pipeline.run(jax.random.PRNGKey(1), compiled=True)
    second = pipeline.run(jax.random.PRNGKey(1), compiled=True)
    third = pipeline.run(jax.random.PRNGKey(2), compiled=True)

    _first_individual, first_xor = first.response
    _second_individual, second_xor = second.response
    _third_individual, third_xor = third.response

    print("Feed-forward aging pipeline experiment")
    print(f"  pipeline={pipeline}")
    print(f"  first response shape={first_xor.shape}")
    print(f"  first uniformity={uniformity(first_xor):.3f}")
    print(f"  same-seed reproducibility={reliability(second_xor, first_xor):.3f}")
    print(f"  different-seed agreement={reliability(third_xor, first_xor):.3f}")


if __name__ == "__main__":
    main()
