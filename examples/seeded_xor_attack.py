"""Run a small seeded XOR attack through FunAttack_v2.

This example requires `evosax`.  The parameters are deliberately small so that
developers can smoke-test orchestration before launching larger scientific
runs.  Increase `nchall`, `n_generations`, and `dim` for real experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Attack.FunAttack_v2 import xor_attk


def main() -> None:
    """Run a reproducible toy XOR attack."""
    try:
        result = xor_attk(
            dim=(2, 32),
            nchall=512,
            threshold=0.5,
            n_generations=25,
            pop_size=8,
            seed=1337,
        )
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "This example requires the optional attack dependency `evosax`. "
            "Install it before running seeded attack examples."
        ) from exc
    print("overall accuracy:", result["overall_acc"])
    print("retry count:", result.get("retry_count"))
    print("seed:", result["seed"])


if __name__ == "__main__":
    main()
