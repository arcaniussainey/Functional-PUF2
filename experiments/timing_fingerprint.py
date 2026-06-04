"""
Single-CRP response time fingerprinting.

The true physical PUF has a characteristic response generation time ``G_true``
that reflects its hardware speed.  An attacker running a software model
responds at ``G_attacker != G_true``.  If ``G_attacker < G_true`` the attacker
must add an artificial delay to impersonate the PUF; if ``G_attacker > G_true``
the attacker cannot hide (unless it pre-computes).

For each CRP exchange the verifier observes a round-trip time:

    R = 2*L + G + net_error + gen_error

where ``L`` is one-way network latency (known to all parties), ``G`` is the
true response time, ``net_error ~ N(0, sigma_net^2)``, and
``gen_error ~ N(0, sigma_gen^2)`` models hardware jitter.

The verifier knows ``G_true`` and ``L`` and computes its expected ``R``.  It
accepts if ``|R_observed - R_expected| <= margin``.

The attacker can add a deliberate delay ``d`` to fake ``G_attacker + d ~ G_true``,
but the delay itself is stochastic -- the attacker can only estimate ``G_true``
from the distribution of observed R values (it cannot see ``L`` or
``sigma_net`` perfectly).

Assumptions
-----------
* ``G_true`` is a hardware constant (nanoseconds), treated as a known-to-verifier
  parameter.  Typical PUF operating frequencies are a few MHz, so ``G_true``
  is on the order of microseconds.  We use milliseconds here to keep numbers
  legible; the relative values are what matter.
* The attacker learns an estimate of ``G_true`` by observing ``k_obs`` exchange
  timings.  It then sets its fake delay to ``max(0, G_est - G_attacker)``.
* ``net_error`` and ``gen_error`` are drawn fresh each exchange.
* The verifier applies a fixed margin and counts accept/reject over many
  simulated exchanges.

Output
------
Two subplots:
  1. Distribution of observed R values for the true PUF vs attacker, with the
     verifier's acceptance window marked.
  2. Accept rate as a function of the verifier's margin for both parties,
     as the attacker's observation count ``k_obs`` grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def _simulate_rtt(
    n: int,
    latency: float,
    gen_time: float,
    sigma_net: float,
    sigma_gen: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate ``n`` round-trip times for a single-CRP exchange.

    RTT = 2 * latency + gen_time + net_noise + gen_noise

    Args:
        n (int): number of simulated exchanges.
        latency (float): one-way network latency.
        gen_time (float): CRP generation time.
        sigma_net (float): network jitter std deviation.
        sigma_gen (float): generation time jitter std deviation.
        rng: numpy random generator.

    Returns:
        np.ndarray: shape (n,) RTT values.
    """
    net_noise = rng.normal(0.0, sigma_net, size=n)
    gen_noise = rng.normal(0.0, sigma_gen, size=n)
    return 2.0 * latency + gen_time + net_noise + gen_noise


def run_experiment(
    G_true: float = 1.0,
    G_attacker: float = 0.1,
    latency: float = 5.0,
    sigma_net: float = 0.5,
    sigma_gen: float = 0.2,
    margins: Optional[np.ndarray] = None,
    n_exchanges: int = 10_000,
    k_obs_values: tuple = (10, 50, 200, 1000),
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Simulate timing-based authentication and measure verifier discrimination.

    Args:
        G_true (float): true PUF generation time (arbitrary units, e.g. ms).
        G_attacker (float): attacker's generation time before any fake delay.
        latency (float): one-way network latency.
        sigma_net (float): network latency jitter std deviation.
        sigma_gen (float): generation time jitter std deviation.
        margins (np.ndarray, optional): array of margin values to sweep.
        n_exchanges (int): number of simulated authentication exchanges.
        k_obs_values (tuple): attacker observation counts to sweep for the
            timing-estimation phase.
        seed (int): RNG seed.
        save_path (str, optional): figure save path.

    Returns:
        Dict with accept rates per margin for true PUF and attacker variants.
    """
    rng = np.random.default_rng(seed)

    if margins is None:
        margins = np.linspace(0.1, 4.0, 80)

    R_expected = 2.0 * latency + G_true

    # True PUF RTT distribution
    rtt_true = _simulate_rtt(n_exchanges, latency, G_true, sigma_net, sigma_gen, rng)

    # Attacker without delay adjustment
    rtt_attacker_raw = _simulate_rtt(
        n_exchanges, latency, G_attacker, sigma_net, sigma_gen, rng
    )

    # Attacker with learned fake delay based on k_obs observation rounds
    attacker_variants: Dict[str, np.ndarray] = {}
    for k_obs in k_obs_values:
        obs_rtt = _simulate_rtt(k_obs, latency, G_true, sigma_net, sigma_gen, rng)
        # Attacker estimates R_expected as the mean of observed RTTs, then
        # backs out an estimate of G_true as R_est - 2*L
        R_est = float(np.mean(obs_rtt))
        G_est = R_est - 2.0 * latency
        fake_delay = max(0.0, G_est - G_attacker)
        rtt_attacker_delayed = _simulate_rtt(
            n_exchanges, latency, G_attacker + fake_delay, sigma_net, sigma_gen, rng
        )
        attacker_variants[f"attacker_k={k_obs}"] = rtt_attacker_delayed

    # Compute accept rates as a function of margin
    def accept_rate(rtts, margin):
        return float(np.mean(np.abs(rtts - R_expected) <= margin))

    true_accept = np.array([accept_rate(rtt_true, m) for m in margins])
    raw_attacker_accept = np.array([accept_rate(rtt_attacker_raw, m) for m in margins])
    variant_accepts = {
        name: np.array([accept_rate(rtts, m) for m in margins])
        for name, rtts in attacker_variants.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Exp 4: Response-time Fingerprinting\n"
        f"G_true={G_true}, G_attacker={G_attacker}, L={latency}, "
        f"σ_net={sigma_net}, σ_gen={sigma_gen}",
        fontsize=11,
    )

    ax = axes[0]
    ax.hist(rtt_true, bins=80, alpha=0.6, label="True PUF", color="black", density=True)
    ax.hist(rtt_attacker_raw, bins=80, alpha=0.5, label="Attacker (no delay)", color="firebrick", density=True)
    # Best attacker (most observations)
    best_k = max(k_obs_values)
    ax.hist(
        attacker_variants[f"attacker_k={best_k}"],
        bins=80, alpha=0.5, label=f"Attacker (k_obs={best_k})", color="steelblue", density=True,
    )
    ax.axvline(R_expected, color="green", linestyle="--", label=f"R_expected={R_expected:.1f}")
    ax.set_xlabel("Round-trip time")
    ax.set_ylabel("Density")
    ax.set_title("RTT distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(margins, true_accept, label="True PUF", color="black", linewidth=2)
    ax.plot(margins, raw_attacker_accept, label="Attacker (no delay, no obs)", color="firebrick", linestyle="--")
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(k_obs_values)))
    for (name, acc_curve), color in zip(variant_accepts.items(), colors):
        k = int(name.split("=")[1])
        ax.plot(margins, acc_curve, label=f"Attacker k_obs={k}", linestyle="-.", color=color)
    ax.set_xlabel("Verifier margin")
    ax.set_ylabel("Accept rate")
    ax.set_title("Accept rate vs margin")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Mark the smallest margin at which the true PUF accept rate first reaches 0.95.
    # true_accept is monotonically increasing, so searchsorted works directly.
    target_idx = np.searchsorted(true_accept, 0.95)
    if target_idx < len(margins):
        operating_margin = float(margins[target_idx])
        ax.axvline(operating_margin, color="green", linestyle=":", alpha=0.7,
                   label=f"95% true accept margin={operating_margin:.2f}")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[exp4] Figure saved to {save_path}")
    plt.show()

    return {
        "margins": margins.tolist(),
        "true_accept": true_accept.tolist(),
        "raw_attacker_accept": raw_attacker_accept.tolist(),
        "variant_accepts": {k: v.tolist() for k, v in variant_accepts.items()},
        "R_expected": R_expected,
        "G_true": G_true,
        "G_attacker": G_attacker,
    }


if __name__ == "__main__":
    run_experiment(
        G_true=1.0,
        G_attacker=0.1,
        latency=5.0,
        sigma_net=0.5,
        sigma_gen=0.2,
        k_obs_values=(10, 50, 200, 1000),
        save_path="experiments/results/exp4_timing.png",
    )
