"""
Statistical metrics and plotting helpers for PUF experiments.

The metrics operate on binary response matrices with values in ``{0, 1}``.
For population metrics, use shape ``(n_devices, n_challenges)``.  For a single
PUF, shape ``(n_challenges,)`` or ``(n_challenges, 1)`` is accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - environment dependent
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment dependent
    plt = None

ArrayLike = np.ndarray | Sequence[int] | Sequence[float]


def _as_binary_array(values: ArrayLike) -> np.ndarray:
    """Return ``values`` as a NumPy array suitable for binary metrics."""
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("response arrays must not be empty.")
    return array.astype(np.uint8)


def _as_response_matrix(values: ArrayLike) -> np.ndarray:
    """Return responses as shape ``(n_devices, n_challenges)``."""
    array = _as_binary_array(values)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array.reshape(array.shape[0], -1)
    raise ValueError("responses must have one or two dimensions.")


def _require_matplotlib():
    """Return pyplot or raise a clear optional-dependency error."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting helpers.")
    return plt


def _save_if_requested(fig, save_path: Optional[str]):
    """Save ``fig`` when ``save_path`` is provided, then return the figure."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def uniformity(responses: ArrayLike) -> float:
    """
    Return the fraction of one-responses for a single device.

    Ideal value: ``0.5`` for an unbiased response distribution.
    """
    return float(np.mean(_as_binary_array(responses).reshape(-1)))


def intra_distance(responses: ArrayLike, reference: Optional[ArrayLike] = None) -> float:
    """
    Return intra-device Hamming distance under repeated measurements.

    When ``reference`` is omitted, the first row of ``responses`` is used as the
    clean reference and all rows are compared against it.  Ideal value: ``0.0``.
    """
    matrix = _as_response_matrix(responses)
    ref = matrix[0] if reference is None else _as_binary_array(reference).reshape(-1)
    if matrix.shape[1] != ref.shape[0]:
        raise ValueError("reference length must match response length.")
    return float(np.mean(matrix != ref.reshape(1, -1)))


def inter_distance(responses: ArrayLike) -> float:
    """
    Return mean pairwise inter-device Hamming distance.

    Ideal value: ``0.5`` because two independent devices should disagree on
    half of a sufficiently large common challenge set.
    """
    matrix = _as_response_matrix(responses)
    if matrix.shape[0] < 2:
        return 0.0
    distances = [
        np.mean(matrix[left] != matrix[right])
        for left in range(matrix.shape[0])
        for right in range(left + 1, matrix.shape[0])
    ]
    return float(np.mean(distances))


def pairwise_inter_distances(responses: ArrayLike) -> np.ndarray:
    """Return all pairwise inter-device Hamming distances."""
    matrix = _as_response_matrix(responses)
    return np.asarray(
        [
            np.mean(matrix[left] != matrix[right])
            for left in range(matrix.shape[0])
            for right in range(left + 1, matrix.shape[0])
        ],
        dtype=float,
    )


def bit_aliasing(responses: ArrayLike) -> np.ndarray:
    """
    Return per-challenge fraction of devices that respond with ``1``.

    Ideal value: every challenge has aliasing close to ``0.5`` across devices.
    """
    return _as_response_matrix(responses).mean(axis=0)


def reliability(noisy_responses: ArrayLike, clean_responses: ArrayLike) -> float:
    """Return the fraction of noisy responses matching clean responses."""
    noisy = _as_binary_array(noisy_responses).reshape(-1)
    clean = _as_binary_array(clean_responses).reshape(-1)
    if noisy.shape != clean.shape:
        raise ValueError("noisy and clean responses must have matching shapes.")
    return float(np.mean(noisy == clean))


def diffuseness(responses_original: ArrayLike, responses_flipped: ArrayLike) -> float:
    """Return response-change rate after a one-bit challenge perturbation."""
    original = _as_binary_array(responses_original).reshape(-1)
    flipped = _as_binary_array(responses_flipped).reshape(-1)
    if original.shape != flipped.shape:
        raise ValueError("original and flipped responses must have matching shapes.")
    return float(np.mean(original != flipped))


def uniqueness(responses: ArrayLike) -> float:
    """Alias for ``inter_distance`` used in some PUF literature."""
    return inter_distance(responses)


def response_bias(responses: ArrayLike) -> float:
    """Return signed response bias, where ``0.0`` is perfectly balanced."""
    return uniformity(responses) - 0.5


def summary(responses: ArrayLike, noisy_responses: Optional[ArrayLike] = None) -> dict[str, float]:
    """Return a compact dictionary of common PUF quality metrics."""
    matrix = _as_response_matrix(responses)
    aliasing = bit_aliasing(matrix)
    result = {
        "uniformity": round(float(np.mean([uniformity(row) for row in matrix])), 4),
        "inter_distance": round(inter_distance(matrix), 4),
        "bit_aliasing_mean": round(float(np.mean(aliasing)), 4),
        "bit_aliasing_std": round(float(np.std(aliasing)), 4),
    }
    if noisy_responses is not None:
        noisy_matrix = _as_response_matrix(noisy_responses)
        if noisy_matrix.shape != matrix.shape:
            raise ValueError("noisy_responses must match responses shape.")
        reliabilities = [
            reliability(noisy_matrix[index], matrix[index])
            for index in range(matrix.shape[0])
        ]
        result["reliability_mean"] = round(float(np.mean(reliabilities)), 4)
        result["reliability_std"] = round(float(np.std(reliabilities)), 4)
    return result


def plot_uniformity_histogram(
    responses: ArrayLike,
    title: str = "Uniformity by device",
    save_path: Optional[str] = None,
):
    """Plot a histogram of per-device uniformity values."""
    pyplot = _require_matplotlib()
    matrix = _as_response_matrix(responses)
    values = [uniformity(row) for row in matrix]
    fig, ax = pyplot.subplots(figsize=(7, 4))
    ax.hist(values, bins=15)
    ax.axvline(0.5, linestyle="--", label="ideal = 0.5")
    ax.set_xlabel("fraction of one-responses")
    ax.set_ylabel("device count")
    ax.set_title(title)
    ax.legend()
    return _save_if_requested(fig, save_path)


def plot_inter_distance_histogram(
    responses: ArrayLike,
    title: str = "Inter-device Hamming distance",
    save_path: Optional[str] = None,
):
    """Plot a histogram of pairwise inter-device Hamming distances."""
    pyplot = _require_matplotlib()
    values = pairwise_inter_distances(responses)
    fig, ax = pyplot.subplots(figsize=(7, 4))
    ax.hist(values, bins=20)
    ax.axvline(0.5, linestyle="--", label="ideal = 0.5")
    ax.set_xlabel("pairwise Hamming distance")
    ax.set_ylabel("pair count")
    ax.set_title(title)
    ax.legend()
    return _save_if_requested(fig, save_path)


def plot_reliability_curve(
    noise_levels: Sequence[float],
    reliability_values: Sequence[float],
    title: str = "Reliability under noise",
    save_path: Optional[str] = None,
):
    """Plot reliability as a function of a noise or aging parameter."""
    pyplot = _require_matplotlib()
    fig, ax = pyplot.subplots(figsize=(7, 4))
    ax.plot(list(noise_levels), list(reliability_values), marker="o")
    ax.set_xlabel("noise / aging level")
    ax.set_ylabel("reliability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    return _save_if_requested(fig, save_path)


def plot_response_heatmap(
    responses: ArrayLike,
    title: str = "Response heatmap",
    save_path: Optional[str] = None,
):
    """Plot a device-by-challenge response heatmap."""
    pyplot = _require_matplotlib()
    matrix = _as_response_matrix(responses)
    fig, ax = pyplot.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xlabel("challenge index")
    ax.set_ylabel("device index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="response bit")
    return _save_if_requested(fig, save_path)


def plot_overlap_heatmap(
    responses: ArrayLike,
    title: str = "Pairwise response agreement",
    save_path: Optional[str] = None,
):
    """Plot pairwise response agreement between devices."""
    pyplot = _require_matplotlib()
    matrix = _as_response_matrix(responses)
    agreement = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    for left in range(matrix.shape[0]):
        for right in range(matrix.shape[0]):
            agreement[left, right] = np.mean(matrix[left] == matrix[right])
    fig, ax = pyplot.subplots(figsize=(6, 5))
    image = ax.imshow(agreement, vmin=0.0, vmax=1.0)
    ax.set_xlabel("device index")
    ax.set_ylabel("device index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="agreement")
    return _save_if_requested(fig, save_path)


def plot_bit_aliasing(
    responses: ArrayLike,
    title: str = "Bit aliasing by challenge",
    save_path: Optional[str] = None,
):
    """Plot per-challenge bit aliasing values."""
    pyplot = _require_matplotlib()
    values = bit_aliasing(responses)
    fig, ax = pyplot.subplots(figsize=(8, 4))
    ax.plot(np.arange(values.shape[0]), values)
    ax.axhline(0.5, linestyle="--", label="ideal = 0.5")
    ax.set_xlabel("challenge index")
    ax.set_ylabel("fraction one")
    ax.set_title(title)
    ax.legend()
    return _save_if_requested(fig, save_path)


def plot_challenge_response_distribution(
    challenges: ArrayLike,
    responses: ArrayLike,
    title: str = "Challenge-response distribution",
    save_path: Optional[str] = None,
):
    """Plot challenge Hamming weight against response bit."""
    pyplot = _require_matplotlib()
    challenge_array = np.asarray(challenges)
    response_array = _as_binary_array(responses).reshape(-1)
    hamming_weight = np.mean(challenge_array > 0, axis=1)
    fig, ax = pyplot.subplots(figsize=(7, 4))
    ax.scatter(hamming_weight, response_array, alpha=0.4)
    ax.set_xlabel("fraction of +1 challenge bits")
    ax.set_ylabel("response bit")
    ax.set_title(title)
    return _save_if_requested(fig, save_path)
