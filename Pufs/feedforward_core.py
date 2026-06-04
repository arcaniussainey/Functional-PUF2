"""
Shared feed-forward PUF kernels used by classes and operation specs.

This module deliberately has no dependency on ``BasePUF``.  Keeping the core
math separate lets ``Pufs.operations`` import the kernels without introducing a
cycle through the object-oriented PUF classes.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from Pufs.primitives import Challenge, Response, Weight, phi_from_challenges

LoopSpec = Tuple[int, int]
LoopSpecs = Tuple[LoopSpec, ...]
XorLoopSpecs = Tuple[LoopSpecs, ...]


def normalise_loops(loops: Iterable[Sequence[int]], n_stages: int) -> LoopSpecs:
    """Validate and freeze one component's feed-forward loop specification."""
    frozen = tuple((int(src), int(tgt)) for src, tgt in loops)
    for src, tgt in frozen:
        if not 0 <= src < tgt < n_stages:
            raise ValueError(
                "Feed-forward loops must satisfy 0 <= src < tgt < n_stages; "
                f"got {(src, tgt)} for n_stages={n_stages}."
            )
    return frozen


def normalise_xor_loop_specs(
    loop_specs: Iterable[Iterable[Sequence[int]]],
    n_stages: int,
) -> XorLoopSpecs:
    """Validate and freeze loop specifications for a feed-forward XOR PUF."""
    frozen = tuple(normalise_loops(component, n_stages) for component in loop_specs)
    if not frozen:
        raise ValueError("At least one component loop specification is required.")
    return frozen


def validate_non_cascading(loops: LoopSpecs) -> None:
    """
    Validate the non-cascading topology required by symbolic FF variants.

    Loops are applied in ascending ``src`` order.  Any earlier injection target
    must be after every later loop's observed prefix; otherwise the later FF bit
    depends on a previously modified challenge position.
    """
    ordered = tuple(sorted(loops, key=lambda item: item[0]))
    for left_idx, left_loop in enumerate(ordered):
        for right_loop in ordered[left_idx + 1 :]:
            if left_loop[1] <= right_loop[0]:
                raise ValueError(
                    "Cascading feed-forward topology requires FF_Arbiter; "
                    f"loop {left_loop} modifies a later loop prefix {right_loop}."
                )


def required_weight_rows(loops: LoopSpecs) -> int:
    """Return rows needed for one feed-forward component weight matrix."""
    return 1 + len(loops)


def required_xor_weight_rows(loop_specs: XorLoopSpecs) -> int:
    """Return rows needed for all components in a feed-forward XOR PUF."""
    return sum(required_weight_rows(component) for component in loop_specs)


def suffix_product(challenge: Challenge) -> jax.Array:
    """Return right-to-left cumulative products for ``{-1, +1}`` challenges."""
    return jnp.cumprod(challenge[:, ::-1], axis=1)[:, ::-1].astype(jnp.float32)


def feedforward_response_from_weight(
    weight: Weight,
    challenge: Challenge,
    loops: LoopSpecs,
) -> Response:
    """
    Evaluate a feed-forward Arbiter PUF from a compact weight matrix.

    ``weight[0]`` is the main arbiter.  ``weight[i + 1]`` is the padded
    intermediate-arbiter weight for ``loops[i]``; only the prefix through the
    loop's ``src`` stage is read.
    """
    expected_rows = required_weight_rows(loops)
    if weight.shape[0] < expected_rows:
        raise ValueError(
            f"Feed-forward response needs at least {expected_rows} weight rows; "
            f"got {weight.shape[0]}."
        )

    effective = challenge.astype(jnp.float32)
    for loop_index, (src, tgt) in enumerate(sorted(loops, key=lambda item: item[0])):
        prefix_phi = phi_from_challenges(effective[:, : src + 1])
        loop_weight = weight[loop_index + 1, : src + 2]
        ff_delta = prefix_phi @ loop_weight
        ff_bit = jnp.where(ff_delta > 0, 1.0, -1.0).astype(effective.dtype)
        effective = effective.at[:, tgt].multiply(ff_bit)

    main_phi = phi_from_challenges(effective)
    delta = main_phi @ weight[0]
    return ((jnp.sign(delta) + 1) / 2).astype(jnp.uint8)[:, None]


def feedforward_xor_response_from_weight(
    weight: Weight,
    challenge: Challenge,
    loop_specs: XorLoopSpecs,
) -> tuple[Response, Response]:
    """
    Evaluate a feed-forward XOR PUF from a flattened component weight matrix."""
    expected_rows = required_xor_weight_rows(loop_specs)
    if weight.shape[0] < expected_rows:
        raise ValueError(
            f"Feed-forward XOR response needs at least {expected_rows} weight rows; "
            f"got {weight.shape[0]}."
        )

    offset = 0
    component_responses = []
    for loops in loop_specs:
        row_count = required_weight_rows(loops)
        component_weight = weight[offset : offset + row_count]
        response = feedforward_response_from_weight(component_weight, challenge, loops)
        component_responses.append(response[:, 0])
        offset += row_count

    individual = jnp.stack(component_responses, axis=1).astype(jnp.uint8)
    xor_response = lax.reduce(individual, jnp.uint8(0), lax.bitwise_xor, (1,))
    return individual, xor_response


def generate_weight_np(rng: np.random.Generator, n_stages: int) -> np.ndarray:
    """Generate one paper-correct additive-delay-model weight vector with NumPy."""
    delays = (rng.standard_normal((4, n_stages)) + 500.0) * 4.0
    first_delta = delays[0] - delays[1]
    second_delta = delays[2] - delays[3]
    shifted = np.concatenate([[0.0], (first_delta + second_delta) / 2.0])
    subtracted = np.concatenate([(first_delta - second_delta) / 2.0, [0.0]])
    weight = shifted + subtracted
    weight[0] = (first_delta[0] - second_delta[0]) / 2.0
    return weight.astype(np.float32)


def padded_prefix_weight(rng: np.random.Generator, prefix_len: int, n_stages: int) -> np.ndarray:
    """Generate a prefix FF weight and pad it to the full stage count."""
    if prefix_len > n_stages:
        raise ValueError("prefix_len cannot exceed n_stages.")
    prefix_weight = generate_weight_np(rng, prefix_len)
    padded = np.zeros((n_stages + 1,), dtype=np.float32)
    padded[: prefix_len + 1] = prefix_weight
    return padded
