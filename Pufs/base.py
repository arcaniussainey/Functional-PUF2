"""
Pufs/base.py
------------
Abstract base class for all PUF implementations.

BasePUF provides the shared pytree registration, weight storage, and
the bridge between the pipeline DSL (Pufs/pipeline.py) and the
concrete response logic implemented by each subclass.

Extension points
----------------
To add a new PUF variant (FeedForward, Aging, etc.):
  1. Subclass BasePUF.
  2. Implement _compute_response(weight, challenge) -> Response.
  3. Optionally override get_delta_response() for variant-specific delay models.
  4. Call super().__init__(rng, dim) -- weight generation is automatic.

The pipeline DSL (compose, generate_weights, add_gaussian_noise, age, ...)
is importable from Pufs.pipeline and is independent of this class.  BasePUF
exposes run_pipeline() as a convenience bridge so a PUF can execute a
pipeline seeded from its own internal state.
"""

from __future__ import annotations

import abc
from typing import Tuple

from jax.tree_util import register_pytree_node_class

from Pufs.primitives import (
    PRNGKey, Weight, Challenge, Response, Delta,
    generate_weights,
    get_delta_response,
)
from Pufs.pipeline import PUFPipeline, PipelineState
from Pufs.operations import OperationPipeline, OperationState


@register_pytree_node_class
class BasePUF(abc.ABC):
    """
    Abstract base class for all PUF implementations.

    Subclasses must implement _compute_response() which defines the
    response logic for the specific PUF variant (Arbiter, XOR, FeedForward, etc.).

    The rng key provided to the constructor is used to generate weights.
    This process is deterministic -- given the same rng you will get the
    same weights.

    Note: tree_unflatten reconstructs using the RNG. Manually changing
    weights is not recommended.

    The dim init param should be a tuple describing the shape of the
    weights matrix, e.g. (1, 64) for Arbiter or (3, 64) for a 3-XOR PUF.
    """

    def __init__(self, rng: PRNGKey, dim: Tuple[int, int]) -> None:
        """Initialise weights deterministically from *rng* and *dim*."""
        self.rng    = rng
        self.dim    = dim
        self.weight = generate_weights(rng, dim)

    # -- pytree registration ------------------------------------------------

    def tree_flatten(self) -> Tuple[Tuple[Weight], Tuple[PRNGKey, Tuple[int, int]]]:
        """Return (children, aux_data) for JAX pytree serialisation.

        The weight is a dynamic child, not auxiliary metadata.  This preserves
        learned, perturbed, or manually assigned weights through JAX tree
        transforms instead of reconstructing from the constructor RNG.
        """
        children = (self.weight,)
        aux_data = (self.rng, self.dim)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[PRNGKey, Tuple[int, int]],
        children: Tuple[Weight],
    ) -> "BasePUF":
        """Reconstruct a PUF instance from pytree aux_data and children."""
        rng, dim = aux_data
        (weight,) = children
        obj = cls.__new__(cls)
        obj.rng = rng
        obj.dim = dim
        obj.weight = weight
        return obj

    # -- abstract interface -------------------------------------------------

    @abc.abstractmethod
    def _compute_response(self, weight: Weight, challenge: Challenge) -> Response:
        """
        Compute the PUF response given a weight matrix and a challenge matrix.

        Subclasses implement this to define their specific response logic
        (single arbiter sign, XOR reduce, feed-forward injection, etc.).

        Args:
            weight (Weight): weight matrix, shape (k, n)
            challenge (Challenge): challenge matrix, shape (N, n)

        Returns:
            Response: binary response; shape and structure vary by subclass
        """

    # -- response interface -------------------------------------------------

    def get_response(self, challenge: Challenge) -> Response:
        """
        Evaluate the PUF on a batch of challenges using the stored weight.

        Args:
            challenge (Challenge): challenge matrix, shape (N, n)

        Returns:
            Response: binary responses
        """
        return self._compute_response(self.weight, challenge)

    def get_delta_response(self, challenge: Challenge) -> Delta:
        """
        Raw (unthresholded) delay differences for each challenge.

        Subclasses may override this for variant-specific delay models.

        Args:
            challenge (Challenge): challenge matrix, shape (N, n)

        Returns:
            Delta: shape (N, k), dtype float32
        """
        return get_delta_response(self.weight, challenge)

    # -- pipeline bridge ----------------------------------------------------

    def run_pipeline(self, rng: PRNGKey, pipeline: PUFPipeline) -> PipelineState:
        """
        Execute a pipeline, seeding the initial state with this PUF's
        stored weight so pipeline steps that modify weight (noise, aging)
        operate on the correct starting point.

        Example
        -------
        >>> from Pufs.pipeline import compose, add_gaussian_noise
        >>> from Pufs.pipeline import generate_challenges, evaluate_response
        >>> noisy_pipeline = compose(
        ...     add_gaussian_noise(sigma=0.5),
        ...     generate_challenges(n_challenges=1000),
        ...     evaluate_response(),
        ... )
        >>> state = puf.run_pipeline(rng, noisy_pipeline)
        >>> responses = state.response     # shape (1000, 1)

        Args:
            rng (PRNGKey): PRNG key for stochastic pipeline steps
            pipeline (PUFPipeline): pipeline to execute

        Returns:
            PipelineState: final state after all pipeline steps
        """
        initial = PipelineState(rng=rng, weight=self.weight)
        return pipeline.run_from(initial)

    def run_operations(
        self,
        rng: PRNGKey,
        pipeline: OperationPipeline,
        *,
        jit: bool = True,
    ) -> OperationState:
        """
        Execute an immutable OperationPipeline seeded with this PUF's weight.

        This is the preferred bridge for new code.  OperationPipeline contains
        static OperationSpec values only, so it can lower to a cached JAX
        executor while runtime arrays remain in OperationState.

        Args:
            rng (PRNGKey): PRNG key for stochastic operation specs.
            pipeline (OperationPipeline): immutable spec sequence to execute.
            jit (bool): whether to use the cached JIT-lowered executor.

        Returns:
            OperationState: final state after all operations have run.
        """
        initial = OperationState(rng=rng, weight=self.weight)
        return pipeline.run(initial, jit=jit)

    # -- utility ------------------------------------------------------------

    def clone(self) -> "BasePUF":
        """
        Return a new PUF instance with identical weights constructed
        from the same rng.  The clone is an independent object.
        """
        return type(self)(self.rng, self.dim)

    def __call__(self, challenge: Challenge) -> Response:
        """Shorthand for get_response(challenge)."""
        return self.get_response(challenge)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"{type(self).__name__}(dim={self.dim})"
