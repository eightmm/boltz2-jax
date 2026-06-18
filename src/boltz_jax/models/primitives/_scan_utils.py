"""Helpers to run homogeneous layer stacks via ``jax.lax.scan``.

Mappers return per-layer parameters as a Python ``list`` of identical-structure
pytrees. ``stack_layer_params`` collapses that list into one pytree with a
leading layer axis so the stack body can be compiled once via ``lax.scan``
instead of being unrolled by XLA.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def stack_layer_params(layer_list: list) -> object:
    """Stack a list of identical-structure pytrees on a new leading axis.

    Each leaf ``x_i`` of the ``i``-th layer becomes ``stacked[i] == x_i``.
    """

    if not layer_list:
        msg = "stack_layer_params requires at least one layer"
        raise ValueError(msg)
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *layer_list)
