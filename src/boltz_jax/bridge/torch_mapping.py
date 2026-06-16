"""Explicit PyTorch state_dict to JAX parameter mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

TransitionParams = dict[str, dict[str, jnp.ndarray]]


def map_transition_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> TransitionParams:
    """Map one Boltz Transition module from PyTorch keys to a JAX pytree."""

    required_keys = (
        f"{prefix}.norm.weight",
        f"{prefix}.norm.bias",
        f"{prefix}.fc1.weight",
        f"{prefix}.fc2.weight",
        f"{prefix}.fc3.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required Transition state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm.bias"]),
        },
        "fc1": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc1.weight"])},
        "fc2": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc2.weight"])},
        "fc3": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc3.weight"])},
    }


def _linear_kernel(weight: Any) -> jnp.ndarray:
    """Convert PyTorch Linear.weight [out_dim, in_dim] to JAX [in_dim, out_dim]."""

    return _to_jax_array(weight).T


def _to_jax_array(value: Any) -> jnp.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)
