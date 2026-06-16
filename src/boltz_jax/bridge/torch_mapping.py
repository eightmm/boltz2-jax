"""Explicit PyTorch state_dict to JAX parameter mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

TransitionParams = dict[str, dict[str, jnp.ndarray]]
AttentionPairBiasParams = dict[str, dict[str, jnp.ndarray]]
TriangleMultiplicationParams = dict[str, dict[str, jnp.ndarray]]


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


def map_attention_pair_bias_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> AttentionPairBiasParams:
    """Map one Boltz AttentionPairBias v2 module to a JAX pytree."""

    required_keys = (
        f"{prefix}.proj_q.weight",
        f"{prefix}.proj_q.bias",
        f"{prefix}.proj_k.weight",
        f"{prefix}.proj_v.weight",
        f"{prefix}.proj_g.weight",
        f"{prefix}.proj_z.0.weight",
        f"{prefix}.proj_z.0.bias",
        f"{prefix}.proj_z.1.weight",
        f"{prefix}.proj_o.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required AttentionPairBias state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "proj_q": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.proj_q.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.proj_q.bias"]),
        },
        "proj_k": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_k.weight"])},
        "proj_v": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_v.weight"])},
        "proj_g": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_g.weight"])},
        "proj_z_norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.proj_z.0.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.proj_z.0.bias"]),
        },
        "proj_z": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_z.1.weight"])},
        "proj_o": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_o.weight"])},
    }


def map_triangle_multiplication_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> TriangleMultiplicationParams:
    """Map one Boltz TriangleMultiplication module to a JAX pytree."""

    required_keys = (
        f"{prefix}.norm_in.weight",
        f"{prefix}.norm_in.bias",
        f"{prefix}.p_in.weight",
        f"{prefix}.g_in.weight",
        f"{prefix}.norm_out.weight",
        f"{prefix}.norm_out.bias",
        f"{prefix}.p_out.weight",
        f"{prefix}.g_out.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required TriangleMultiplication state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "norm_in": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_in.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_in.bias"]),
        },
        "p_in": {"kernel": _linear_kernel(state_dict[f"{prefix}.p_in.weight"])},
        "g_in": {"kernel": _linear_kernel(state_dict[f"{prefix}.g_in.weight"])},
        "norm_out": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_out.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_out.bias"]),
        },
        "p_out": {"kernel": _linear_kernel(state_dict[f"{prefix}.p_out.weight"])},
        "g_out": {"kernel": _linear_kernel(state_dict[f"{prefix}.g_out.weight"])},
    }


def _linear_kernel(weight: Any) -> jnp.ndarray:
    """Convert PyTorch Linear.weight [out_dim, in_dim] to JAX [in_dim, out_dim]."""

    return _to_jax_array(weight).T


def _to_jax_array(value: Any) -> jnp.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)
