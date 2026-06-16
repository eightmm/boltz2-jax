"""Pure JAX Pairformer blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from boltz_jax.models.attention import attention_pair_bias_forward
from boltz_jax.models.transition import transition_forward
from boltz_jax.models.triangle import triangle_multiplication_forward
from boltz_jax.models.triangle_attention import triangle_attention_forward

PairformerLayerParams = Mapping[str, Mapping[str, jnp.ndarray]]


def pairformer_layer_forward(
    params: PairformerLayerParams,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz PairformerLayer in eval mode without dropout."""

    z = z + triangle_multiplication_forward(
        params["tri_mul_out"], z, pair_mask, "outgoing", eps=eps
    )
    z = z + triangle_multiplication_forward(
        params["tri_mul_in"], z, pair_mask, "incoming", eps=eps
    )
    z = z + triangle_attention_forward(
        params["tri_att_start"], z, pair_mask, starting=True, eps=eps
    )
    z = z + triangle_attention_forward(
        params["tri_att_end"], z, pair_mask, starting=False, eps=eps
    )
    z = z + transition_forward(params["transition_z"], z, eps=eps)

    s_normed = _layer_norm(
        s.astype(jnp.float32),
        params["pre_norm_s"]["scale"],
        params["pre_norm_s"]["bias"],
        eps,
    )
    s = s.astype(jnp.float32) + attention_pair_bias_forward(
        params["attention"],
        s=s_normed,
        z=z.astype(jnp.float32),
        mask=mask.astype(jnp.float32),
        k_in=s_normed,
        eps=eps,
    )
    s = s + transition_forward(params["transition_s"], s, eps=eps)
    return s, z


def _layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale + bias
