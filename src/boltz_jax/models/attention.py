"""Pure JAX attention blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

AttentionPairBiasParams = Mapping[str, Mapping[str, jnp.ndarray]]


def attention_pair_bias_forward(
    params: AttentionPairBiasParams,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    k_in: jnp.ndarray | None = None,
    multiplicity: int = 1,
    inf: float = 1e6,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz AttentionPairBias v2 with mapped PyTorch parameters."""

    if k_in is None:
        k_in = s
    batch, _, c_s = s.shape
    num_heads = params["proj_z"]["kernel"].shape[-1]
    head_dim = c_s // num_heads

    q = _linear(s, params["proj_q"]["kernel"], params["proj_q"]["bias"]).reshape(
        batch, -1, num_heads, head_dim
    )
    k = _linear(k_in, params["proj_k"]["kernel"]).reshape(
        batch, -1, num_heads, head_dim
    )
    v = _linear(k_in, params["proj_v"]["kernel"]).reshape(
        batch, -1, num_heads, head_dim
    )

    bias = _layer_norm(
        z,
        params["proj_z_norm"]["scale"],
        params["proj_z_norm"]["bias"],
        eps,
    )
    bias = _linear(bias, params["proj_z"]["kernel"])
    bias = jnp.transpose(bias, (0, 3, 1, 2))
    bias = jnp.repeat(bias, multiplicity, axis=0)

    g = jax.nn.sigmoid(_linear(s, params["proj_g"]["kernel"]))
    attn = jnp.einsum("bihd,bjhd->bhij", q.astype(jnp.float32), k.astype(jnp.float32))
    attn = attn / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
    attn = attn + bias.astype(jnp.float32)
    attn = attn + (1.0 - mask[:, None, None].astype(jnp.float32)) * -inf
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.einsum("bhij,bjhd->bihd", attn, v.astype(jnp.float32)).astype(v.dtype)
    out = out.reshape(batch, -1, c_s)
    return _linear(g * out, params["proj_o"]["kernel"])


def _linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    out = x @ kernel
    if bias is not None:
        out = out + bias
    return out


def _layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale + bias
