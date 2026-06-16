"""Pure JAX triangle attention blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

TriangleAttentionParams = Mapping[
    str, Mapping[str, jnp.ndarray | Mapping[str, jnp.ndarray]]
]


def triangle_attention_forward(
    params: TriangleAttentionParams,
    x: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    starting: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz TriangleAttention starting or ending node."""

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = _layer_norm(
        x,
        params["layer_norm"]["scale"],
        params["layer_norm"]["bias"],
        eps,
    )
    mask = mask[..., :, None, None, :]
    mask_bias = inf * (mask - 1.0)

    triangle_bias = _linear(x, params["linear"]["kernel"])
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))
    triangle_bias = jnp.expand_dims(triangle_bias, axis=1)

    x = _attention(
        params["mha"],
        q_x=x,
        kv_x=x,
        tri_bias=triangle_bias,
        mask_bias=mask_bias,
    )

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
    return x


def _attention(
    params: Mapping[str, Mapping[str, jnp.ndarray]],
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    tri_bias: jnp.ndarray,
    mask_bias: jnp.ndarray,
) -> jnp.ndarray:
    no_heads = tri_bias.shape[2]
    c_hidden = params["linear_g"]["kernel"].shape[-1] // no_heads

    q = _linear(q_x, params["linear_q"]["kernel"])
    k = _linear(kv_x, params["linear_k"]["kernel"])
    v = _linear(kv_x, params["linear_v"]["kernel"])

    q = q.reshape(q.shape[:-1] + (no_heads, c_hidden))
    k = k.reshape(k.shape[:-1] + (no_heads, c_hidden))
    v = v.reshape(v.shape[:-1] + (no_heads, c_hidden))
    q = jnp.swapaxes(q, -2, -3) / jnp.sqrt(jnp.asarray(c_hidden, dtype=jnp.float32))
    k = jnp.swapaxes(k, -2, -3)
    v = jnp.swapaxes(v, -2, -3)

    scores = jnp.matmul(
        q.astype(jnp.float32),
        jnp.swapaxes(k.astype(jnp.float32), -1, -2),
    )
    scores = scores + mask_bias.astype(jnp.float32) + tri_bias.astype(jnp.float32)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, v.astype(jnp.float32)).astype(v.dtype)
    out = jnp.swapaxes(out, -2, -3)

    gate = jax.nn.sigmoid(_linear(q_x, params["linear_g"]["kernel"]))
    gate = gate.reshape(gate.shape[:-1] + (no_heads, c_hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (c_hidden * no_heads,))
    return _linear(out, params["linear_o"]["kernel"])


def _linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, kernel, precision=jax.lax.Precision.HIGHEST)


def _layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale + bias
