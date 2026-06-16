"""Pure JAX triangle multiplication blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp

TriangleDirection = Literal["outgoing", "incoming"]
TriangleMultiplicationParams = Mapping[str, Mapping[str, jnp.ndarray]]


def triangle_multiplication_forward(
    params: TriangleMultiplicationParams,
    x: jnp.ndarray,
    mask: jnp.ndarray,
    direction: TriangleDirection,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz triangle multiplication with mapped PyTorch parameters."""

    x = _layer_norm(x, params["norm_in"]["scale"], params["norm_in"]["bias"], eps)
    x_in = x
    projected = _linear(x, params["p_in"]["kernel"]) * jax.nn.sigmoid(
        _linear(x, params["g_in"]["kernel"])
    )
    projected = projected * mask[..., None]
    a, b = jnp.split(projected.astype(jnp.float32), 2, axis=-1)

    if direction == "outgoing":
        out = jnp.einsum("bikd,bjkd->bijd", a, b)
    elif direction == "incoming":
        out = jnp.einsum("bkid,bkjd->bijd", a, b)
    else:
        msg = f"Unsupported triangle multiplication direction: {direction!r}"
        raise ValueError(msg)

    out = _layer_norm(out, params["norm_out"]["scale"], params["norm_out"]["bias"], eps)
    out = _linear(out, params["p_out"]["kernel"])
    gate = jax.nn.sigmoid(_linear(x_in, params["g_out"]["kernel"]))
    return out * gate


def _linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return x @ kernel


def _layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale + bias
