"""Pure JAX Transition forward pass."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

TransitionParams = Mapping[str, Mapping[str, jnp.ndarray]]


def transition_forward(
    params: TransitionParams,
    x: jnp.ndarray,
    chunk_size: int | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run a Boltz Transition block using mapped PyTorch parameters."""

    x = _layer_norm(x, params["norm"]["scale"], params["norm"]["bias"], eps)
    fc1_kernel = params["fc1"]["kernel"]
    fc2_kernel = params["fc2"]["kernel"]
    fc3_kernel = params["fc3"]["kernel"]

    if chunk_size is None:
        hidden = jax.nn.silu(x @ fc1_kernel) * (x @ fc2_kernel)
        return hidden @ fc3_kernel

    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    out = jnp.zeros((*x.shape[:-1], fc3_kernel.shape[-1]), dtype=x.dtype)
    hidden_dim = fc3_kernel.shape[0]
    for start in range(0, hidden_dim, chunk_size):
        stop = min(start + chunk_size, hidden_dim)
        hidden = jax.nn.silu(x @ fc1_kernel[:, start:stop]) * (
            x @ fc2_kernel[:, start:stop]
        )
        out = out + hidden @ fc3_kernel[start:stop, :]
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
