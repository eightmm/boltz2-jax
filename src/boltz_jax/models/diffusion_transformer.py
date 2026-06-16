"""Pure JAX diffusion transformer blocks for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

Params = Mapping[str, object]


def adaln_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz AdaLN with mapped PyTorch parameters."""

    a = _layer_norm_no_affine(a, eps)
    s = _layer_norm_scale(s, params["s_norm"]["scale"], eps)
    return jax.nn.sigmoid(
        _linear(s, params["s_scale"]["kernel"], params["s_scale"]["bias"])
    ) * a + _linear(s, params["s_bias"]["kernel"])


def conditioned_transition_block_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz ConditionedTransitionBlock with mapped PyTorch parameters."""

    a = adaln_forward(params["adaln"], a, s, eps)
    swish_x, swish_gate = jnp.split(_linear(a, params["swish_gate"]["kernel"]), 2, -1)
    b = (jax.nn.silu(swish_gate) * swish_x) * _linear(
        a, params["a_to_b"]["kernel"]
    )
    out = _linear(b, params["b_to_a"]["kernel"])
    gate = jax.nn.sigmoid(
        _linear(
            s,
            params["output_projection"]["kernel"],
            params["output_projection"]["bias"],
        )
    )
    return gate * out


def diffusion_transformer_layer_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
    inf: float = 1e6,
) -> jnp.ndarray:
    """Run one Boltz DiffusionTransformerLayer with pair-bias attention."""

    b = adaln_forward(params["adaln"], a, s, eps)
    k_in = b
    if to_keys is not None:
        k_in = to_keys(b)
        mask = jnp.squeeze(to_keys(mask[..., None]), axis=-1)

    b = _attention_pair_bias_no_proj_z_forward(
        params["pair_bias_attn"],
        s=b,
        bias=bias,
        mask=mask,
        k_in=k_in,
        multiplicity=multiplicity,
        inf=inf,
    )
    b = jax.nn.sigmoid(
        _linear(
            s,
            params["output_projection"]["kernel"],
            params["output_projection"]["bias"],
        )
    ) * b

    a = a + b
    a = a + conditioned_transition_block_forward(params["transition"], a, s, eps)
    if "post_lnorm" in params:
        post_lnorm = params["post_lnorm"]
        a = _layer_norm(a, post_lnorm["scale"], post_lnorm["bias"], eps)
    return a


def _attention_pair_bias_no_proj_z_forward(
    params: Params,
    s: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    k_in: jnp.ndarray,
    multiplicity: int,
    inf: float,
) -> jnp.ndarray:
    batch, _, c_s = s.shape
    num_heads = bias.shape[-1]
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


def _layer_norm_scale(x: jnp.ndarray, scale: jnp.ndarray, eps: float) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * scale


def _layer_norm_no_affine(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps)
