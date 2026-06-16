"""Pure JAX MSA module for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from boltz_jax.models.transition import transition_forward
from boltz_jax.models.triangle import triangle_multiplication_forward
from boltz_jax.models.triangle_attention import triangle_attention_forward

Params = Mapping[str, object]


def msa_module_forward(
    params: Params,
    z: jnp.ndarray,
    emb: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    num_tokens: int = 33,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz MSAModule without stochastic MSA subsampling."""

    msa = jax.nn.one_hot(feats["msa"].astype(jnp.int32), num_tokens)
    m = jnp.concatenate(
        (
            msa,
            feats["has_deletion"][..., None],
            feats["deletion_value"][..., None],
            feats["msa_paired"][..., None],
        ),
        axis=-1,
    ).astype(jnp.float32)
    m = _linear(m, params["msa_proj"]["kernel"])
    m = m + _linear(emb, params["s_proj"]["kernel"])[:, None]

    token_mask = feats["token_pad_mask"].astype(jnp.float32)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    msa_mask = feats["msa_mask"].astype(jnp.float32)

    for layer in params["layers"]:
        z, m = msa_layer_forward(layer, z, m, token_mask, msa_mask, eps=eps)
    return z


def msa_layer_forward(
    params: Params,
    z: jnp.ndarray,
    m: jnp.ndarray,
    token_mask: jnp.ndarray,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz MSALayer in eval mode."""

    m = m + pair_weighted_averaging_forward(
        params["pair_weighted_averaging"], m, z, token_mask, eps=eps
    )
    m = m + transition_forward(params["msa_transition"], m, eps=eps)
    z = z + outer_product_mean_forward(params["outer_product_mean"], m, msa_mask, eps)
    z = pairformer_no_seq_layer_forward(params["pairformer_layer"], z, token_mask, eps)
    return z, m


def pair_weighted_averaging_forward(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    inf: float = 1e6,
) -> jnp.ndarray:
    """Run Boltz PairWeightedAveraging."""

    m = _layer_norm(m, params["norm_m"]["scale"], params["norm_m"]["bias"], eps)
    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    v = _linear(m, params["proj_m"]["kernel"])
    v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
    v = jnp.transpose(v, (0, 3, 1, 2, 4))
    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    b = b + (1.0 - mask[:, None]) * -inf
    w = jax.nn.softmax(b, axis=-1)
    g = jax.nn.sigmoid(_linear(m, params["proj_g"]["kernel"]))
    o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
    o = jnp.transpose(o, (0, 2, 3, 1, 4))
    o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
    return _linear(g * o, params["proj_o"]["kernel"])


def outer_product_mean_forward(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz OuterProductMean."""

    mask = mask[..., None].astype(m.dtype)
    m = _layer_norm(m, params["norm"]["scale"], params["norm"]["bias"], eps)
    a = _linear(m, params["proj_a"]["kernel"]) * mask
    b = _linear(m, params["proj_b"]["kernel"]) * mask
    pair_mask = mask[:, :, None, :] * mask[:, :, :, None]
    num_mask = jnp.maximum(jnp.sum(pair_mask, axis=1), 1.0)
    z = jnp.einsum("bsic,bsjd->bijcd", a.astype(jnp.float32), b.astype(jnp.float32))
    z = jnp.reshape(z, (*z.shape[:3], -1)) / num_mask
    return _linear(
        z.astype(m.dtype), params["proj_o"]["kernel"], params["proj_o"]["bias"]
    )


def pairformer_no_seq_layer_forward(
    params: Params,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz PairformerNoSeqLayer in eval mode."""

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
    return z


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
