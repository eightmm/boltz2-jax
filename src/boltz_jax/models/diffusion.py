"""Pure JAX diffusion score model assembly for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from boltz_jax.models.atom import (
    atom_attention_decoder_forward,
    atom_attention_encoder_forward,
    diffusion_transformer_forward,
)
from boltz_jax.models.conditioning import single_conditioning_forward

Params = Mapping[str, object]


def diffusion_score_model_forward(
    params: Params,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    diffusion_conditioning: Mapping[str, object],
    multiplicity: int = 1,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz DiffusionModule.forward using precomputed conditioning."""

    s, _ = single_conditioning_forward(
        params["single_conditioner"],
        times,
        jnp.repeat(s_trunk, multiplicity, axis=0),
        jnp.repeat(s_inputs, multiplicity, axis=0),
        eps=eps,
    )

    a, q_skip, c_skip = atom_attention_encoder_forward(
        params["atom_attention_encoder"],
        feats=feats,
        q=diffusion_conditioning["q"].astype(jnp.float32),
        c=diffusion_conditioning["c"].astype(jnp.float32),
        atom_enc_bias=diffusion_conditioning["atom_enc_bias"].astype(jnp.float32),
        to_keys=diffusion_conditioning["to_keys"],
        r=r_noisy,
        multiplicity=multiplicity,
        eps=eps,
    )

    s_to_a = params["s_to_a_linear"]
    a = a + _linear(
        _layer_norm(
            s,
            s_to_a["norm"]["scale"],
            s_to_a["norm"]["bias"],
            eps,
        ),
        s_to_a["linear"]["kernel"],
    )

    mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)
    a = diffusion_transformer_forward(
        params["token_transformer"],
        a=a,
        s=s,
        bias=diffusion_conditioning["token_trans_bias"].astype(jnp.float32),
        mask=mask.astype(jnp.float32),
        multiplicity=multiplicity,
        eps=eps,
    )
    a_norm = params["a_norm"]
    a = _layer_norm(a, a_norm["scale"], a_norm["bias"], eps)

    return atom_attention_decoder_forward(
        params["atom_attention_decoder"],
        a=a,
        q=q_skip,
        c=c_skip,
        atom_dec_bias=diffusion_conditioning["atom_dec_bias"].astype(jnp.float32),
        feats=feats,
        to_keys=diffusion_conditioning["to_keys"],
        multiplicity=multiplicity,
        eps=eps,
    )


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
