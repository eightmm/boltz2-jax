"""Pure JAX InputEmbedder for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from boltz_jax.models.atom import (
    atom_attention_encoder_forward,
    get_indexing_matrix,
    single_to_keys,
)
from boltz_jax.models.diffusion_conditioning import atom_encoder_forward

Params = Mapping[str, object]


def input_embedder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz InputEmbedder."""

    q, c, p = atom_encoder_forward(
        params["atom_encoder"],
        feats,
        structure_prediction=False,
        eps=eps,
    )
    atom_enc_proj = params["atom_enc_proj_z"]
    atom_enc_bias = _linear(
        _layer_norm(
            p,
            atom_enc_proj["norm"]["scale"],
            atom_enc_proj["norm"]["bias"],
            eps,
        ),
        atom_enc_proj["linear"]["kernel"],
    )
    a, _, _ = atom_attention_encoder_forward(
        params["atom_attention_encoder"],
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=_to_keys_from_feats(feats),
        structure_prediction=False,
        eps=eps,
    )

    s = a
    s = s + _linear(
        feats["res_type"].astype(jnp.float32),
        params["res_type_encoding"]["kernel"],
    )
    msa_profile = jnp.concatenate(
        (
            feats["profile"].astype(jnp.float32),
            feats["deletion_mean"][..., None].astype(jnp.float32),
        ),
        axis=-1,
    )
    s = s + _linear(msa_profile, params["msa_profile_encoding"]["kernel"])
    s = s + params["method_conditioning_init"][
        feats["method_feature"].astype(jnp.int32)
    ]
    s = s + params["modified_conditioning_init"][feats["modified"].astype(jnp.int32)]
    cyclic = jnp.minimum(feats["cyclic_period"], 1.0)[..., None]
    s = s + _linear(cyclic, params["cyclic_conditioning_init"]["kernel"])
    s = s + params["mol_type_conditioning_init"][feats["mol_type"].astype(jnp.int32)]
    return s


def _to_keys_from_feats(feats: Mapping[str, jnp.ndarray]):
    atoms = feats["ref_pos"].shape[1]
    w = 32
    h_keys = 128
    indexing = get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)
    return lambda x: single_to_keys(x, indexing, w=w, h_keys=h_keys)


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
