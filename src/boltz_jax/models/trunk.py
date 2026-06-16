"""Pure JAX Boltz-2 trunk graph."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi

import jax
import jax.numpy as jnp

from boltz_jax.models.diffusion import conditioned_diffusion_score_forward
from boltz_jax.models.input_embedder import input_embedder_forward
from boltz_jax.models.msa import msa_module_forward
from boltz_jax.models.pairformer import pairformer_module_forward

Params = Mapping[str, object]


def boltz2_graph_score_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    *,
    recycling_steps: int = 0,
    token_layers: int | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run non-template Boltz-2 trunk, conditioning, and score in one JAX graph."""

    trunk = boltz2_trunk_forward(
        params["trunk"],
        feats,
        recycling_steps=recycling_steps,
        eps=eps,
    )
    return conditioned_diffusion_score_forward(
        params["conditioned_diffusion"],
        s_inputs=trunk["s_inputs"],
        s_trunk=trunk["s"],
        z_trunk=trunk["z"],
        relative_position_encoding=trunk["relative_position_encoding"],
        r_noisy=r_noisy,
        times=times,
        feats=feats,
        token_layers=token_layers,
        multiplicity=multiplicity,
        eps=eps,
    )


def boltz2_trunk_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    recycling_steps: int = 0,
    use_bond_type_feature: bool = True,
    cyclic_pos_enc: bool = False,
    fix_sym_check: bool = False,
    eps: float = 1e-5,
) -> dict[str, jnp.ndarray]:
    """Run the non-template Boltz-2 trunk in eval mode."""

    s_inputs = input_embedder_forward(params["input_embedder"], feats, eps=eps)
    s_init = _linear(s_inputs, params["s_init"]["kernel"])
    z_init = (
        _linear(s_inputs, params["z_init_1"]["kernel"])[:, :, None, :]
        + _linear(s_inputs, params["z_init_2"]["kernel"])[:, None, :, :]
    )
    relative_position_encoding = relative_position_forward(
        params["rel_pos"],
        feats,
        cyclic_pos_enc=cyclic_pos_enc,
        fix_sym_check=fix_sym_check,
    )
    z_init = z_init + relative_position_encoding
    z_init = z_init + _linear(
        feats["token_bonds"].astype(jnp.float32),
        params["token_bonds"]["kernel"],
    )
    if use_bond_type_feature and "token_bonds_type" in params:
        z_init = z_init + params["token_bonds_type"][
            feats["type_bonds"].astype(jnp.int32)
        ]
    z_init = z_init + contact_conditioning_forward(
        params["contact_conditioning"],
        feats,
    )

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    mask = feats["token_pad_mask"].astype(jnp.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    for _ in range(recycling_steps + 1):
        s = s_init + _linear(
            _layer_norm(
                s,
                params["s_norm"]["scale"],
                params["s_norm"]["bias"],
                eps,
            ),
            params["s_recycle"]["kernel"],
        )
        z = z_init + _linear(
            _layer_norm(
                z,
                params["z_norm"]["scale"],
                params["z_norm"]["bias"],
                eps,
            ),
            params["z_recycle"]["kernel"],
        )
        z = z + msa_module_forward(params["msa_module"], z, s_inputs, feats, eps=eps)
        s, z = pairformer_module_forward(
            params["pairformer_module"],
            s,
            z,
            mask,
            pair_mask,
            eps=eps,
        )

    return {
        "s_inputs": s_inputs,
        "s": s,
        "z": z,
        "relative_position_encoding": relative_position_encoding,
    }


def relative_position_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    r_max: int = 32,
    s_max: int = 2,
    cyclic_pos_enc: bool = False,
    fix_sym_check: bool = False,
) -> jnp.ndarray:
    """Run Boltz RelativePositionEncoder."""

    b_same_chain = feats["asym_id"][:, :, None] == feats["asym_id"][:, None, :]
    b_same_residue = (
        feats["residue_index"][:, :, None] == feats["residue_index"][:, None, :]
    )
    b_same_entity = feats["entity_id"][:, :, None] == feats["entity_id"][:, None, :]

    d_residue = feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
    if cyclic_pos_enc:
        period = jnp.where(
            feats["cyclic_period"] > 0,
            feats["cyclic_period"],
            jnp.zeros_like(feats["cyclic_period"]) + 10000,
        )
        d_residue = d_residue - period * jnp.round(d_residue / period)
    d_residue = jnp.clip(d_residue + r_max, 0, 2 * r_max).astype(jnp.int32)
    d_residue = jnp.where(b_same_chain, d_residue, 2 * r_max + 1)
    a_rel_pos = jax.nn.one_hot(d_residue, 2 * r_max + 2)

    d_token = jnp.clip(
        feats["token_index"][:, :, None] - feats["token_index"][:, None, :] + r_max,
        0,
        2 * r_max,
    ).astype(jnp.int32)
    d_token = jnp.where(b_same_chain & b_same_residue, d_token, 2 * r_max + 1)
    a_rel_token = jax.nn.one_hot(d_token, 2 * r_max + 2)

    d_chain = jnp.clip(
        feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + s_max,
        0,
        2 * s_max,
    ).astype(jnp.int32)
    same_chain_condition = ~b_same_entity if fix_sym_check else b_same_chain
    d_chain = jnp.where(same_chain_condition, 2 * s_max + 1, d_chain)
    a_rel_chain = jax.nn.one_hot(d_chain, 2 * s_max + 2)

    p = jnp.concatenate(
        (
            a_rel_pos.astype(jnp.float32),
            a_rel_token.astype(jnp.float32),
            b_same_entity[..., None].astype(jnp.float32),
            a_rel_chain.astype(jnp.float32),
        ),
        axis=-1,
    )
    return _linear(p, params["linear_layer"]["kernel"])


def contact_conditioning_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    cutoff_min: float = 4.0,
    cutoff_max: float = 20.0,
) -> jnp.ndarray:
    """Run Boltz ContactConditioning."""

    contact_conditioning = feats["contact_conditioning"][:, :, :, 2:]
    contact_threshold_normalized = (feats["contact_threshold"] - cutoff_min) / (
        cutoff_max - cutoff_min
    )
    fourier_proj = params["fourier_embedding"]["proj"]
    flat = contact_threshold_normalized.reshape((-1, 1))
    fourier = jnp.cos(
        2.0 * pi * _linear(flat, fourier_proj["kernel"], fourier_proj["bias"])
    ).reshape((*contact_threshold_normalized.shape, -1))
    contact_conditioning = jnp.concatenate(
        (
            contact_conditioning.astype(jnp.float32),
            contact_threshold_normalized[..., None].astype(jnp.float32),
            fourier.astype(jnp.float32),
        ),
        axis=-1,
    )
    encoded = _linear(
        contact_conditioning,
        params["encoder"]["kernel"],
        params["encoder"]["bias"],
    )
    flags = feats["contact_conditioning"]
    return (
        encoded * (1.0 - jnp.sum(flags[:, :, :, 0:2], axis=-1, keepdims=True))
        + params["encoding_unspecified"] * flags[:, :, :, 0:1]
        + params["encoding_unselected"] * flags[:, :, :, 1:2]
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
