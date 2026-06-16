"""Pure JAX diffusion conditioning for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from boltz_jax.models.atom import get_indexing_matrix, single_to_keys
from boltz_jax.models.conditioning import pairwise_conditioning_forward

Params = Mapping[str, object]


def diffusion_conditioning_forward(
    params: Params,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    relative_position_encoding: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    token_layers: int | None = None,
    atoms_per_window_queries: int = 32,
    atoms_per_window_keys: int = 128,
    eps: float = 1e-5,
) -> dict[str, jnp.ndarray]:
    """Run Boltz DiffusionConditioning."""

    z = pairwise_conditioning_forward(
        params["pairwise_conditioner"],
        z_trunk,
        relative_position_encoding,
        eps=eps,
    )
    q, c, p = atom_encoder_forward(
        params["atom_encoder"],
        feats,
        s_trunk,
        z,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=atoms_per_window_keys,
        eps=eps,
    )
    token_proj = params["token_trans_proj_z"]
    if token_layers is not None:
        token_proj = token_proj[:token_layers]
    return {
        "q": q,
        "c": c,
        "atom_enc_bias": _projection_list_forward(
            params["atom_enc_proj_z"], p, eps
        ),
        "atom_dec_bias": _projection_list_forward(
            params["atom_dec_proj_z"], p, eps
        ),
        "token_trans_bias": _projection_list_forward(token_proj, z, eps),
    }


def atom_encoder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    s_trunk: jnp.ndarray,
    z: jnp.ndarray,
    atoms_per_window_queries: int = 32,
    atoms_per_window_keys: int = 128,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Boltz AtomEncoder for structure diffusion conditioning."""

    atom_ref_pos = feats["ref_pos"]
    batch, atoms, _ = atom_ref_pos.shape
    atom_mask = feats["atom_pad_mask"].astype(bool)
    atom_uid = feats["ref_space_uid"]

    atom_feats = jnp.concatenate(
        (
            atom_ref_pos,
            feats["ref_charge"][..., None],
            feats["ref_element"],
            jnp.reshape(feats["ref_atom_name_chars"], (batch, atoms, 4 * 64)),
        ),
        axis=-1,
    )
    c = _linear(
        atom_feats,
        params["embed_atom_features"]["kernel"],
        params["embed_atom_features"]["bias"],
    )

    w = atoms_per_window_queries
    h_keys = atoms_per_window_keys
    num_windows = atoms // w
    indexing = get_indexing_matrix(k=num_windows, w=w, h_keys=h_keys)

    def to_keys(x: jnp.ndarray) -> jnp.ndarray:
        return single_to_keys(x, indexing, w=w, h_keys=h_keys)

    atom_ref_pos_queries = jnp.reshape(atom_ref_pos, (batch, num_windows, w, 1, 3))
    atom_ref_pos_keys = jnp.reshape(
        to_keys(atom_ref_pos), (batch, num_windows, 1, h_keys, 3)
    )
    d = atom_ref_pos_keys - atom_ref_pos_queries
    d_norm = 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))

    atom_mask_queries = jnp.reshape(atom_mask, (batch, num_windows, w, 1))
    atom_mask_keys = jnp.reshape(
        to_keys(atom_mask[..., None].astype(jnp.float32)),
        (batch, num_windows, 1, h_keys),
    ).astype(bool)
    atom_uid_queries = jnp.reshape(atom_uid, (batch, num_windows, w, 1))
    atom_uid_keys = jnp.reshape(
        to_keys(atom_uid[..., None].astype(jnp.float32)),
        (batch, num_windows, 1, h_keys),
    )
    valid = (
        atom_mask_queries
        & atom_mask_keys
        & (atom_uid_queries == atom_uid_keys.astype(atom_uid.dtype))
    ).astype(jnp.float32)[..., None]

    p = _linear(d, params["embed_atompair_ref_pos"]["kernel"]) * valid
    p = p + _linear(d_norm, params["embed_atompair_ref_dist"]["kernel"]) * valid
    p = p + _linear(valid, params["embed_atompair_mask"]["kernel"]) * valid

    q = c
    s_to_c = params["s_to_c_trans"]
    s_to_c_out = _linear(
        _layer_norm(
            s_trunk.astype(jnp.float32),
            s_to_c["norm"]["scale"],
            s_to_c["norm"]["bias"],
            eps,
        ),
        s_to_c["linear"]["kernel"],
    )
    c = c + jnp.einsum(
        "bat,btd->bad",
        feats["atom_to_token"].astype(jnp.float32),
        s_to_c_out,
    ).astype(c.dtype)

    z_to_p = params["z_to_p_trans"]
    z_to_p_out = _linear(
        _layer_norm(
            z.astype(jnp.float32),
            z_to_p["norm"]["scale"],
            z_to_p["norm"]["bias"],
            eps,
        ),
        z_to_p["linear"]["kernel"],
    )
    atom_to_token_queries = jnp.reshape(
        feats["atom_to_token"].astype(jnp.float32),
        (batch, num_windows, w, feats["atom_to_token"].shape[-1]),
    )
    atom_to_token_keys = to_keys(feats["atom_to_token"].astype(jnp.float32))
    p = p + jnp.einsum(
        "bijd,bwki,bwlj->bwkld",
        z_to_p_out,
        atom_to_token_queries,
        atom_to_token_keys,
    ).astype(p.dtype)

    p = p + _linear(
        jax.nn.relu(jnp.reshape(c, (batch, num_windows, w, 1, c.shape[-1]))),
        params["c_to_p_trans_q"]["kernel"],
    )
    p = p + _linear(
        jax.nn.relu(
            jnp.reshape(to_keys(c), (batch, num_windows, 1, h_keys, c.shape[-1]))
        ),
        params["c_to_p_trans_k"]["kernel"],
    )
    p = p + _p_mlp_forward(params["p_mlp"], p)
    return q, c, p


def _projection_list_forward(
    params: list[Params],
    x: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    return jnp.concatenate(
        [
            _linear(
                _layer_norm(x, layer["norm"]["scale"], layer["norm"]["bias"], eps),
                layer["linear"]["kernel"],
            )
            for layer in params
        ],
        axis=-1,
    )


def _p_mlp_forward(params: list[Params], p: jnp.ndarray) -> jnp.ndarray:
    out = p
    for layer in params:
        out = _linear(jax.nn.relu(out), layer["kernel"])
    return out


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
