"""Pure JAX atom-window transformer utilities for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.nn
import jax.numpy as jnp

from boltz_jax.models.diffusion_transformer import (
    diffusion_transformer_layer_forward,
)

Params = Mapping[str, object]


def get_indexing_matrix(k: int, w: int, h_keys: int) -> jnp.ndarray:
    """Return Boltz atom-window key indexing matrix."""

    if w % 2 != 0:
        msg = f"W must be even, got {w}"
        raise ValueError(msg)
    if h_keys % (w // 2) != 0:
        msg = f"H must be divisible by W // 2, got W={w}, H={h_keys}"
        raise ValueError(msg)

    h = h_keys // (w // 2)
    if h % 2 != 0:
        msg = f"H // (W // 2) must be even, got {h}"
        raise ValueError(msg)

    arange = jnp.arange(2 * k)
    index = jnp.clip((arange[None, :] - arange[:, None]) + h // 2, 0, h + 1)
    index = jnp.reshape(index, (k, 2, 2 * k))[:, 0, :]
    onehot = jax.nn.one_hot(index, h + 2)[..., 1:-1]
    onehot = jnp.transpose(onehot, (1, 0, 2))
    return jnp.reshape(onehot, (2 * k, h * k)).astype(jnp.float32)


def single_to_keys(
    single: jnp.ndarray,
    indexing_matrix: jnp.ndarray,
    w: int,
    h_keys: int,
) -> jnp.ndarray:
    """Map atom-window query-aligned values to key windows."""

    batch, atoms, dim = single.shape
    k = atoms // w
    single = jnp.reshape(single, (batch, 2 * k, w // 2, dim))
    keys = jnp.einsum("bjid,jk->bkid", single, indexing_matrix)
    return jnp.reshape(keys, (batch, k, h_keys, dim))


def diffusion_transformer_forward(
    params: Params,
    a: jnp.ndarray,
    s: jnp.ndarray,
    bias: jnp.ndarray,
    mask: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run a Boltz DiffusionTransformer stack."""

    layers = params["layers"]
    num_layers = len(layers)
    batch, rows, cols, dim = bias.shape
    bias = jnp.reshape(bias, (batch, rows, cols, num_layers, dim // num_layers))
    for index, layer_params in enumerate(layers):
        a = diffusion_transformer_layer_forward(
            layer_params,
            a,
            s,
            bias[:, :, :, index],
            mask,
            to_keys=to_keys,
            multiplicity=multiplicity,
            eps=eps,
        )
    return a


def atom_transformer_forward(
    params: Params,
    q: jnp.ndarray,
    c: jnp.ndarray,
    bias: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    mask: jnp.ndarray,
    attn_window_queries: int,
    attn_window_keys: int,
    multiplicity: int = 1,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz AtomTransformer with window reshaping."""

    w = attn_window_queries
    h_keys = attn_window_keys
    batch, atoms, dim = q.shape
    num_windows = atoms // w

    q = jnp.reshape(q, (batch * num_windows, w, dim))
    c = jnp.reshape(c, (batch * num_windows, w, c.shape[-1]))
    mask = jnp.reshape(mask, (batch * num_windows, w))
    bias = jnp.repeat(bias, multiplicity, axis=0)
    bias = jnp.reshape(
        bias, (bias.shape[0] * num_windows, w, h_keys, bias.shape[-1])
    )

    def to_keys_new(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.reshape(x, (batch, num_windows * w, -1))
        return jnp.reshape(to_keys(x), (batch * num_windows, h_keys, -1))

    q = diffusion_transformer_forward(
        params["diffusion_transformer"],
        a=q,
        s=c,
        bias=bias,
        mask=mask.astype(jnp.float32),
        to_keys=to_keys_new,
        multiplicity=1,
        eps=eps,
    )
    return jnp.reshape(q, (batch, num_windows * w, dim))


def atom_attention_encoder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    q: jnp.ndarray,
    c: jnp.ndarray,
    atom_enc_bias: jnp.ndarray,
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    r: jnp.ndarray | None = None,
    multiplicity: int = 1,
    attn_window_queries: int = 32,
    attn_window_keys: int = 128,
    structure_prediction: bool = True,
    eps: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run Boltz AtomAttentionEncoder using precomputed atom conditioning."""

    atom_mask = feats["atom_pad_mask"].astype(bool)
    if structure_prediction:
        if r is None:
            msg = "r is required when structure_prediction=True"
            raise ValueError(msg)
        q = jnp.repeat(q, multiplicity, axis=0)
        q = q + _linear(r, params["r_to_q_trans"]["kernel"])
    c = jnp.repeat(c, multiplicity, axis=0)
    atom_mask = jnp.repeat(atom_mask, multiplicity, axis=0)

    q = atom_transformer_forward(
        params["atom_encoder"],
        q=q,
        c=c,
        bias=atom_enc_bias,
        to_keys=to_keys,
        mask=atom_mask.astype(jnp.float32),
        attn_window_queries=attn_window_queries,
        attn_window_keys=attn_window_keys,
        multiplicity=multiplicity,
        eps=eps,
    )

    q_to_a = jax.nn.relu(_linear(q, params["atom_to_token_trans"]["kernel"])).astype(
        jnp.float32
    )
    atom_to_token = jnp.repeat(
        feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
    )
    atom_to_token_mean = atom_to_token / (
        jnp.sum(atom_to_token, axis=1, keepdims=True) + 1e-6
    )
    a = jnp.einsum("bat,bad->btd", atom_to_token_mean, q_to_a).astype(q.dtype)
    return a, q, c


def atom_attention_decoder_forward(
    params: Params,
    a: jnp.ndarray,
    q: jnp.ndarray,
    c: jnp.ndarray,
    atom_dec_bias: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    to_keys: Callable[[jnp.ndarray], jnp.ndarray],
    multiplicity: int = 1,
    attn_window_queries: int = 32,
    attn_window_keys: int = 128,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Run Boltz AtomAttentionDecoder using precomputed atom conditioning."""

    atom_to_token = jnp.repeat(
        feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
    )
    a_to_q = jnp.einsum(
        "bat,btd->bad",
        atom_to_token,
        _linear(a.astype(jnp.float32), params["a_to_q_trans"]["kernel"]),
    )
    q = q + a_to_q.astype(q.dtype)
    atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)

    q = atom_transformer_forward(
        params["atom_decoder"],
        q=q,
        c=c,
        bias=atom_dec_bias,
        to_keys=to_keys,
        mask=atom_mask.astype(jnp.float32),
        attn_window_queries=attn_window_queries,
        attn_window_keys=attn_window_keys,
        multiplicity=multiplicity,
        eps=eps,
    )

    update = params["atom_feat_to_atom_pos_update"]
    q = _layer_norm(q, update["norm"]["scale"], update["norm"]["bias"], eps)
    return _linear(q, update["linear"]["kernel"])


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
