"""Explicit PyTorch state_dict to JAX parameter mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

TransitionParams = dict[str, dict[str, jnp.ndarray]]
AttentionPairBiasParams = dict[str, dict[str, jnp.ndarray]]
TriangleMultiplicationParams = dict[str, dict[str, jnp.ndarray]]
TriangleAttentionParams = dict[str, dict[str, jnp.ndarray]]
PairformerLayerParams = dict[str, dict[str, jnp.ndarray]]
PairformerModuleParams = dict[str, list[PairformerLayerParams]]
SingleConditioningParams = dict[str, Any]
PairwiseConditioningParams = dict[str, Any]
AdaLNParams = dict[str, dict[str, jnp.ndarray]]
ConditionedTransitionBlockParams = dict[str, Any]
DiffusionTransformerLayerParams = dict[str, Any]
DiffusionTransformerParams = dict[str, list[DiffusionTransformerLayerParams]]
AtomTransformerParams = dict[str, DiffusionTransformerParams]
AtomAttentionEncoderParams = dict[str, Any]
AtomAttentionDecoderParams = dict[str, Any]
DiffusionScoreModelParams = dict[str, Any]


def map_transition_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> TransitionParams:
    """Map one Boltz Transition module from PyTorch keys to a JAX pytree."""

    required_keys = (
        f"{prefix}.norm.weight",
        f"{prefix}.norm.bias",
        f"{prefix}.fc1.weight",
        f"{prefix}.fc2.weight",
        f"{prefix}.fc3.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required Transition state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm.bias"]),
        },
        "fc1": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc1.weight"])},
        "fc2": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc2.weight"])},
        "fc3": {"kernel": _linear_kernel(state_dict[f"{prefix}.fc3.weight"])},
    }


def map_attention_pair_bias_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> AttentionPairBiasParams:
    """Map one Boltz AttentionPairBias v2 module to a JAX pytree."""

    required_keys = (
        f"{prefix}.proj_q.weight",
        f"{prefix}.proj_q.bias",
        f"{prefix}.proj_k.weight",
        f"{prefix}.proj_v.weight",
        f"{prefix}.proj_g.weight",
        f"{prefix}.proj_z.0.weight",
        f"{prefix}.proj_z.0.bias",
        f"{prefix}.proj_z.1.weight",
        f"{prefix}.proj_o.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required AttentionPairBias state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "proj_q": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.proj_q.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.proj_q.bias"]),
        },
        "proj_k": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_k.weight"])},
        "proj_v": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_v.weight"])},
        "proj_g": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_g.weight"])},
        "proj_z_norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.proj_z.0.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.proj_z.0.bias"]),
        },
        "proj_z": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_z.1.weight"])},
        "proj_o": {"kernel": _linear_kernel(state_dict[f"{prefix}.proj_o.weight"])},
    }


def map_adaln_state_dict(state_dict: Mapping[str, Any], prefix: str) -> AdaLNParams:
    """Map one Boltz AdaLN module to a JAX pytree."""

    required_keys = (
        f"{prefix}.s_norm.weight",
        f"{prefix}.s_scale.weight",
        f"{prefix}.s_scale.bias",
        f"{prefix}.s_bias.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = f"Missing required AdaLN state_dict keys for prefix {prefix!r}: {missing}"
        raise KeyError(msg)

    return {
        "s_norm": {"scale": _to_jax_array(state_dict[f"{prefix}.s_norm.weight"])},
        "s_scale": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.s_scale.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.s_scale.bias"]),
        },
        "s_bias": {"kernel": _linear_kernel(state_dict[f"{prefix}.s_bias.weight"])},
    }


def map_conditioned_transition_block_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> ConditionedTransitionBlockParams:
    """Map one Boltz ConditionedTransitionBlock to a JAX pytree."""

    required_keys = (
        f"{prefix}.swish_gate.0.weight",
        f"{prefix}.a_to_b.weight",
        f"{prefix}.b_to_a.weight",
        f"{prefix}.output_projection.0.weight",
        f"{prefix}.output_projection.0.bias",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required ConditionedTransitionBlock state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "adaln": map_adaln_state_dict(state_dict, f"{prefix}.adaln"),
        "swish_gate": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.swish_gate.0.weight"])
        },
        "a_to_b": {"kernel": _linear_kernel(state_dict[f"{prefix}.a_to_b.weight"])},
        "b_to_a": {"kernel": _linear_kernel(state_dict[f"{prefix}.b_to_a.weight"])},
        "output_projection": {
            "kernel": _linear_kernel(
                state_dict[f"{prefix}.output_projection.0.weight"]
            ),
            "bias": _to_jax_array(state_dict[f"{prefix}.output_projection.0.bias"]),
        },
    }


def map_diffusion_transformer_layer_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_heads: int = 8,
) -> DiffusionTransformerLayerParams:
    """Map one Boltz DiffusionTransformerLayer to a nested JAX pytree."""

    required_keys = (
        f"{prefix}.pair_bias_attn.proj_q.weight",
        f"{prefix}.pair_bias_attn.proj_q.bias",
        f"{prefix}.pair_bias_attn.proj_k.weight",
        f"{prefix}.pair_bias_attn.proj_v.weight",
        f"{prefix}.pair_bias_attn.proj_g.weight",
        f"{prefix}.pair_bias_attn.proj_o.weight",
        f"{prefix}.output_projection_linear.weight",
        f"{prefix}.output_projection_linear.bias",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required DiffusionTransformerLayer state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    params: DiffusionTransformerLayerParams = {
        "adaln": map_adaln_state_dict(state_dict, f"{prefix}.adaln"),
        "pair_bias_attn": {
            "num_heads": num_heads,
            "proj_q": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.pair_bias_attn.proj_q.weight"]
                ),
                "bias": _to_jax_array(
                    state_dict[f"{prefix}.pair_bias_attn.proj_q.bias"]
                ),
            },
            "proj_k": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.pair_bias_attn.proj_k.weight"]
                )
            },
            "proj_v": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.pair_bias_attn.proj_v.weight"]
                )
            },
            "proj_g": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.pair_bias_attn.proj_g.weight"]
                )
            },
            "proj_o": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.pair_bias_attn.proj_o.weight"]
                )
            },
        },
        "output_projection": {
            "kernel": _linear_kernel(
                state_dict[f"{prefix}.output_projection_linear.weight"]
            ),
            "bias": _to_jax_array(
                state_dict[f"{prefix}.output_projection_linear.bias"]
            ),
        },
        "transition": map_conditioned_transition_block_state_dict(
            state_dict, f"{prefix}.transition"
        ),
    }

    post_lnorm_weight = f"{prefix}.post_lnorm.weight"
    post_lnorm_bias = f"{prefix}.post_lnorm.bias"
    if post_lnorm_weight in state_dict or post_lnorm_bias in state_dict:
        if post_lnorm_weight not in state_dict or post_lnorm_bias not in state_dict:
            msg = (
                "Missing required DiffusionTransformerLayer post_lnorm "
                f"state_dict keys for prefix {prefix!r}"
            )
            raise KeyError(msg)
        params["post_lnorm"] = {
            "scale": _to_jax_array(state_dict[post_lnorm_weight]),
            "bias": _to_jax_array(state_dict[post_lnorm_bias]),
        }

    return params


def map_diffusion_transformer_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_heads: int,
    num_layers: int | None = None,
) -> DiffusionTransformerParams:
    """Map a Boltz DiffusionTransformer stack to a JAX pytree."""

    layer_indices = _module_list_indices(state_dict, f"{prefix}.layers", num_layers)
    return {
        "layers": [
            map_diffusion_transformer_layer_state_dict(
                state_dict,
                f"{prefix}.layers.{index}",
                num_heads=num_heads,
            )
            for index in layer_indices
        ]
    }


def map_atom_transformer_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_heads: int,
    num_layers: int | None = None,
) -> AtomTransformerParams:
    """Map a Boltz AtomTransformer to a JAX pytree."""

    return {
        "diffusion_transformer": map_diffusion_transformer_state_dict(
            state_dict,
            f"{prefix}.diffusion_transformer",
            num_heads=num_heads,
            num_layers=num_layers,
        )
    }


def map_atom_attention_encoder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_heads: int = 4,
    num_layers: int | None = None,
) -> AtomAttentionEncoderParams:
    """Map a Boltz AtomAttentionEncoder to a JAX pytree."""

    required_keys = (
        f"{prefix}.r_to_q_trans.weight",
        f"{prefix}.atom_to_token_trans.0.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required AtomAttentionEncoder state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "r_to_q_trans": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.r_to_q_trans.weight"])
        },
        "atom_encoder": map_atom_transformer_state_dict(
            state_dict,
            f"{prefix}.atom_encoder",
            num_heads=num_heads,
            num_layers=num_layers,
        ),
        "atom_to_token_trans": {
            "kernel": _linear_kernel(
                state_dict[f"{prefix}.atom_to_token_trans.0.weight"]
            )
        },
    }


def map_atom_attention_decoder_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_heads: int = 4,
    num_layers: int | None = None,
) -> AtomAttentionDecoderParams:
    """Map a Boltz AtomAttentionDecoder to a JAX pytree."""

    required_keys = (
        f"{prefix}.a_to_q_trans.weight",
        f"{prefix}.atom_feat_to_atom_pos_update.0.weight",
        f"{prefix}.atom_feat_to_atom_pos_update.0.bias",
        f"{prefix}.atom_feat_to_atom_pos_update.1.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required AtomAttentionDecoder state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "a_to_q_trans": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.a_to_q_trans.weight"])
        },
        "atom_decoder": map_atom_transformer_state_dict(
            state_dict,
            f"{prefix}.atom_decoder",
            num_heads=num_heads,
            num_layers=num_layers,
        ),
        "atom_feat_to_atom_pos_update": {
            "norm": {
                "scale": _to_jax_array(
                    state_dict[f"{prefix}.atom_feat_to_atom_pos_update.0.weight"]
                ),
                "bias": _to_jax_array(
                    state_dict[f"{prefix}.atom_feat_to_atom_pos_update.0.bias"]
                ),
            },
            "linear": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.atom_feat_to_atom_pos_update.1.weight"]
                )
            },
        },
    }


def map_triangle_multiplication_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> TriangleMultiplicationParams:
    """Map one Boltz TriangleMultiplication module to a JAX pytree."""

    required_keys = (
        f"{prefix}.norm_in.weight",
        f"{prefix}.norm_in.bias",
        f"{prefix}.p_in.weight",
        f"{prefix}.g_in.weight",
        f"{prefix}.norm_out.weight",
        f"{prefix}.norm_out.bias",
        f"{prefix}.p_out.weight",
        f"{prefix}.g_out.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required TriangleMultiplication state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "norm_in": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_in.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_in.bias"]),
        },
        "p_in": {"kernel": _linear_kernel(state_dict[f"{prefix}.p_in.weight"])},
        "g_in": {"kernel": _linear_kernel(state_dict[f"{prefix}.g_in.weight"])},
        "norm_out": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_out.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_out.bias"]),
        },
        "p_out": {"kernel": _linear_kernel(state_dict[f"{prefix}.p_out.weight"])},
        "g_out": {"kernel": _linear_kernel(state_dict[f"{prefix}.g_out.weight"])},
    }


def map_triangle_attention_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> TriangleAttentionParams:
    """Map one Boltz TriangleAttention module to a JAX pytree."""

    required_keys = (
        f"{prefix}.layer_norm.weight",
        f"{prefix}.layer_norm.bias",
        f"{prefix}.linear.weight",
        f"{prefix}.mha.linear_q.weight",
        f"{prefix}.mha.linear_k.weight",
        f"{prefix}.mha.linear_v.weight",
        f"{prefix}.mha.linear_o.weight",
        f"{prefix}.mha.linear_g.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required TriangleAttention state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "layer_norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.layer_norm.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.layer_norm.bias"]),
        },
        "linear": {"kernel": _linear_kernel(state_dict[f"{prefix}.linear.weight"])},
        "mha": {
            "linear_q": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.mha.linear_q.weight"])
            },
            "linear_k": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.mha.linear_k.weight"])
            },
            "linear_v": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.mha.linear_v.weight"])
            },
            "linear_o": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.mha.linear_o.weight"])
            },
            "linear_g": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.mha.linear_g.weight"])
            },
        },
    }


def map_pairformer_layer_state_dict(
    state_dict: Mapping[str, Any], prefix: str
) -> PairformerLayerParams:
    """Map one Boltz PairformerLayer to a nested JAX pytree."""

    required_keys = (
        f"{prefix}.pre_norm_s.weight",
        f"{prefix}.pre_norm_s.bias",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required PairformerLayer state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "pre_norm_s": {
            "scale": _to_jax_array(state_dict[f"{prefix}.pre_norm_s.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.pre_norm_s.bias"]),
        },
        "attention": map_attention_pair_bias_state_dict(
            state_dict, f"{prefix}.attention"
        ),
        "tri_mul_out": map_triangle_multiplication_state_dict(
            state_dict, f"{prefix}.tri_mul_out"
        ),
        "tri_mul_in": map_triangle_multiplication_state_dict(
            state_dict, f"{prefix}.tri_mul_in"
        ),
        "tri_att_start": map_triangle_attention_state_dict(
            state_dict, f"{prefix}.tri_att_start"
        ),
        "tri_att_end": map_triangle_attention_state_dict(
            state_dict, f"{prefix}.tri_att_end"
        ),
        "transition_s": map_transition_state_dict(state_dict, f"{prefix}.transition_s"),
        "transition_z": map_transition_state_dict(state_dict, f"{prefix}.transition_z"),
    }


def map_pairformer_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "pairformer_module",
    num_layers: int | None = None,
) -> PairformerModuleParams:
    """Map a Boltz PairformerModule stack to a JAX pytree."""

    layer_indices = sorted(
        {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith(f"{prefix}.layers.")
        }
    )
    if num_layers is not None:
        layer_indices = layer_indices[:num_layers]
    if not layer_indices:
        msg = f"No PairformerModule layers found for prefix {prefix!r}"
        raise KeyError(msg)

    return {
        "layers": [
            map_pairformer_layer_state_dict(state_dict, f"{prefix}.layers.{index}")
            for index in layer_indices
        ]
    }


def map_single_conditioning_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_transitions: int | None = None,
) -> SingleConditioningParams:
    """Map a Boltz SingleConditioning module to a JAX pytree."""

    required_keys = (
        f"{prefix}.norm_single.weight",
        f"{prefix}.norm_single.bias",
        f"{prefix}.single_embed.weight",
        f"{prefix}.single_embed.bias",
        f"{prefix}.fourier_embed.proj.weight",
        f"{prefix}.fourier_embed.proj.bias",
        f"{prefix}.norm_fourier.weight",
        f"{prefix}.norm_fourier.bias",
        f"{prefix}.fourier_to_single.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required SingleConditioning state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    transition_indices = _module_list_indices(
        state_dict, f"{prefix}.transitions", num_transitions
    )

    return {
        "norm_single": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_single.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_single.bias"]),
        },
        "single_embed": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.single_embed.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.single_embed.bias"]),
        },
        "fourier_embed": {
            "proj": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.fourier_embed.proj.weight"]
                ),
                "bias": _to_jax_array(
                    state_dict[f"{prefix}.fourier_embed.proj.bias"]
                ),
            }
        },
        "norm_fourier": {
            "scale": _to_jax_array(state_dict[f"{prefix}.norm_fourier.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.norm_fourier.bias"]),
        },
        "fourier_to_single": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.fourier_to_single.weight"])
        },
        "transitions": [
            map_transition_state_dict(state_dict, f"{prefix}.transitions.{index}")
            for index in transition_indices
        ],
    }


def map_pairwise_conditioning_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_transitions: int | None = None,
) -> PairwiseConditioningParams:
    """Map a Boltz PairwiseConditioning module to a JAX pytree."""

    required_keys = (
        f"{prefix}.dim_pairwise_init_proj.0.weight",
        f"{prefix}.dim_pairwise_init_proj.0.bias",
        f"{prefix}.dim_pairwise_init_proj.1.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required PairwiseConditioning state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    transition_indices = _module_list_indices(
        state_dict, f"{prefix}.transitions", num_transitions
    )

    return {
        "dim_pairwise_init_proj": {
            "norm": {
                "scale": _to_jax_array(
                    state_dict[f"{prefix}.dim_pairwise_init_proj.0.weight"]
                ),
                "bias": _to_jax_array(
                    state_dict[f"{prefix}.dim_pairwise_init_proj.0.bias"]
                ),
            },
            "linear": {
                "kernel": _linear_kernel(
                    state_dict[f"{prefix}.dim_pairwise_init_proj.1.weight"]
                )
            },
        },
        "transitions": [
            map_transition_state_dict(state_dict, f"{prefix}.transitions.{index}")
            for index in transition_indices
        ],
    }


def map_diffusion_score_model_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_token_layers: int | None = None,
) -> DiffusionScoreModelParams:
    """Map Boltz DiffusionModule score_model weights to a JAX pytree."""

    required_keys = (
        f"{prefix}.s_to_a_linear.0.weight",
        f"{prefix}.s_to_a_linear.0.bias",
        f"{prefix}.s_to_a_linear.1.weight",
        f"{prefix}.a_norm.weight",
        f"{prefix}.a_norm.bias",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = (
            "Missing required DiffusionModule score_model state_dict keys "
            f"for prefix {prefix!r}: {missing}"
        )
        raise KeyError(msg)

    return {
        "single_conditioner": map_single_conditioning_state_dict(
            state_dict, f"{prefix}.single_conditioner"
        ),
        "atom_attention_encoder": map_atom_attention_encoder_state_dict(
            state_dict, f"{prefix}.atom_attention_encoder"
        ),
        "s_to_a_linear": {
            "norm": {
                "scale": _to_jax_array(state_dict[f"{prefix}.s_to_a_linear.0.weight"]),
                "bias": _to_jax_array(state_dict[f"{prefix}.s_to_a_linear.0.bias"]),
            },
            "linear": {
                "kernel": _linear_kernel(state_dict[f"{prefix}.s_to_a_linear.1.weight"])
            },
        },
        "token_transformer": map_diffusion_transformer_state_dict(
            state_dict,
            f"{prefix}.token_transformer",
            num_heads=8,
            num_layers=num_token_layers,
        ),
        "a_norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.a_norm.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.a_norm.bias"]),
        },
        "atom_attention_decoder": map_atom_attention_decoder_state_dict(
            state_dict, f"{prefix}.atom_attention_decoder"
        ),
    }


def _module_list_indices(
    state_dict: Mapping[str, Any],
    prefix: str,
    num_items: int | None,
) -> list[int]:
    indices = sorted(
        {
            int(key.removeprefix(f"{prefix}.").split(".", 1)[0])
            for key in state_dict
            if key.startswith(f"{prefix}.")
        }
    )
    if num_items is not None:
        indices = indices[:num_items]
    if not indices:
        msg = f"No module list items found for prefix {prefix!r}"
        raise KeyError(msg)
    return indices


def _linear_kernel(weight: Any) -> jnp.ndarray:
    """Convert PyTorch Linear.weight [out_dim, in_dim] to JAX [in_dim, out_dim]."""

    return _to_jax_array(weight).T


def _to_jax_array(value: Any) -> jnp.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)
