"""Single Boltz2.forward-equivalent JAX inference wrapper.

Mirrors ``boltz.model.models.boltz2.Boltz2.forward`` (trunk -> structure
sampling -> distogram -> (bfactor) -> confidence -> (affinity)) and returns
ONE result dict. The individual head functions are reused unchanged so the
wrapper does not alter numerics.

Key mapping (boltz_jax key -> Boltz2.forward key):
    sample_atom_coords -> sample_atom_coords  (structure sampler)
    pdistogram         -> pdistogram          (distogram head)
    pbfactor           -> pbfactor            (bfactor head, optional)
    plddt/pae/pde/ptm/iptm/complex_*/... -> confidence_module outputs
    affinity_*         -> affinity_module outputs (only if affinity_params given)

Omitted vs Boltz2.forward: templates, training-only branches, miniformer,
affinity ensemble (single affinity module only), and the ``s``/``z`` trunk
tensors are not surfaced in the returned dict (kept internal).
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from boltz_jax.models.heads.affinity import affinity_module_forward
from boltz_jax.models.heads.bfactor import bfactor_forward
from boltz_jax.models.heads.confidence import confidence_module_forward
from boltz_jax.models.heads.distogram import distogram_forward
from boltz_jax.models.trunk_blocks.trunk import (
    boltz2_sample_forward,
    boltz2_trunk_forward,
)

Params = Mapping[str, object]


def boltz2_predict(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    *,
    recycling_steps: int = 3,
    num_sampling_steps: int = 200,
    augmentation: bool = True,
    steering_args: Mapping[str, object] | None = None,
    run_confidence: bool = True,
    run_distogram: bool = True,
    run_bfactor: bool = False,
    affinity_params: Params | None = None,
    eps: float = 1e-5,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    **sample_kwargs: object,
) -> dict[str, object]:
    """Run the full Boltz-2 inference graph and return one result dict.

    Computes the deterministic trunk once, then reuses it for structure sampling
    and downstream heads. Head numerics are identical to calling each head
    function directly with the same trunk/sample tensors.
    """
    multiplicity = int(sample_kwargs.pop("multiplicity", 1))
    trunk_use_scan = sample_kwargs.get(
        "trunk_use_scan", sample_kwargs.get("use_scan", True)
    )

    # NOTE: boltz2_trunk_forward has no compute_dtype param, so the trunk runs
    # fp32 regardless of a `compute_dtype` in sample_kwargs (only the diffusion
    # sampler honors it). Backend/precision/chunk knobs ARE forwarded below.
    trunk = boltz2_trunk_forward(
        params["trunk"],
        feats,
        recycling_steps=recycling_steps,
        eps=eps,
        use_scan=bool(trunk_use_scan),
        chunk_size=int(sample_kwargs.get("chunk_size", 128)),
        triangle_attention_chunk=sample_kwargs.get("triangle_attention_chunk"),
        triangle_attention_q_chunk=sample_kwargs.get("triangle_attention_q_chunk"),
        transition_hidden_chunk=sample_kwargs.get("transition_hidden_chunk"),
        matmul_precision=str(sample_kwargs.get("matmul_precision", "highest")),
        attention_backend=str(sample_kwargs.get("attention_backend", "xla")),
        triangle_backend=str(sample_kwargs.get("triangle_backend", "xla")),
        glu_backend=str(sample_kwargs.get("glu_backend", "xla")),
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
    )
    s_inputs, s, z = trunk["s_inputs"], trunk["s"], trunk["z"]

    sample_out = boltz2_sample_forward(
        params,
        feats,
        key,
        recycling_steps=recycling_steps,
        num_sampling_steps=num_sampling_steps,
        augmentation=augmentation,
        steering_args=steering_args,
        multiplicity=multiplicity,
        eps=eps,
        trunk=trunk,
        **sample_kwargs,
    )
    sample_atom_coords = sample_out["sample_atom_coords"]

    out: dict[str, object] = {"sample_atom_coords": sample_atom_coords}

    pdistogram = None
    if run_distogram or run_confidence:
        pdistogram = distogram_forward(params, z)
        if run_distogram:
            out["pdistogram"] = pdistogram

    if run_bfactor:
        out["pbfactor"] = bfactor_forward(params, s)

    if run_confidence:
        # Boltz2.forward feeds the first distogram's logits: pdistogram[:,:,:,0].
        pred_distogram_logits = pdistogram[:, :, :, 0]
        conf = confidence_module_forward(
            params["confidence"],
            s_inputs=s_inputs,
            s=s,
            z=z,
            x_pred=sample_atom_coords,
            feats=feats,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=multiplicity,
            eps=eps,
            use_scan=bool(sample_kwargs.get("use_scan", True)),
            chunk_size=int(sample_kwargs.get("chunk_size", 128)),
            triangle_attention_chunk=sample_kwargs.get("triangle_attention_chunk"),
            triangle_attention_q_chunk=sample_kwargs.get("triangle_attention_q_chunk"),
            transition_hidden_chunk=sample_kwargs.get("transition_hidden_chunk"),
            matmul_precision=str(sample_kwargs.get("matmul_precision", "highest")),
            attention_backend=str(sample_kwargs.get("attention_backend", "xla")),
            triangle_backend=str(sample_kwargs.get("triangle_backend", "xla")),
            glu_backend=str(sample_kwargs.get("glu_backend", "xla")),
        )
        out.update(conf)

    if affinity_params is not None:
        aff = affinity_module_forward(
            affinity_params,
            s_inputs=s_inputs,
            z=z,
            x_pred=sample_atom_coords,
            feats=feats,
            multiplicity=multiplicity,
            eps=eps,
        )
        out.update(aff)

    return out
