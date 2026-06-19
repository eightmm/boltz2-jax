"""torch-vs-JAX RMSD matrix over precision/backend on a fixed feature set.

torch fp32 reference (built from the checkpoint, identical injected noise) vs
JAX runs at {fp32,bf16} x {xla, tri+glu tokamax}. Same features (so MSA/template
are fixed), same noise -> RMSD isolates precision + backend numerics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, "scripts")
from benchmark_boltz2_graph import (  # noqa: E402
    _load_features_pt,
    _load_torch_graph,
    _tree_to_jax,
    _tree_to_torch,
)
from compare_sampling_rmsd import _kabsch_rmsd, _raw_rmsd, _torch_sample  # noqa: E402

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict  # noqa: E402
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict  # noqa: E402
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward  # noqa: E402

CONFIGS = [
    ("fp32_xla", jnp.float32, "xla", "xla", "xla"),
    ("bf16_xla", jnp.bfloat16, "xla", "xla", "xla"),
    ("fp32_tokamax_trigl", jnp.float32, "xla", "tokamax", "tokamax"),
    ("bf16_tokamax_trigl", jnp.bfloat16, "xla", "tokamax", "tokamax"),
    ("bf16_tokamax_all", jnp.bfloat16, "tokamax", "tokamax", "tokamax"),
]


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"))
    p.add_argument("--features-pt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("outputs/integrin9_rmsd_matrix.json"))
    a = p.parse_args()

    state = load_checkpoint_state_dict(a.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _load_torch_graph(state, a.msa_layers, a.pairformer_layers, a.token_layers, device)
    jax_params = map_boltz2_graph_state_dict(
        state, num_msa_layers=a.msa_layers, num_pairformer_layers=a.pairformer_layers,
        num_token_layers=a.token_layers, token_transformer_heads=16,
    )
    feats_np, record_id = _load_features_pt(a.features_pt)
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    real_idx = np.where(np.asarray(feats_np["atom_pad_mask"]).reshape(-1).astype(bool))[0]
    torch_feats = _tree_to_torch(feats_np, device)
    jax_feats = _tree_to_jax(feats_np)

    rng = np.random.default_rng(a.seed)
    init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
    step_noises = rng.standard_normal((a.steps, 1, n_atoms, 3)).astype(np.float32)

    # torch references: fp32 (ground truth) + bf16-mixed (trunk bf16 island).
    with torch.no_grad():
        ref_fp32 = _torch_sample(
            torch_model, torch_feats, a.steps, init_noise, step_noises,
            device, recycling_steps=a.recycling,
        )[0][real_idx]
        ref_bf16 = _torch_sample(
            torch_model, torch_feats, a.steps, init_noise, step_noises,
            device, recycling_steps=a.recycling,
            trunk_autocast_dtype=torch.bfloat16,
        )[0][real_idx]

    refs = {jnp.float32: ("torch_fp32", ref_fp32), jnp.bfloat16: ("torch_bf16", ref_bf16)}
    print(f"record={record_id} n_real_atoms={real_idx.size} steps={a.steps}")
    # torch's own precision drift:
    bf_vs_fp = {"raw_rmsd": _raw_rmsd(ref_fp32, ref_bf16),
                "aligned_rmsd": _kabsch_rmsd(ref_fp32, ref_bf16),
                "max_abs": float(np.max(np.abs(ref_fp32 - ref_bf16)))}
    print(f"torch_bf16 vs torch_fp32: aligned_RMSD={bf_vs_fp['aligned_rmsd']:.4f} A")
    print(f"{'config':<22} {'ref':<11} {'raw':>9} {'aligned':>9} {'max|d|':>9}")

    results = {"torch_bf16_vs_fp32": bf_vs_fp}
    for name, dt, attn, tri, glu in CONFIGS:
        out = boltz2_sample_forward(
            jax_params, jax_feats, jax.random.PRNGKey(0),
            num_sampling_steps=a.steps, recycling_steps=a.recycling,
            token_layers=a.token_layers, augmentation=False,
            alignment_reverse_diff=True,
            init_noise=jnp.asarray(init_noise), step_noises=jnp.asarray(step_noises),
            compute_dtype=dt, attention_backend=attn, triangle_backend=tri, glu_backend=glu,
        )
        j = np.asarray(jax.block_until_ready(out["sample_atom_coords"]))[0][real_idx]
        ref_name, ref = refs[dt]
        raw, aligned = _raw_rmsd(ref, j), _kabsch_rmsd(ref, j)
        mx = float(np.max(np.abs(ref - j)))
        results[name] = {"vs_ref": ref_name, "raw_rmsd": raw,
                         "aligned_rmsd": aligned, "max_abs": mx}
        print(f"{name:<22} {ref_name:<11} {raw:>9.4f} {aligned:>9.4f} {mx:>9.4f}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(
        {"record_id": record_id, "n_real_atoms": int(real_idx.size),
         "steps": a.steps, "recycling": a.recycling, "results": results}, indent=1))
    print(f"WROTE {a.output}")


if __name__ == "__main__":
    main()
