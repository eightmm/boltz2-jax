"""PyTorch Boltz-2 reference inference -> CIF + time + peak VRAM.

Builds the torch reference graph from the checkpoint, runs the same Euler
sampler as compare_sampling_rmsd (fp32 and bf16-mixed trunk), writes CIF, and
reports wall-clock + torch peak VRAM. For side-by-side structural comparison
against the JAX outputs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from benchmark_boltz2_graph import (  # noqa: E402
    _load_features_pt,
    _load_torch_graph,
    _tree_to_torch,
)
from compare_sampling_rmsd import _torch_sample  # noqa: E402

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict  # noqa: E402
from boltz_jax.data.write.structure import write_prediction  # noqa: E402

CONFIGS = [("torch_fp32", None), ("torch_bf16", torch.bfloat16)]


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"))
    p.add_argument("--features-pt", required=True, type=Path)
    p.add_argument("--structure-npz", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/integrin9_structures"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = load_checkpoint_state_dict(a.checkpoint)
    model = _load_torch_graph(state, a.msa_layers, a.pairformer_layers, a.token_layers, device)
    feats_np, _ = _load_features_pt(a.features_pt)
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    atom_mask = np.asarray(feats_np["atom_pad_mask"]).reshape(-1)
    torch_feats = _tree_to_torch(feats_np, device)

    rng = np.random.default_rng(a.seed)
    init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
    step_noises = rng.standard_normal((a.steps, 1, n_atoms, 3)).astype(np.float32)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, dt in CONFIGS:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            coords = _torch_sample(
                model, torch_feats, a.steps, init_noise, step_noises,
                device, recycling_steps=a.recycling, trunk_autocast_dtype=dt,
            )
        dt_s = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        out = a.out_dir / f"integrin9_{name}.cif"
        written = write_prediction(
            structure_npz=a.structure_npz, coords=coords,
            atom_pad_mask=atom_mask, out_path=out, fmt="cif",
        )
        print(f"RESULT cfg={name} steps={a.steps} time={dt_s:.2f}s "
              f"peak_vram={peak:.0f}MiB WROTE {written}")


if __name__ == "__main__":
    main()
