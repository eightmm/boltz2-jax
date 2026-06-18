"""Measure the 3 weight-compatible SPEED-opt configs on GPU (JAX only).

Configs on real 1US0_A (314 tok), steps=200 recycling=3, augmentation off,
alignment_reverse_diff on, use_scan=True, with identical injected noise:

  (1) baseline      : matmul_precision="highest" (current default)
  (2) tf32          : matmul_precision="default" + global jax flag "default"
  (3) tf32+chunk256 : (2) + chunk_size=256

Reports steady latency + peak VRAM per config, and aligned-RMSD drift of the
denoised coords for the TF32 paths vs the highest baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import (  # noqa: E402
    _jax_memory_stats,
    _load_features_pt,
    _tree_to_jax,
)


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Aligned RMSD between two [N,3] coordinate sets (no mass weighting)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = a @ r.T
    return float(np.sqrt(((a_rot - b) ** 2).sum(1).mean()))


def _run(jax_params, jax_feats, n_atoms, steps, recycling, token_layers,
         iters, chunk_size, matmul_precision, seed):
    if matmul_precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    else:
        jax.config.update("jax_default_matmul_precision", "default")

    rng = np.random.default_rng(0)
    init_noise = jax.numpy.asarray(
        rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
    )
    step_noises = jax.numpy.asarray(
        rng.standard_normal((steps, 1, n_atoms, 3)).astype(np.float32)
    )

    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            num_sampling_steps=steps,
            recycling_steps=recycling,
            token_layers=token_layers,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
            chunk_size=chunk_size,
            matmul_precision=matmul_precision,
        )
    )

    def call(s):
        return sampler(
            jax_params, jax_feats, jax.random.PRNGKey(s),
            init_noise=init_noise, step_noises=step_noises,
        )["sample_atom_coords"]

    out = call(seed)
    out.block_until_ready()
    coords = np.asarray(out)[0]

    times = []
    for _ in range(iters):
        t = time.perf_counter()
        call(seed).block_until_ready()
        times.append((time.perf_counter() - t) * 1000.0)
    mem = _jax_memory_stats()
    peak = (mem.get("peak_bytes_in_use", 0) / (1024**2)) if mem else None
    return statistics.mean(times), times, peak, coords


def main() -> None:
    jax.config.update("jax_enable_x64", False)
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"))
    p.add_argument("--features-pt", type=Path,
                   default=Path("outputs/real_features/1US0_A.pt"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--config", required=True,
                   choices=["baseline", "tf32", "tf32_chunk256"])
    p.add_argument("--output", type=Path,
                   default=Path("outputs/speed_opt_bench.json"))
    args = p.parse_args()

    cfg = {
        "baseline": ("highest", 128),
        "tf32": ("default", 128),
        "tf32_chunk256": ("default", 256),
    }[args.config]
    precision, chunk_size = cfg

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    jax_params = map_boltz2_graph_state_dict(
        state_cpu, num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers, token_transformer_heads=16,
    )
    feats_np, record_id = _load_features_pt(args.features_pt)
    n_tokens = int(feats_np["token_pad_mask"].shape[1])
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    jax_feats = _tree_to_jax(feats_np)

    mean, times, peak, coords = _run(
        jax_params, jax_feats, n_atoms, args.steps, args.recycling,
        args.token_layers, args.iters, chunk_size, precision, seed=1,
    )

    payload = {
        "config": args.config,
        "record_id": record_id,
        "n_tokens": n_tokens,
        "n_atoms": n_atoms,
        "steps": args.steps,
        "recycling": args.recycling,
        "matmul_precision": precision,
        "chunk_size": chunk_size,
        "steady_mean_ms": mean,
        "times_ms": times,
        "peak_mib": peak,
    }
    np.save(str(args.output.with_suffix(".coords.npy")), coords)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"config={args.config} prec={precision} chunk={chunk_size} "
          f"n_tok={n_tokens} n_atoms={n_atoms}")
    print(f"steady_mean_ms={mean:.1f} peak_mib={peak:.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
