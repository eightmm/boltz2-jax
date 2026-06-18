"""Full-graph JAX-vs-PyTorch Boltz-2 sampling speed + peak-VRAM benchmark.

Compiles the WHOLE JAX sampler (trunk + diffusion conditioning + N-step
sampling loop via ``lax.scan``) into ONE jitted XLA graph and compares against
the PyTorch reference loop (reused from ``compare_sampling_rmsd``) on real
features, at production defaults (num_sampling_steps=200, recycling_steps=3).

No-steering / augmentation-off / alignment_reverse_diff path, fp32, tf32 off.
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
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import (  # noqa: E402
    BOLTZ_SRC,
    _jax_memory_stats,
    _load_features_pt,
    _load_torch_graph,
    _tree_to_jax,
    _tree_to_torch,
)
from compare_sampling_rmsd import _torch_sample  # noqa: E402


def _bench_torch(torch_model, feats, num_steps, n_atoms, device, warmup, iters,
                 recycling):
    rng = np.random.default_rng(0)

    def one():
        init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
        step_noises = rng.standard_normal((num_steps, 1, n_atoms, 3)).astype(np.float32)
        with torch.no_grad():
            return _torch_sample(
                torch_model, feats, num_steps, init_noise, step_noises, device,
                recycling_steps=recycling,
            )

    for _ in range(warmup):
        one()
        if device == "cuda":
            torch.cuda.synchronize()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        one()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return statistics.mean(times), times, peak


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"))
    p.add_argument("--features-pt", type=Path,
                   default=Path("outputs/real_features/1UBQ_A.pt"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--measure-aug", action="store_true",
                   help="Also time eager JAX augmentation=True as a reference.")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/sampling_fullgraph_benchmark.json"))
    args = p.parse_args()

    assert BOLTZ_SRC.exists(), f"boltz src not found: {BOLTZ_SRC}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_model = _load_torch_graph(
        state_cpu, args.msa_layers, args.pairformer_layers, args.token_layers, device
    )
    jax_params = map_boltz2_graph_state_dict(
        state_cpu, num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers, token_transformer_heads=16,
    )

    feats_np, record_id = _load_features_pt(args.features_pt)
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    torch_feats = _tree_to_torch(feats_np, device)
    jax_feats = _tree_to_jax(feats_np)

    # --- JAX: one jitted full graph (trunk + conditioning + scan loop).
    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            num_sampling_steps=args.steps,
            recycling_steps=args.recycling,
            token_layers=args.token_layers,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
        )
    )

    def jax_call(seed):
        return sampler(jax_params, jax_feats, jax.random.PRNGKey(seed))[
            "sample_atom_coords"
        ]

    start = time.perf_counter()
    jax_call(0).block_until_ready()
    compile_plus_first_ms = (time.perf_counter() - start) * 1000.0

    jtimes = []
    for i in range(args.iters):
        start = time.perf_counter()
        jax_call(i + 1).block_until_ready()
        jtimes.append((time.perf_counter() - start) * 1000.0)
    jax_steady_mean = statistics.mean(jtimes)
    mem = _jax_memory_stats()
    jax_peak_mib = (mem.get("peak_bytes_in_use", 0) / (1024**2)) if mem else None

    # --- Optional: eager JAX augmentation=True latency (reference path cost).
    aug_block = None
    if args.measure_aug:
        aug_sampler = jax.jit(
            partial(
                boltz2_sample_forward,
                num_sampling_steps=args.steps,
                recycling_steps=args.recycling,
                token_layers=args.token_layers,
                augmentation=True,
                alignment_reverse_diff=True,
                use_scan=False,  # aug path is eager-only (no scan)
            )
        )

        def aug_call(seed):
            return aug_sampler(jax_params, jax_feats, jax.random.PRNGKey(seed))[
                "sample_atom_coords"
            ]

        aug_call(0).block_until_ready()
        atimes = []
        for i in range(args.iters):
            start = time.perf_counter()
            aug_call(i + 1).block_until_ready()
            atimes.append((time.perf_counter() - start) * 1000.0)
        aug_mean = statistics.mean(atimes)
        aug_block = {
            "steady_mean_ms": aug_mean,
            "times_ms": atimes,
            "delta_vs_aug_off_ms": aug_mean - jax_steady_mean,
            "note": "eager augmentation=True (random per-step rigid rotation); "
                    "rigid + removed by per-step weighted_rigid_align so final "
                    "structure quality is unchanged. Timing reference only.",
        }

    # --- PyTorch reference loop (SAME recycling, augmentation=False).
    torch_mean, ttimes, torch_peak = _bench_torch(
        torch_model, torch_feats, args.steps, n_atoms, device,
        args.warmup, args.iters, args.recycling,
    )
    torch_peak_mib = torch_peak / (1024**2) if torch_peak else None

    speedup = (torch_mean / jax_steady_mean) if jax_steady_mean else None

    payload = {
        "record_id": record_id,
        "features_pt": str(args.features_pt),
        "n_atoms": n_atoms,
        "steps": args.steps,
        "recycling": args.recycling,
        "iters": args.iters,
        "dtype": "fp32",
        "tf32": False,
        "matmul_precision": "highest (default)",
        "augmentation": False,
        "alignment": True,
        "augmentation_note": (
            "JAX compiled scan path runs in DETERMINISTIC SERVING MODE "
            "(augmentation off, steering off). PyTorch reference is run with the "
            "SAME augmentation=False so the comparison is apples-to-apples. "
            "Augmentation (random per-step rigid rotation) is part of the Boltz "
            "predict default but is rigid and removed by per-step "
            "weighted_rigid_align, so it does not change final structure quality."
        ),
        "mode": "deterministic serving mode (augmentation off)",
        "jax_aug_on_eager": aug_block,
        "jax": {
            "compile_plus_first_ms": compile_plus_first_ms,
            "steady_mean_ms": jax_steady_mean,
            "times_ms": jtimes,
            "peak_mib": jax_peak_mib,
        },
        "torch": {
            "steady_mean_ms": torch_mean,
            "times_ms": ttimes,
            "peak_mib": torch_peak_mib,
        },
        "jax_vs_torch_speedup": speedup,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Full-graph sampling benchmark ===")
    print(f"record={record_id}  n_atoms={n_atoms}  steps={args.steps}  "
          f"recycling={args.recycling}  iters={args.iters}  fp32 tf32=off")
    print(f"{'metric':<32}{'JAX':>16}{'PyTorch':>16}")
    print(f"{'compile+first run (ms)':<32}{compile_plus_first_ms:>16.1f}{'-':>16}")
    print(f"{'steady mean latency (ms)':<32}{jax_steady_mean:>16.1f}"
          f"{torch_mean:>16.1f}")
    pj = f"{jax_peak_mib:.1f}" if jax_peak_mib else "n/a"
    pt = f"{torch_peak_mib:.1f}" if torch_peak_mib else "n/a"
    print(f"{'peak VRAM (MiB)':<32}{pj:>16}{pt:>16}")
    if speedup:
        print(f"\nJAX vs PyTorch steady speedup: {speedup:.2f}x")
    if aug_block:
        print(f"JAX eager augmentation=True steady (ms): "
              f"{aug_block['steady_mean_ms']:.1f} "
              f"(delta vs aug-off: {aug_block['delta_vs_aug_off_ms']:+.1f} ms)")
    print("NOTE: deterministic serving mode (augmentation OFF) on BOTH sides.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
