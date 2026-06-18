"""Reduced-precision (bf16, fp16) vs fp32 JAX Boltz-2 full-graph sampling bench.

Runs the SAME single jitted full-graph sampler (trunk + diffusion conditioning
+ N-step lax.scan loop) at three precisions on real 1UBQ_A features and reports,
per precision:
  * steady-state mean latency (>=iters runs, block_until_ready),
  * JAX peak VRAM (peak_bytes_in_use),
  * structural DRIFT vs the fp32 result (raw + Kabsch-aligned RMSD, max abs diff).

Precision handling
------------------
* Params pytree cast to target dtype (jax.tree.map x->x.astype(dtype)).
* Input feature FLOAT arrays cast to target dtype; integer/boolean arrays
  (masks, indices, one-hot sources) keep their original dtype.
* Injected init_noise / step_noises (the SAME numpy seed-0 arrays for every
  precision) cast to target dtype so trunk + conditioning + score all run low.
* The weighted rigid align is a fp32 island INSIDE the sampler
  (_weighted_rigid_align upcasts its SVD to float32 -- verified in trunk.py
  lines ~190-194/415). That is a justified numerical necessity (SVD is
  unstable in bf16/fp16), not a workaround.

Drift is apples-to-apples: only precision differs (same fixed noise,
augmentation=False, alignment on, no steering).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_sampling_rmsd import _kabsch_rmsd, _raw_rmsd  # noqa: E402

_DTYPES = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}


def _jax_peak_mib() -> float | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use", 0) / (1024**2)


def _cast_floating(x, dtype):
    """Cast floating array leaves to ``dtype``; leave int/bool arrays and
    non-array scalar (int/float/bool/None) leaves untouched."""
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(dtype)
    return x


def _cast_params(params, dtype):
    return jax.tree.map(lambda x: _cast_floating(x, dtype), params)


def _cast_feats(feats, dtype):
    return {k: _cast_floating(v, dtype) for k, v in feats.items()}


def _run_one(args, pr: str) -> dict:
    """Run a single precision in THIS process. Peak VRAM is process-cumulative,
    so the orchestrator runs each precision in a fresh subprocess for a clean
    per-precision peak_bytes_in_use."""
    dtype = _DTYPES[pr]
    params_f32 = load_params(args.weights)
    feats_f32 = load_features_npz(args.features_npz)
    n_atoms = int(feats_f32["atom_pad_mask"].shape[1])

    rng = np.random.default_rng(0)
    init_noise_np = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
    step_noises_np = rng.standard_normal(
        (args.steps, 1, n_atoms, 3)
    ).astype(np.float32)

    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            token_layers=args.token_layers,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
        )
    )

    params = _cast_params(params_f32, dtype)
    feats = _cast_feats(feats_f32, dtype)
    init_noise = jnp.asarray(init_noise_np, dtype=dtype)
    step_noises = jnp.asarray(step_noises_np, dtype=dtype)

    def call():
        return sampler(
            params, feats, jax.random.PRNGKey(0),
            init_noise=init_noise, step_noises=step_noises,
        )["sample_atom_coords"]

    start = time.perf_counter()
    out = call().block_until_ready()
    compile_plus_first_ms = (time.perf_counter() - start) * 1000.0
    out_np = np.asarray(out.astype(jnp.float32)).reshape(-1, 3)

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        call().block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "dtype": pr,
        "matmul_precision": args.matmul_precision,
        "compile_plus_first_ms": compile_plus_first_ms,
        "steady_mean_ms": statistics.mean(times),
        "times_ms": times,
        "peak_mib": _jax_peak_mib(),
        "has_nan": bool(np.isnan(out_np).any()),
        "has_inf": bool(np.isinf(out_np).any()),
        "coords": out_np.tolist(),
    }


def _worker(args) -> None:
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    jax.config.update("jax_enable_x64", False)
    res = _run_one(args, args.single)
    sys.stdout.write("__RESULT__" + json.dumps(res) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("outputs/native_weights/boltz2_conf.safetensors"))
    p.add_argument("--features-npz", type=Path,
                   default=Path("outputs/real_features/1UBQ_A.npz"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--precisions", type=str, default="fp32,bf16,fp16")
    p.add_argument("--matmul-precision", type=str, default="highest",
                   choices=["highest", "high", "default"],
                   help="jax_default_matmul_precision for the --precisions rows. "
                        "'default' lets bf16/fp16 matmuls run on tensor cores in "
                        "the low dtype (real speedup); 'highest' keeps fp32 accum.")
    p.add_argument("--single", type=str, default=None,
                   help="internal: run one precision in this process")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/precision_benchmark.json"))
    args = p.parse_args()

    if args.single is not None:
        _worker(args)
        return

    precisions = [x.strip() for x in args.precisions.split(",") if x.strip()]
    for pr in precisions:
        assert pr in _DTYPES, f"unknown precision {pr!r}; pick from {list(_DTYPES)}"

    feats_f32 = load_features_npz(args.features_npz)
    n_atoms = int(feats_f32["atom_pad_mask"].shape[1])
    atom_mask_np = np.asarray(feats_f32["atom_pad_mask"]).reshape(-1) > 0.5

    # Row plan: each requested precision at the chosen matmul mode, PLUS a
    # fp32/highest reference (drift baseline + speedup denominator). Deduped,
    # baseline first. Row key = "<precision>/<matmul>".
    baseline = ("fp32", "highest")
    plan: list[tuple[str, str]] = [baseline]
    for pr in precisions:
        row = (pr, args.matmul_precision)
        if row not in plan:
            plan.append(row)

    results: dict[str, dict] = {}
    coords: dict[str, np.ndarray] = {}

    def _key(pr: str, mm: str) -> str:
        return f"{pr}/{mm}"

    for pr, mm in plan:
        key = _key(pr, mm)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--weights", str(args.weights),
               "--features-npz", str(args.features_npz),
               "--steps", str(args.steps),
               "--recycling", str(args.recycling),
               "--iters", str(args.iters),
               "--token-layers", str(args.token_layers),
               "--matmul-precision", mm,
               "--single", pr]
        print(f"[orchestrator] running {key} in subprocess (clean peak VRAM)...")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env={**os.environ})
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("__RESULT__")), None)
        if line is None:
            sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
            raise RuntimeError(f"{key} subprocess produced no result")
        r = json.loads(line[len("__RESULT__"):])
        coords[key] = np.asarray(r.pop("coords"), dtype=np.float64).reshape(-1, 3)
        results[key] = r

    # Drift vs fp32/highest reference over real (unmasked) atoms.
    ref_key = _key(*baseline)
    ref = coords[ref_key][atom_mask_np]
    base_ms = results[ref_key]["steady_mean_ms"]
    for key in results:
        cur = coords[key][atom_mask_np]
        finite = bool(np.isfinite(cur).all() and np.isfinite(ref).all())
        if key == ref_key:
            raw = aligned = mad = 0.0
        elif not finite:
            raw = aligned = mad = float("nan")
        else:
            raw = _raw_rmsd(cur, ref)
            aligned = _kabsch_rmsd(cur, ref)
            mad = float(np.max(np.abs(cur - ref)))
        results[key].update(
            raw_rmsd_vs_fp32=raw,
            aligned_rmsd_vs_fp32=aligned,
            max_abs_diff=mad,
            speedup_vs_fp32_highest=(base_ms / results[key]["steady_mean_ms"])
            if results[key]["steady_mean_ms"]
            else None,
        )

    payload = {
        "features_npz": str(args.features_npz),
        "weights": str(args.weights),
        "n_atoms": n_atoms,
        "n_real_atoms": int(atom_mask_np.sum()),
        "steps": args.steps,
        "recycling": args.recycling,
        "iters": args.iters,
        "noise_seed": 0,
        "drift_baseline": ref_key,
        "fp32_island": "weighted_rigid_align SVD (numerical necessity)",
        "rows": {_key(pr, mm): results[_key(pr, mm)] for pr, mm in plan},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Reduced-precision sampling benchmark (full graph) ===")
    print(f"n_atoms={n_atoms} real={int(atom_mask_np.sum())} steps={args.steps} "
          f"recycling={args.recycling} iters={args.iters} noise_seed=0 "
          f"baseline={ref_key}")
    hdr = (f"{'row':<14}{'steady_ms':>12}{'peak_MiB':>11}{'speedup':>9}"
           f"{'aln_RMSD':>10}{'max_abs':>10}{'finite':>8}")
    print(hdr)
    for pr, mm in plan:
        key = _key(pr, mm)
        r = results[key]
        fin = "no" if (r["has_nan"] or r["has_inf"]) else "yes"
        pk = f"{r['peak_mib']:.1f}" if r["peak_mib"] is not None else "n/a"
        sp = (f"{r['speedup_vs_fp32_highest']:.2f}x"
              if r["speedup_vs_fp32_highest"] else "-")
        print(f"{key:<14}{r['steady_mean_ms']:>12.1f}{pk:>11}{sp:>9}"
              f"{r['aligned_rmsd_vs_fp32']:>10.4f}{r['max_abs_diff']:>10.4f}"
              f"{fin:>8}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
