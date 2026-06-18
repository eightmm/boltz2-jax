"""Length x precision sweep for the JAX Boltz-2 full-graph sampler.

Sweeps token length {64,128,256,512} (atoms = tokens*8) x precision/matmul rows
{fp32/highest, fp32/default, bf16/default, fp16/default} on SYNTHETIC features
(reuses ``benchmark_boltz2_graph._make_feats``) with native params (torch-free).

Per (length, row) it measures:
  * steady-state mean latency (--iters, warmup 1, block_until_ready),
  * JAX peak VRAM (peak_bytes_in_use, per-row subprocess, PREALLOCATE=false),
  * drift = Kabsch-aligned RMSD + max abs diff vs the fp32/highest result AT THE
    SAME LENGTH, using identical injected init+step noise (seed 0) per length.

NOTE: synthetic feats -> RMSD is a NUMERICAL drift proxy, not biological.
Each row runs in its own subprocess for a clean per-row peak VRAM. OOM/NaN in a
row is recorded as a failed row and the sweep CONTINUES.

Mirrors scripts/benchmark_precision.py (cast policy, fp32-island align, noise
injection) but parameterizes length and runs the full grid.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.native import load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import _make_feats  # noqa: E402
from compare_sampling_rmsd import _kabsch_rmsd, _raw_rmsd  # noqa: E402

_DTYPES = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}

# (precision, matmul_precision). Baseline first (drift ref + speedup denom).
_ROWS: list[tuple[str, str]] = [
    ("fp32", "highest"),
    ("fp32", "default"),
    ("bf16", "default"),
    ("fp16", "default"),
]


def _key(pr: str, mm: str) -> str:
    return f"{pr}/{mm}"


def _jax_peak_mib() -> float | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use", 0) / (1024**2)


def _cast_floating(x, dtype):
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(dtype)
    return x


def _cast_params(params, dtype):
    return jax.tree.map(lambda x: _cast_floating(x, dtype), params)


def _cast_feats(feats, dtype):
    return {k: _cast_floating(jnp.asarray(v), dtype) for k, v in feats.items()}


def _run_one(args, pr: str) -> dict:
    """Run a single (length, precision) in THIS process. Peak VRAM is
    process-cumulative, so each row runs in a fresh subprocess."""
    dtype = _DTYPES[pr]
    tokens = args.tokens
    atoms = tokens * args.atoms_per_token
    params_f32 = load_params(args.weights)
    feats_np = _make_feats(tokens, atoms, args.msa_rows)
    n_atoms = atoms

    # Same seed-0 injected noise for every row AT THIS LENGTH.
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
    feats = _cast_feats(feats_np, dtype)
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
        "tokens": tokens,
        "atoms": atoms,
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
    try:
        res = _run_one(args, args.single)
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res = {
            "ok": False,
            "dtype": args.single,
            "tokens": args.tokens,
            "atoms": args.tokens * args.atoms_per_token,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    sys.stdout.write("__RESULT__" + json.dumps(res) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=Path("outputs/native_weights/boltz2_conf.safetensors"))
    p.add_argument("--lengths", type=str, default="64,128,256,512",
                   help="comma-sep token lengths")
    p.add_argument("--atoms-per-token", type=int, default=8)
    p.add_argument("--msa-rows", type=int, default=1,
                   help="tiny MSA depth so trunk MSA cost doesn't dominate")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--matmul-precision", type=str, default="highest",
                   choices=["highest", "high", "default"],
                   help="internal: matmul mode for the single row")
    p.add_argument("--tokens", type=int, default=None,
                   help="internal: token length for the single row")
    p.add_argument("--single", type=str, default=None,
                   help="internal: run one precision in this process")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/length_precision_sweep.json"))
    args = p.parse_args()

    if args.single is not None:
        _worker(args)
        return

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    note = ("SYNTHETIC features (benchmark_boltz2_graph._make_feats); RMSD is a "
            "NUMERICAL drift proxy vs fp32/highest at same length, not biological.")

    grid: dict[int, dict[str, dict]] = {}
    failed: list[str] = []

    for tokens in lengths:
        atoms = tokens * args.atoms_per_token
        results: dict[str, dict] = {}
        coords: dict[str, np.ndarray] = {}
        for pr, mm in _ROWS:
            key = _key(pr, mm)
            tag = f"L{tokens}(a{atoms}) {key}"
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--weights", str(args.weights),
                   "--tokens", str(tokens),
                   "--atoms-per-token", str(args.atoms_per_token),
                   "--msa-rows", str(args.msa_rows),
                   "--steps", str(args.steps),
                   "--recycling", str(args.recycling),
                   "--iters", str(args.iters),
                   "--token-layers", str(args.token_layers),
                   "--matmul-precision", mm,
                   "--single", pr]
            print(f"[orchestrator] running {tag} in subprocess...", flush=True)
            env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = next((ln for ln in proc.stdout.splitlines()
                         if ln.startswith("__RESULT__")), None)
            if line is None:
                # Hard crash (e.g. OOM kills process). Record + continue.
                tail = (proc.stderr or proc.stdout)[-600:]
                print(f"[orchestrator] {tag} FAILED (no result): {tail}", flush=True)
                results[key] = {"ok": False, "dtype": pr, "tokens": tokens,
                                "atoms": atoms, "error": "subprocess crash/OOM",
                                "stderr_tail": tail}
                coords[key] = None
                failed.append(f"L{tokens} {key}")
                continue
            r = json.loads(line[len("__RESULT__"):])
            if not r.get("ok"):
                print(f"[orchestrator] {tag} FAILED: {r.get('error')}", flush=True)
                coords[key] = None
                results[key] = r
                failed.append(f"L{tokens} {key}")
                continue
            c = r.pop("coords")
            coords[key] = np.asarray(c, dtype=np.float64).reshape(-1, 3)
            results[key] = r

        # Drift vs fp32/highest at THIS length.
        ref_key = _key("fp32", "highest")
        ref = coords.get(ref_key)
        base_ms = (results[ref_key].get("steady_mean_ms")
                   if results.get(ref_key, {}).get("ok") else None)
        for key in results:
            r = results[key]
            if not r.get("ok"):
                continue
            cur = coords[key]
            finite = bool(cur is not None and np.isfinite(cur).all()
                          and ref is not None and np.isfinite(ref).all())
            if key == ref_key:
                raw = aligned = mad = 0.0
            elif not finite:
                raw = aligned = mad = float("nan")
            else:
                raw = _raw_rmsd(cur, ref)
                aligned = _kabsch_rmsd(cur, ref)
                mad = float(np.max(np.abs(cur - ref)))
            sm = r.get("steady_mean_ms")
            r.update(
                raw_rmsd=raw,
                aligned_rmsd=aligned,
                max_abs_diff=mad,
                finite=finite,
                speedup_vs_fp32_highest_same_length=(
                    base_ms / sm if (base_ms and sm) else None),
            )
        grid[tokens] = results

    payload = {
        "steps": args.steps,
        "recycling": args.recycling,
        "iters": args.iters,
        "atoms_per_token": args.atoms_per_token,
        "msa_rows": args.msa_rows,
        "token_layers": args.token_layers,
        "noise_seed": 0,
        "lengths": lengths,
        "rows": [_key(pr, mm) for pr, mm in _ROWS],
        "drift_baseline": "fp32/highest (same length)",
        "fp32_island": "weighted_rigid_align SVD (numerical necessity)",
        "note": note,
        "failed_rows": failed,
        "grid": {str(t): grid[t] for t in grid},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Length x precision sweep (full-graph sampler) ===")
    print(f"steps={args.steps} recycling={args.recycling} iters={args.iters} "
          f"atoms/token={args.atoms_per_token} msa_rows={args.msa_rows} "
          f"noise_seed=0")
    print(note)
    for tokens in lengths:
        atoms = tokens * args.atoms_per_token
        print(f"\n-- L={tokens} tokens / {atoms} atoms --")
        hdr = (f"{'row':<14}{'steady_ms':>12}{'peak_MiB':>11}{'speedup':>9}"
               f"{'aln_RMSD':>10}{'max_abs':>10}{'finite':>8}")
        print(hdr)
        for pr, mm in _ROWS:
            key = _key(pr, mm)
            r = grid[tokens].get(key, {})
            if not r.get("ok"):
                print(f"{key:<14}{'FAILED: ' + str(r.get('error', '?')):>60}")
                continue
            pk = f"{r['peak_mib']:.1f}" if r.get("peak_mib") is not None else "n/a"
            sp = r.get("speedup_vs_fp32_highest_same_length")
            sp = f"{sp:.2f}x" if sp else "-"
            fin = "yes" if r.get("finite") else "no"
            print(f"{key:<14}{r['steady_mean_ms']:>12.1f}{pk:>11}{sp:>9}"
                  f"{r['aligned_rmsd']:>10.4f}{r['max_abs_diff']:>10.4f}{fin:>8}")
    if failed:
        print(f"\nFAILED rows: {failed}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
