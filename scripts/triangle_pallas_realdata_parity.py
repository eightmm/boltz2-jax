"""Real-data end-to-end parity: triangle_backend xla vs pallas.

Runs the REAL 1UBQ_A features (outputs/real_features/1UBQ_A.npz) through the
real JAX sampler (boltz2_sample_forward) twice with IDENTICAL fixed injected
init/step noise (seed 0): once with triangle_backend="xla", once with
"pallas". Each backend runs in its own fresh subprocess for clean JAX state.
The parent compares the final sample_atom_coords over real (unmasked) atoms:
raw RMSD, Kabsch-aligned RMSD (Angstrom), max abs diff.

Fixed config: augmentation=False, alignment_reverse_diff=True, recycling=3,
use_scan=True, fp32, matmul_precision="highest", multiplicity=1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FEATS = REPO / "outputs" / "real_features" / "1UBQ_A.npz"
WEIGHTS = REPO / "outputs" / "native_weights" / "boltz2_conf.safetensors"


def _run_child(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from boltz_jax.bridge.native import load_features_npz, load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    feats = load_features_npz(args.feats)
    params = load_params(args.weights)
    atoms = int(feats["atom_pad_mask"].shape[1])

    # IDENTICAL injected noise across backends -> only the triangle kernel
    # differs between xla and pallas runs.
    rng = np.random.default_rng(args.seed)
    init_noise = jnp.asarray(
        rng.standard_normal((args.multiplicity, atoms, 3)).astype(np.float32)
    )
    step_noises = jnp.asarray(
        rng.standard_normal((args.steps, args.multiplicity, atoms, 3)).astype(
            np.float32
        )
    )

    # Confirm the pallas kernel is actually imported/exercised.
    kernel_imported = False
    if args.triangle_backend == "pallas":
        from boltz_jax.models.triangle import triangle_attention as _ta

        kernel_imported = hasattr(_ta, "triangle_attention_pallas")

    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            multiplicity=args.multiplicity,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
            chunk_size=args.chunk_size,
            matmul_precision=args.matmul_precision,
            triangle_backend=args.triangle_backend,
            compute_dtype=jnp.float32,
            init_noise=init_noise,
            step_noises=step_noises,
        )
    )

    start = time.perf_counter()
    out = sampler(params, feats, jax.random.PRNGKey(args.seed))[
        "sample_atom_coords"
    ].block_until_ready()
    compile_plus_first_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    out = sampler(params, feats, jax.random.PRNGKey(args.seed))[
        "sample_atom_coords"
    ].block_until_ready()
    steady_ms = (time.perf_counter() - start) * 1000.0

    out_np = np.asarray(out).astype(np.float32)
    coords_path = Path(args.coords_out)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, out_np)
    return {
        "ok": True,
        "triangle_backend": args.triangle_backend,
        "atoms": atoms,
        "real_atoms": int(np.asarray(feats["atom_pad_mask"]).sum()),
        "compile_plus_first_ms": compile_plus_first_ms,
        "steady_ms": steady_ms,
        "finite": bool(np.isfinite(out_np).all()),
        "kernel_imported": kernel_imported,
        "coords_path": str(coords_path),
    }


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument("--triangle-backend", choices=("xla", "pallas"), required=True)
    parser.add_argument("--coords-out", type=str, required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])
    try:
        result = _run_child(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "triangle_backend": args.triangle_backend,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    os.write(args.result_fd, json.dumps(result).encode())


def _spawn(backend: str, args: argparse.Namespace) -> dict:
    coords_out = REPO / "outputs" / f"_realparity_coords_{backend}.npy"
    read_fd, write_fd = os.pipe()
    env = dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_ALLOCATOR="platform",
    )
    cmd = [
        sys.executable, __file__, "--child",
        "--feats", str(args.feats),
        "--weights", str(args.weights),
        "--steps", str(args.steps),
        "--recycling", str(args.recycling),
        "--chunk-size", str(args.chunk_size),
        "--matmul-precision", args.matmul_precision,
        "--multiplicity", str(args.multiplicity),
        "--seed", str(args.seed),
        "--triangle-backend", backend,
        "--coords-out", str(coords_out),
        "--result-fd", str(write_fd),
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env, pass_fds=(write_fd,),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    os.close(write_fd)
    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0 or not chunks:
        return {
            "ok": False,
            "triangle_backend": backend,
            "error": f"child failed rc={proc.returncode}",
            "stderr_tail": stderr[-3000:],
        }
    result = json.loads(b"".join(chunks).decode())
    result["stderr_tail"] = stderr[-1500:]
    return result


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feats", type=Path, default=FEATS)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "default"),
        default="highest",
    )
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/triangle_pallas_realdata_parity.json"),
    )
    args = parser.parse_args()

    rows = {}
    for backend in ("xla", "pallas"):
        print(f"[run] backend={backend} steps={args.steps}", flush=True)
        row = _spawn(backend, args)
        rows[backend] = row
        if row.get("ok"):
            print(
                f"  ok steady={row['steady_ms']:.0f}ms finite={row['finite']} "
                f"compile+first={row['compile_plus_first_ms']:.0f}ms "
                f"kernel_imported={row.get('kernel_imported')}",
                flush=True,
            )
        else:
            print(f"  FAILED: {row.get('error')}", flush=True)
            if row.get("stderr_tail"):
                print(row["stderr_tail"], flush=True)

    parity = {}
    if rows["xla"].get("ok") and rows["pallas"].get("ok"):
        a = np.load(rows["xla"]["coords_path"]).reshape(-1, 3).astype(np.float64)
        b = np.load(rows["pallas"]["coords_path"]).reshape(-1, 3).astype(np.float64)
        # Restrict to real (unmasked) atoms.
        feats = np.load(args.feats)
        mask = feats["atom_pad_mask"].reshape(-1).astype(bool)
        a, b = a[mask], b[mask]

        raw_rmsd = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
        max_abs = float(np.abs(a - b).max())
        # Kabsch
        ac, bc = a - a.mean(0), b - b.mean(0)
        h = ac.T @ bc
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        aligned_rmsd = float(np.sqrt(((ac @ rot.T - bc) ** 2).sum(1).mean()))

        if aligned_rmsd < 0.1:
            verdict = "SAFE_OPT_IN"
        elif aligned_rmsd > 0.5:
            verdict = "BROKEN_NEEDS_FIX"
        else:
            verdict = "BORDERLINE"

        parity = {
            "raw_rmsd_A": raw_rmsd,
            "aligned_rmsd_A": aligned_rmsd,
            "max_abs_diff_A": max_abs,
            "n_real_atoms": int(mask.sum()),
            "verdict": verdict,
        }
        print(
            f"[parity] raw={raw_rmsd:.6g} aligned={aligned_rmsd:.6g} "
            f"max_abs={max_abs:.6g} verdict={verdict}",
            flush=True,
        )
    else:
        parity = {"error": "one or both backends failed"}

    out = {
        "structure": "1UBQ_A",
        "config": {
            "augmentation": False,
            "alignment_reverse_diff": True,
            "recycling": args.recycling,
            "num_sampling_steps": args.steps,
            "use_scan": True,
            "compute_dtype": "float32",
            "matmul_precision": args.matmul_precision,
            "multiplicity": args.multiplicity,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
        "rows": {
            b: {k: v for k, v in r.items() if k not in ("coords_path", "stderr_tail")}
            for b, r in rows.items()
        },
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
