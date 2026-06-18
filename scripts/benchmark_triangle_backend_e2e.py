"""End-to-end benchmark: triangle_backend xla vs pallas through the full sampler.

Runs synthetic full-size Boltz feature tensors through the real JAX sampler
(boltz2_sample_forward) with native weights. Each (length, backend) runs in its
own fresh subprocess so JAX peak memory and nvidia-smi process peak are not
contaminated by other rows. The child also dumps the final sample_atom_coords to
an .npy so the parent can compute the pallas-vs-xla aligned (Kabsch) RMSD using
IDENTICAL injected init/step noise (same seed).

Config (fixed, stated): use_scan=True, augmentation=False,
alignment_reverse_diff=True, recycling=3, num_sampling_steps=20,
matmul_precision=highest, multiplicity=1, chunk_size=128, token_layers=24.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _smi_used_for_pid(pid: int) -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return int(parts[1])
    return 0


class SmiPeakSampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.pid = os.getpid()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _smi_used_for_pid(self.pid))
            time.sleep(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak = max(self.peak, _smi_used_for_pid(self.pid))
        return self.peak


def _jax_memory_stats() -> dict:
    import jax

    try:
        return dict(jax.devices()[0].memory_stats() or {})
    except Exception:  # noqa: BLE001
        return {}


def _run_child(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import _make_feats  # noqa: E402

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    tokens = args.tokens
    atoms = tokens * args.atoms_per_token
    feats_np = _make_feats(tokens, atoms, args.msa_rows)
    feats = {key: jnp.asarray(value) for key, value in feats_np.items()}
    params = load_params(args.weights)

    # IDENTICAL injected noise across backends for the same seed -> the only
    # difference between xla and pallas runs is the triangle-attention kernel.
    rng = np.random.default_rng(args.seed)
    init_noise = jnp.asarray(
        rng.standard_normal((args.multiplicity, atoms, 3)).astype(np.float32)
    )
    step_noises = jnp.asarray(
        rng.standard_normal((args.steps, args.multiplicity, atoms, 3)).astype(
            np.float32
        )
    )

    _dtypes = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            token_layers=args.token_layers,
            multiplicity=args.multiplicity,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
            chunk_size=args.chunk_size,
            matmul_precision=args.matmul_precision,
            triangle_backend=args.triangle_backend,
            token_attention_chunk=(args.token_attention_chunk or None),
            compute_dtype=_dtypes[args.compute_dtype],
            init_noise=init_noise,
            step_noises=step_noises,
        )
    )

    smi = SmiPeakSampler()
    smi.start()
    start = time.perf_counter()
    out = sampler(params, feats, jax.random.PRNGKey(args.seed))[
        "sample_atom_coords"
    ].block_until_ready()
    compile_plus_first_ms = (time.perf_counter() - start) * 1000.0

    times = []
    for index in range(args.iters):
        start = time.perf_counter()
        out = sampler(params, feats, jax.random.PRNGKey(args.seed))[
            "sample_atom_coords"
        ].block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)

    mem = _jax_memory_stats()
    smi_peak = smi.stop()
    out_np = np.asarray(out).astype(np.float32)
    # Persist deterministic final coords (seed reused, no per-iter reseed) for
    # the parent's parity comparison.
    coords_path = Path(args.coords_out)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, out_np)
    return {
        "ok": True,
        "tokens": tokens,
        "atoms": atoms,
        "triangle_backend": args.triangle_backend,
        "compute_dtype": args.compute_dtype,
        "token_attention_chunk": args.token_attention_chunk,
        "compile_plus_first_ms": compile_plus_first_ms,
        "steady_mean_ms": statistics.mean(times) if times else None,
        "steady_min_ms": min(times) if times else None,
        "times_ms": times,
        "peak_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "bytes_in_use_mib": mem.get("bytes_in_use", 0) / 1024**2,
        "smi_mib": smi_peak,
        "finite": bool(np.isfinite(out_np).all()),
        "coords_path": str(coords_path),
    }


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument(
        "--triangle-backend",
        choices=("xla", "pallas", "tokamax"),
        required=True,
    )
    parser.add_argument("--coords-out", type=str, required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])
    try:
        result = _run_child(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "tokens": args.tokens,
            "atoms": args.tokens * args.atoms_per_token,
            "triangle_backend": args.triangle_backend,
            "compute_dtype": args.compute_dtype,
            "token_attention_chunk": args.token_attention_chunk,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    os.write(args.result_fd, json.dumps(result).encode())


def _spawn(tokens: int, backend: str, args: argparse.Namespace) -> dict:
    coords_out = REPO / "outputs" / f"_e2e_coords_{tokens}_{backend}.npy"
    read_fd, write_fd = os.pipe()
    env = dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_ALLOCATOR="platform",
    )
    cmd = [
        sys.executable,
        __file__,
        "--child",
        "--weights",
        str(args.weights),
        "--tokens",
        str(tokens),
        "--atoms-per-token",
        str(args.atoms_per_token),
        "--msa-rows",
        str(args.msa_rows),
        "--steps",
        str(args.steps),
        "--recycling",
        str(args.recycling),
        "--iters",
        str(args.iters),
        "--token-layers",
        str(args.token_layers),
        "--chunk-size",
        str(args.chunk_size),
        "--matmul-precision",
        args.matmul_precision,
        "--multiplicity",
        str(args.multiplicity),
        "--compute-dtype",
        args.compute_dtype,
        "--token-attention-chunk",
        str(args.token_attention_chunk),
        "--seed",
        str(args.seed),
        "--triangle-backend",
        backend,
        "--coords-out",
        str(coords_out),
        "--result-fd",
        str(write_fd),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        pass_fds=(write_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
            "tokens": tokens,
            "atoms": tokens * args.atoms_per_token,
            "triangle_backend": backend,
            "compute_dtype": args.compute_dtype,
            "token_attention_chunk": args.token_attention_chunk,
            "error": f"child failed rc={proc.returncode}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-3000:],
        }
    result = json.loads(b"".join(chunks).decode())
    result["stdout_tail"] = stdout[-500:]
    result["stderr_tail"] = stderr[-1500:]
    return result


def _kabsch_rmsd(a, b):
    """Aligned RMSD over atoms (a, b: [N, 3]) via Kabsch superposition."""
    import numpy as np

    a = a.reshape(-1, 3).astype(np.float64)
    b = b.reshape(-1, 3).astype(np.float64)
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = ac @ rot.T
    return float(np.sqrt(((a_rot - bc) ** 2).sum(1).mean()))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/native_weights/boltz2_conf.safetensors"),
    )
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--atoms-per-token", type=int, default=8)
    parser.add_argument("--msa-rows", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "default"),
        default="highest",
    )
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--token-attention-chunk", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--backends", nargs="+", default=["xla", "pallas"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/triangle_backend_e2e.json"),
    )
    args = parser.parse_args()

    rows = []
    for tokens in args.lengths:
        for backend in args.backends:
            print(
                f"[run] tokens={tokens} atoms={tokens * args.atoms_per_token} "
                f"backend={backend} dtype={args.compute_dtype} "
                f"token_chunk={args.token_attention_chunk}",
                flush=True,
            )
            row = _spawn(tokens, backend, args)
            rows.append(row)
            if row.get("ok"):
                print(
                    f"  steady={row['steady_mean_ms']:.1f} ms "
                    f"peak={row['peak_mib']:.0f} MiB smi={row['smi_mib']} MiB "
                    f"finite={row['finite']} "
                    f"compile+first={row['compile_plus_first_ms']:.0f} ms",
                    flush=True,
                )
            else:
                print(f"  failed: {row.get('error')}", flush=True)
                if row.get("stderr_tail"):
                    print(row["stderr_tail"], flush=True)

    # Per-length pallas-vs-xla aligned RMSD --------------------------------
    parity = {}
    for tokens in args.lengths:
        xla_row = next(
            (
                r
                for r in rows
                if r["tokens"] == tokens
                and r["triangle_backend"] == "xla"
                and r.get("ok")
            ),
            None,
        )
        pal_row = next(
            (
                r
                for r in rows
                if r["tokens"] == tokens
                and r["triangle_backend"] == "pallas"
                and r.get("ok")
            ),
            None,
        )
        if xla_row and pal_row:
            a = np.load(xla_row["coords_path"])
            b = np.load(pal_row["coords_path"])
            rmsd = _kabsch_rmsd(a, b)
            parity[str(tokens)] = {
                "aligned_rmsd": rmsd,
                "xla_compile_plus_first_ms": xla_row["compile_plus_first_ms"],
                "pallas_compile_plus_first_ms": pal_row["compile_plus_first_ms"],
            }
            print(f"[parity] tokens={tokens} aligned_rmsd={rmsd:.6g}", flush=True)
        else:
            parity[str(tokens)] = {
                "aligned_rmsd": None,
                "note": "one or both backends failed/OOM",
            }
            print(f"[parity] tokens={tokens} skipped (missing backend)", flush=True)

    out = {
        "config": {
            "use_scan": True,
            "augmentation": False,
            "alignment_reverse_diff": True,
            "recycling": args.recycling,
            "num_sampling_steps": args.steps,
            "compute_dtype": args.compute_dtype,
            "matmul_precision": args.matmul_precision,
            "multiplicity": args.multiplicity,
            "chunk_size": args.chunk_size,
            "token_attention_chunk": args.token_attention_chunk,
            "token_layers": args.token_layers,
            "atoms_per_token": args.atoms_per_token,
            "iters": args.iters,
        },
        "rows": [
            {
                k: v
                for k, v in r.items()
                if k not in ("times_ms", "coords_path", "stdout_tail", "stderr_tail")
            }
            for r in rows
        ],
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== triangle_backend e2e ===")
    print(
        f"{'tokens':>7}{'atoms':>7}{'backend':>8}"
        f"{'dtype':>9}{'tchunk':>8}{'steady ms':>12}"
        f"{'peak MiB':>11}{'smi MiB':>9}{'finite':>8}{'ok':>5}"
    )
    for r in rows:
        if not r.get("ok"):
            print(
                f"{r['tokens']:>7}{r['atoms']:>7}{r['triangle_backend']:>8}"
                f"{r.get('compute_dtype', args.compute_dtype):>9}"
                f"{r.get('token_attention_chunk', args.token_attention_chunk):>8}"
                f"{'ERR':>12}{'ERR':>11}{'ERR':>9}{'-':>8}{'no':>5}"
            )
            continue
        print(
            f"{r['tokens']:>7}{r['atoms']:>7}{r['triangle_backend']:>8}"
            f"{r.get('compute_dtype', args.compute_dtype):>9}"
            f"{r.get('token_attention_chunk', args.token_attention_chunk):>8}"
            f"{r['steady_mean_ms']:>12.1f}{r['peak_mib']:>11.0f}"
            f"{r['smi_mib']:>9.0f}{str(r['finite']):>8}{'yes':>5}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
