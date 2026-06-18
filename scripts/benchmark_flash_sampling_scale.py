"""Scale benchmark for JAX sampling attention backends.

Runs synthetic full-size Boltz feature tensors through the real JAX sampler with
native weights. Each row runs in a fresh subprocess so JAX peak memory is not
contaminated by previous rows.
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


def _configure_jax_compilation_cache(jax_module, args: argparse.Namespace) -> None:
    """Enable JAX persistent compilation cache for repeated fixed-shape probes."""

    if args.compilation_cache_dir is None:
        return
    cache_dir = args.compilation_cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax_module.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax_module.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        args.cache_min_compile_time_secs,
    )
    jax_module.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        args.cache_min_entry_size_bytes,
    )


def _effective_chunk_record(tokens: int, args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from boltz_jax.models.trunk_blocks.trunk import resolve_long_sequence_chunks

    chunks = resolve_long_sequence_chunks(
        tokens,
        chunk_size=args.chunk_size,
        triangle_attention_chunk=(args.triangle_attention_chunk or None),
        triangle_attention_q_chunk=(args.triangle_attention_q_chunk or None),
        token_attention_chunk=(args.token_attention_chunk or None),
    )
    return {
        "effective_chunk_size": chunks["chunk_size"],
        "effective_token_attention_chunk": chunks["token_attention_chunk"],
        "effective_triangle_attention_chunk": chunks["triangle_attention_chunk"],
        "effective_triangle_attention_q_chunk": chunks["triangle_attention_q_chunk"],
    }


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
    _configure_jax_compilation_cache(jax, args)

    tokens = args.tokens
    atoms = tokens * args.atoms_per_token
    feats_np = _make_feats(tokens, atoms, args.msa_rows)
    feats = {key: jnp.asarray(value) for key, value in feats_np.items()}
    params = load_params(args.weights)
    effective_chunks = _effective_chunk_record(tokens, args)

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
    trunk_use_scan = _scan_arg(args.trunk_use_scan)
    score_use_scan = _scan_arg(args.score_use_scan)
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
            trunk_use_scan=trunk_use_scan,
            score_use_scan=score_use_scan,
            chunk_size=args.chunk_size,
            triangle_attention_chunk=(args.triangle_attention_chunk or None),
            triangle_attention_q_chunk=(args.triangle_attention_q_chunk or None),
            transition_hidden_chunk=(args.transition_hidden_chunk or None),
            matmul_precision=args.matmul_precision,
            attention_backend=args.attention_backend,
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
        out = sampler(params, feats, jax.random.PRNGKey(args.seed + index + 1))[
            "sample_atom_coords"
        ].block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)

    mem = _jax_memory_stats()
    smi_peak = smi.stop()
    out_np = np.asarray(out)
    return {
        "ok": True,
        "tokens": tokens,
        "atoms": atoms,
        "multiplicity": args.multiplicity,
        "steps": args.steps,
        "recycling": args.recycling,
        "attention_backend": args.attention_backend,
        "triangle_backend": args.triangle_backend,
        "compute_dtype": args.compute_dtype,
        "token_attention_chunk": args.token_attention_chunk,
        "triangle_attention_chunk": args.triangle_attention_chunk,
        "triangle_attention_q_chunk": args.triangle_attention_q_chunk,
        "transition_hidden_chunk": args.transition_hidden_chunk,
        **effective_chunks,
        "trunk_use_scan": args.trunk_use_scan,
        "score_use_scan": args.score_use_scan,
        "compilation_cache_dir": (
            None
            if args.compilation_cache_dir is None
            else str(args.compilation_cache_dir)
        ),
        "cache_min_compile_time_secs": args.cache_min_compile_time_secs,
        "cache_min_entry_size_bytes": args.cache_min_entry_size_bytes,
        "compile_plus_first_ms": compile_plus_first_ms,
        "steady_mean_ms": statistics.mean(times) if times else None,
        "steady_min_ms": min(times) if times else None,
        "times_ms": times,
        "peak_bytes_in_use_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "bytes_in_use_mib": mem.get("bytes_in_use", 0) / 1024**2,
        "smi_process_peak_mib": smi_peak,
        "has_nan": bool(np.isnan(out_np).any()),
        "has_inf": bool(np.isinf(out_np).any()),
    }


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument("--result-fd", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])
    try:
        result = _run_child(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "tokens": args.tokens,
            "atoms": args.tokens * args.atoms_per_token,
            "multiplicity": args.multiplicity,
            "steps": args.steps,
            "recycling": args.recycling,
            "attention_backend": args.attention_backend,
            "triangle_backend": args.triangle_backend,
            "compute_dtype": args.compute_dtype,
            "token_attention_chunk": args.token_attention_chunk,
            "triangle_attention_chunk": args.triangle_attention_chunk,
            "triangle_attention_q_chunk": args.triangle_attention_q_chunk,
            "transition_hidden_chunk": args.transition_hidden_chunk,
            **_effective_chunk_record(args.tokens, args),
            "trunk_use_scan": args.trunk_use_scan,
            "score_use_scan": args.score_use_scan,
            "compilation_cache_dir": (
                None
                if args.compilation_cache_dir is None
                else str(args.compilation_cache_dir)
            ),
            "cache_min_compile_time_secs": args.cache_min_compile_time_secs,
            "cache_min_entry_size_bytes": args.cache_min_entry_size_bytes,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    os.write(args.result_fd, json.dumps(result).encode())


def _spawn(
    tokens: int, multiplicity: int, backend: str, args: argparse.Namespace
) -> dict:
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
        "--triangle-attention-chunk",
        str(args.triangle_attention_chunk),
        "--triangle-attention-q-chunk",
        str(args.triangle_attention_q_chunk),
        "--transition-hidden-chunk",
        str(args.transition_hidden_chunk),
        "--matmul-precision",
        args.matmul_precision,
        "--multiplicity",
        str(multiplicity),
        "--attention-backend",
        backend,
        "--triangle-backend",
        args.triangle_backend,
        "--compute-dtype",
        args.compute_dtype,
        "--token-attention-chunk",
        str(args.token_attention_chunk),
        "--trunk-use-scan",
        args.trunk_use_scan,
        "--score-use-scan",
        args.score_use_scan,
        "--cache-min-compile-time-secs",
        str(args.cache_min_compile_time_secs),
        "--cache-min-entry-size-bytes",
        str(args.cache_min_entry_size_bytes),
        "--seed",
        str(args.seed),
        "--result-fd",
        str(write_fd),
    ]
    if args.compilation_cache_dir is not None:
        cmd.extend(["--compilation-cache-dir", str(args.compilation_cache_dir)])
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
            "multiplicity": multiplicity,
            "steps": args.steps,
            "recycling": args.recycling,
            "attention_backend": backend,
            "triangle_backend": args.triangle_backend,
            "compute_dtype": args.compute_dtype,
            "token_attention_chunk": args.token_attention_chunk,
            "triangle_attention_chunk": args.triangle_attention_chunk,
            "triangle_attention_q_chunk": args.triangle_attention_q_chunk,
            "transition_hidden_chunk": args.transition_hidden_chunk,
            **_effective_chunk_record(tokens, args),
            "trunk_use_scan": args.trunk_use_scan,
            "score_use_scan": args.score_use_scan,
            "compilation_cache_dir": (
                None
                if args.compilation_cache_dir is None
                else str(args.compilation_cache_dir)
            ),
            "cache_min_compile_time_secs": args.cache_min_compile_time_secs,
            "cache_min_entry_size_bytes": args.cache_min_entry_size_bytes,
            "error": f"child failed rc={proc.returncode}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-2000:],
        }
    result = json.loads(b"".join(chunks).decode())
    result["stdout_tail"] = stdout[-1000:]
    result["stderr_tail"] = stderr[-2000:]
    return result


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/native_weights/boltz2_conf.safetensors"),
    )
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--atoms-per-token", type=int, default=8)
    parser.add_argument("--msa-rows", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--triangle-attention-chunk", type=int, default=0)
    parser.add_argument("--triangle-attention-q-chunk", type=int, default=0)
    parser.add_argument("--transition-hidden-chunk", type=int, default=0)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "default"),
        default="highest",
    )
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--attention-backend", choices=("xla", "flash"), default="xla")
    parser.add_argument("--triangle-backend", choices=("xla", "pallas"), default="xla")
    parser.add_argument("--token-attention-chunk", type=int, default=0)
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=None,
        help="Enable JAX persistent compilation cache in each child process.",
    )
    parser.add_argument("--cache-min-compile-time-secs", type=float, default=1.0)
    parser.add_argument("--cache-min-entry-size-bytes", type=int, default=0)
    parser.add_argument(
        "--trunk-use-scan",
        choices=("auto", "true", "false"),
        default="auto",
        help="Override boltz2_sample_forward trunk_use_scan.",
    )
    parser.add_argument(
        "--score-use-scan",
        choices=("auto", "true", "false"),
        default="auto",
        help="Override boltz2_sample_forward score_use_scan.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument("--lengths", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--multiplicities", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--backends", nargs="+", default=["xla", "flash"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/flash_sampling_scale.json"),
    )
    args = parser.parse_args()

    rows = []
    for tokens in args.lengths:
        for multiplicity in args.multiplicities:
            for backend in args.backends:
                print(
                    f"[run] tokens={tokens} atoms={tokens * args.atoms_per_token} "
                    f"multiplicity={multiplicity} backend={backend} "
                    f"triangle_backend={args.triangle_backend} "
                    f"dtype={args.compute_dtype} "
                    f"token_chunk={args.token_attention_chunk} "
                    f"tri_att_chunk={args.triangle_attention_chunk} "
                    f"tri_att_q_chunk={args.triangle_attention_q_chunk} "
                    f"trans_h_chunk={args.transition_hidden_chunk} "
                    f"score_scan={args.score_use_scan}",
                    flush=True,
                )
                row = _spawn(tokens, multiplicity, backend, args)
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                if row.get("ok"):
                    steady = row.get("steady_mean_ms")
                    steady_s = "n/a" if steady is None else f"{steady:.1f} ms"
                    print(
                        f"  steady={steady_s} "
                        f"peak={row['peak_bytes_in_use_mib']:.0f} MiB "
                        f"smi={row['smi_process_peak_mib']} MiB",
                        flush=True,
                    )
                else:
                    print(f"  failed: {row.get('error')}", flush=True)

    print("\n=== flash sampling scale ===")
    print(
        f"{'tokens':>7}{'atoms':>7}{'mult':>6}{'backend':>8}"
        f"{'tri':>8}{'dtype':>9}{'tchunk':>8}{'achunk':>8}{'qchunk':>8}"
        f"{'hchunk':>8}{'score':>7}{'steady ms':>12}"
        f"{'peak MiB':>11}{'smi MiB':>9}{'ok':>5}"
    )
    for row in rows:
        tri_chunk = row.get(
            "effective_triangle_attention_chunk",
            row.get("triangle_attention_chunk", args.triangle_attention_chunk),
        )
        tri_q_chunk = row.get(
            "effective_triangle_attention_q_chunk",
            row.get("triangle_attention_q_chunk", args.triangle_attention_q_chunk),
        )
        trans_h_chunk = row.get(
            "transition_hidden_chunk", args.transition_hidden_chunk
        )
        token_chunk = row.get(
            "effective_token_attention_chunk",
            row.get("token_attention_chunk", args.token_attention_chunk),
        )
        if not row.get("ok"):
            print(
                f"{row['tokens']:>7}{row['atoms']:>7}"
                f"{row['multiplicity']:>6}{row['attention_backend']:>8}"
                f"{row.get('triangle_backend', args.triangle_backend):>8}"
                f"{row.get('compute_dtype', args.compute_dtype):>9}"
                f"{token_chunk or 0:>8}"
                f"{tri_chunk or 0:>8}"
                f"{tri_q_chunk or 0:>8}"
                f"{trans_h_chunk:>8}"
                f"{row.get('score_use_scan', args.score_use_scan):>7}"
                f"{'ERR':>12}{'ERR':>11}{'ERR':>9}{'no':>5}"
            )
            continue
        print(
            f"{row['tokens']:>7}{row['atoms']:>7}"
            f"{row['multiplicity']:>6}{row['attention_backend']:>8}"
            f"{row.get('triangle_backend', args.triangle_backend):>8}"
            f"{row.get('compute_dtype', args.compute_dtype):>9}"
            f"{token_chunk or 0:>8}"
            f"{tri_chunk or 0:>8}"
            f"{tri_q_chunk or 0:>8}"
            f"{trans_h_chunk:>8}"
            f"{row.get('score_use_scan', args.score_use_scan):>7}"
            f"{_format_optional_float(row['steady_mean_ms']):>12}"
            f"{row['peak_bytes_in_use_mib']:>11.0f}"
            f"{row['smi_process_peak_mib']:>9.0f}{'yes':>5}"
        )
    print(f"\nwrote {args.output}")


def _scan_arg(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
