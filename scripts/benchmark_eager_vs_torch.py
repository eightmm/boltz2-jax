"""JAX EAGER (no jit) vs PyTorch eager: Boltz-2 sampler latency + peak VRAM.

The point: measure JAX WITHOUT the monolithic ``jax.jit`` compile. In eager
mode JAX dispatches op-by-op (tiny per-op XLA kernels, no big compile, no long
wait) and frees intermediates per-op like torch eager, so it is the fairest
memory comparison and removes the compile-time question.

JAX side calls ``boltz2_sample_forward`` DIRECTLY (no ``jax.jit`` anywhere in
the call path; ``use_scan=False`` -> eager python step loop). PyTorch side
reuses the reference sampler loop from ``compare_sampling_rmsd._torch_sample``.

Each framework runs in its OWN subprocess with
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` for a clean process-level peak,
mirroring ``scripts/vram_diagnosis.py``.

Settings (both sides): num_sampling_steps=50, recycling_steps=3,
augmentation=False, alignment_reverse_diff=True, fp32, tf32 off.

Usage:
    uv run python scripts/benchmark_eager_vs_torch.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = Path("../boltz/.cache/boltz/boltz2_conf.ckpt")
NATIVE_WEIGHTS = Path("outputs/native_weights/boltz2_conf.safetensors")

FEATURES = [
    "outputs/real_features/1UBQ_A.npz",
    "outputs/real_features/1US0_A.pt",
]


# --------------------------------------------------------------------------- #
# nvidia-smi process-level peak sampler (background thread, mirrors vram_diag) #
# --------------------------------------------------------------------------- #
def _smi_used_for_pid(pid: int) -> int:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
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
        self._thr: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _smi_used_for_pid(self.pid))
            time.sleep(self.interval)

    def start(self) -> None:
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        self.peak = max(self.peak, _smi_used_for_pid(self.pid))
        return self.peak


# --------------------------------------------------------------------------- #
# Feature loading (accepts .pt or .npz)                                        #
# --------------------------------------------------------------------------- #
def _load_features(path: Path) -> tuple[dict, str | None]:
    import numpy as np

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            feats = {k: np.asarray(data[k]) for k in data.files
                     if not k.startswith("_")}
        record_id = path.stem
        return feats, record_id

    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    record_id = obj.get("_record_id")
    feats = {}
    for key, value in obj.items():
        if key.startswith("_") or not torch.is_tensor(value):
            continue
        feats[key] = value.detach().cpu().numpy()
    return feats, record_id


# --------------------------------------------------------------------------- #
# JAX EAGER child (NO jax.jit anywhere)                                        #
# --------------------------------------------------------------------------- #
def _run_jax(args: argparse.Namespace, features: str) -> dict:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import _jax_memory_stats  # noqa: E402

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    smi = SmiPeakSampler()
    smi.start()

    jax_params = load_params(args.native_weights)
    feats_np, record_id = _load_features(Path(features))
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    jax_feats = {k: jnp.asarray(v) for k, v in feats_np.items()}

    def call(seed):
        # DIRECT eager call: no jax.jit, use_scan=False (python step loop).
        out = boltz2_sample_forward(
            jax_params, jax_feats, jax.random.PRNGKey(seed),
            num_sampling_steps=args.steps, recycling_steps=args.recycling,
            token_layers=args.token_layers, augmentation=False,
            alignment_reverse_diff=True, use_scan=False,
        )
        return out["sample_atom_coords"]

    # First call (prove no monolithic compile spike: first ~= steady).
    t0 = time.perf_counter()
    call(0).block_until_ready()
    first_ms = (time.perf_counter() - t0) * 1000.0

    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        call(i + 1).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)

    mem = _jax_memory_stats() or {}
    smi_peak = smi.stop()
    steady = sum(times) / len(times)
    return {
        "framework": "jax_eager", "record_id": record_id, "n_atoms": n_atoms,
        "first_ms": first_ms, "steady_mean_ms": steady, "times_ms": times,
        "peak_bytes_in_use_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "bytes_in_use_mib": mem.get("bytes_in_use", 0) / 1024**2,
        "smi_process_peak_mib": smi_peak,
    }


# --------------------------------------------------------------------------- #
# PyTorch eager child                                                          #
# --------------------------------------------------------------------------- #
def _run_torch(args: argparse.Namespace, features: str) -> dict:
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import _load_torch_graph  # noqa: E402
    from compare_sampling_rmsd import _torch_sample  # noqa: E402

    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict

    smi = SmiPeakSampler()
    smi.start()

    device = "cuda"
    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_model = _load_torch_graph(
        state_cpu, args.msa_layers, args.pairformer_layers,
        args.token_layers, device)
    feats_np, record_id = _load_features(Path(features))
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    torch_feats = {k: torch.as_tensor(v, device=device)
                   for k, v in feats_np.items()}
    rng = np.random.default_rng(0)

    def one():
        init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
        step_noises = rng.standard_normal(
            (args.steps, 1, n_atoms, 3)).astype(np.float32)
        with torch.no_grad():
            return _torch_sample(torch_model, torch_feats, args.steps,
                                 init_noise, step_noises, device,
                                 recycling_steps=args.recycling)

    # 1 warmup, then timed iters with cuda events.
    one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        one()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    smi_peak = smi.stop()
    steady = sum(times) / len(times)
    return {
        "framework": "torch_eager", "record_id": record_id, "n_atoms": n_atoms,
        "steady_mean_ms": steady, "times_ms": times,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "smi_process_peak_mib": smi_peak,
    }


# --------------------------------------------------------------------------- #
# Child entry + subprocess dispatch                                            #
# --------------------------------------------------------------------------- #
def _child_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--native-weights", type=Path, default=NATIVE_WEIGHTS)
    p.add_argument("--features", type=str, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--framework", choices=["jax", "torch"], required=True)
    p.add_argument("--result-fd", type=int, required=True)
    a = p.parse_args(sys.argv[2:])
    res = _run_jax(a, a.features) if a.framework == "jax" \
        else _run_torch(a, a.features)
    os.write(a.result_fd, json.dumps(res).encode())


def _spawn(framework: str, features: str, args: argparse.Namespace) -> dict:
    r, w = os.pipe()
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    cmd = [sys.executable, __file__, "--child",
           "--framework", framework, "--features", features,
           "--steps", str(args.steps), "--recycling", str(args.recycling),
           "--iters", str(args.iters), "--checkpoint", str(args.checkpoint),
           "--native-weights", str(args.native_weights),
           "--result-fd", str(w)]
    proc = subprocess.Popen(cmd, env=env, pass_fds=(w,), cwd=str(REPO))
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    proc.wait()
    assert proc.returncode == 0, f"{framework} child failed rc={proc.returncode}"
    return json.loads(b"".join(chunks).decode())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--native-weights", type=Path, default=NATIVE_WEIGHTS)
    p.add_argument("--features", nargs="+", type=str, default=FEATURES)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/eager_vs_torch.json"))
    args = p.parse_args()

    rows = []
    for fp in args.features:
        jx = _spawn("jax", fp, args)
        to = _spawn("torch", fp, args)
        rows.append({"features": fp, "jax_eager": jx, "torch_eager": to})

    payload = {
        "config": {
            "steps": args.steps, "recycling": args.recycling,
            "iters": args.iters, "augmentation": False, "alignment": True,
            "dtype": "fp32", "tf32": False, "jax_mode": "EAGER (no jax.jit)",
            "use_scan": False,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== JAX EAGER vs PyTorch eager (steps={args.steps} "
          f"recycling={args.recycling} iters={args.iters}, fp32) ===")
    for e in rows:
        jx, to = e["jax_eager"], e["torch_eager"]
        rid = jx["record_id"] or "?"
        print(f"\n--- {rid}  (n_atoms={jx['n_atoms']}) ---")
        print(f"{'metric':<34}{'JAX eager':>14}{'torch eager':>14}")
        print(f"{'first-iter latency (ms)':<34}{jx['first_ms']:>14.1f}"
              f"{'-':>14}")
        print(f"{'steady latency (ms)':<34}{jx['steady_mean_ms']:>14.1f}"
              f"{to['steady_mean_ms']:>14.1f}")
        print(f"{'peak working set (MiB)':<34}"
              f"{jx['peak_bytes_in_use_mib']:>14.0f}"
              f"{to['max_memory_allocated_mib']:>14.0f}")
        print(f"{'  (torch reserved pool MiB)':<34}{'':>14}"
              f"{to['max_memory_reserved_mib']:>14.0f}")
        print(f"{'nvidia-smi process peak (MiB)':<34}"
              f"{jx['smi_process_peak_mib']:>14.0f}"
              f"{to['smi_process_peak_mib']:>14.0f}")
        lat_gap = jx["steady_mean_ms"] / to["steady_mean_ms"]
        mem_gap = (jx["peak_bytes_in_use_mib"]
                   - to["max_memory_allocated_mib"])
        fvs = jx["first_ms"] / jx["steady_mean_ms"]
        print(f"latency: JAX eager is {lat_gap:.2f}x torch eager")
        print(f"peak working-set gap (JAX - torch): {mem_gap:+.0f} MiB")
        print(f"first/steady ratio (compile check): {fvs:.2f}x")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
