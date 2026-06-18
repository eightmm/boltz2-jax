"""Peak-VRAM vs sequence-length sweep: JAX vs PyTorch Boltz-2 full-graph sampler.

For synthetic lengths (tokens), runs the SAME 50-step full-graph sample
(recycling=3, fp32, default matmul, augmentation=False, alignment on) and
measures, each framework in its OWN subprocess (XLA_PYTHON_CLIENT_PREALLOCATE=
false, platform allocator):

  - JAX:   peak_bytes_in_use (working set) + nvidia-smi process peak
  - torch: max_memory_allocated + max_memory_reserved + nvidia-smi process peak

Goal: find whether/where the working-set gap (JAX peak_in_use - torch alloc)
and the true total gap (JAX smi - torch smi) cross below zero (JAX < torch).

Optional --bf16 adds a JAX bf16 row per length (params + float feats cast to
bf16, matmul=default) to compare bf16 JAX peak vs torch fp32.

Synthetic features: _make_feats(tokens, atoms=tokens*8, msa_rows=1).

Usage:
    uv run python scripts/vram_length_sweep.py
    uv run python scripts/vram_length_sweep.py --bf16 --tokens 128 256 384
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


# --------------------------------------------------------------------------- #
# nvidia-smi process-level peak sampler (background thread, in child)          #
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
# JAX child                                                                    #
# --------------------------------------------------------------------------- #
def _run_jax(args: argparse.Namespace) -> dict:
    from functools import partial

    import jax
    import jax.numpy as jnp

    # matmul=default (do NOT force highest, per task spec).
    jax.config.update("jax_enable_x64", False)

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import _jax_memory_stats, _make_feats  # noqa: E402

    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
    from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    smi = SmiPeakSampler()
    smi.start()

    tokens = args.tokens
    atoms = tokens * 8
    feats_np = _make_feats(tokens, atoms, args.msa_rows)

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    jax_params = map_boltz2_graph_state_dict(
        state_cpu, num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers, token_transformer_heads=16,
    )

    dtype = jnp.bfloat16 if args.bf16 else jnp.float32

    def _cast_param(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return jnp.asarray(x)

    def _cast_feat(v):
        a = jnp.asarray(v)
        if args.bf16 and jnp.issubdtype(a.dtype, jnp.floating):
            return a.astype(jnp.bfloat16)
        return a

    if args.bf16:
        jax_params = jax.tree.map(_cast_param, jax_params)
    jax_feats = {k: _cast_feat(v) for k, v in feats_np.items()}

    base = partial(
        boltz2_sample_forward,
        num_sampling_steps=args.steps, recycling_steps=args.recycling,
        token_layers=args.token_layers, augmentation=False,
        alignment_reverse_diff=True, use_scan=True,
    )
    sampler = jax.jit(base)

    def call(seed):
        return sampler(jax_params, jax_feats, jax.random.PRNGKey(seed))[
            "sample_atom_coords"]

    call(0).block_until_ready()
    for i in range(args.iters):
        call(i + 1).block_until_ready()

    mem = _jax_memory_stats() or {}
    smi_peak = smi.stop()
    return {
        "framework": "jax", "tokens": tokens, "atoms": atoms,
        "bf16": args.bf16,
        "peak_bytes_in_use_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "bytes_in_use_mib": mem.get("bytes_in_use", 0) / 1024**2,
        "smi_process_peak_mib": smi_peak,
    }


# --------------------------------------------------------------------------- #
# PyTorch child                                                                #
# --------------------------------------------------------------------------- #
def _run_torch(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import (  # noqa: E402
        _load_torch_graph,
        _make_feats,
        _tree_to_torch,
    )
    from compare_sampling_rmsd import _torch_sample  # noqa: E402

    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict

    smi = SmiPeakSampler()
    smi.start()

    device = "cuda"
    tokens = args.tokens
    atoms = tokens * 8
    feats_np = _make_feats(tokens, atoms, args.msa_rows)

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_model = _load_torch_graph(
        state_cpu, args.msa_layers, args.pairformer_layers,
        args.token_layers, device)
    torch_feats = _tree_to_torch(feats_np, device)
    rng = np.random.default_rng(0)

    def one():
        init_noise = rng.standard_normal((1, atoms, 3)).astype(np.float32)
        step_noises = rng.standard_normal(
            (args.steps, 1, atoms, 3)).astype(np.float32)
        with torch.no_grad():
            return _torch_sample(torch_model, torch_feats, args.steps,
                                 init_noise, step_noises, device,
                                 recycling_steps=args.recycling)

    one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.iters):
        one()
        torch.cuda.synchronize()

    smi_peak = smi.stop()
    return {
        "framework": "torch", "tokens": tokens, "atoms": atoms,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "smi_process_peak_mib": smi_peak,
    }


# --------------------------------------------------------------------------- #
# child entry                                                                  #
# --------------------------------------------------------------------------- #
def _child_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--msa-rows", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--framework", choices=["jax", "torch"], required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--result-fd", type=int, required=True)
    a = p.parse_args(sys.argv[2:])
    res = _run_jax(a) if a.framework == "jax" else _run_torch(a)
    os.write(a.result_fd, json.dumps(res).encode())


def _spawn(framework: str, tokens: int, args: argparse.Namespace,
           bf16: bool = False) -> dict:
    r, w = os.pipe()
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    cmd = [sys.executable, __file__, "--child",
           "--framework", framework, "--tokens", str(tokens),
           "--msa-rows", str(args.msa_rows),
           "--steps", str(args.steps), "--recycling", str(args.recycling),
           "--iters", str(args.iters), "--checkpoint", str(args.checkpoint),
           "--result-fd", str(w)]
    if bf16:
        cmd.append("--bf16")
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
    if proc.returncode != 0:
        return {"framework": framework, "tokens": tokens, "bf16": bf16,
                "error": f"child failed rc={proc.returncode}"}
    return json.loads(b"".join(chunks).decode())


def _gaps(jx: dict, to: dict) -> dict:
    if "error" in jx or "error" in to:
        return {}
    return {
        "working_set_gap_mib":
            jx["peak_bytes_in_use_mib"] - to["max_memory_allocated_mib"],
        "smi_total_gap_mib":
            jx["smi_process_peak_mib"] - to["smi_process_peak_mib"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--tokens", nargs="+", type=int,
                   default=[128, 256, 384, 512, 768])
    p.add_argument("--msa-rows", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--bf16", action="store_true",
                   help="Also measure a JAX bf16 row per length.")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/vram_length_sweep.json"))
    args = p.parse_args()

    rows = []
    for tk in args.tokens:
        print(f"\n--- tokens={tk} (atoms={tk * 8}) ---", flush=True)
        jx = _spawn("jax", tk, args)
        to = _spawn("torch", tk, args)
        entry = {"tokens": tk, "atoms": tk * 8,
                 "jax_fp32": jx, "torch_fp32": to,
                 "gaps": _gaps(jx, to)}
        if args.bf16:
            entry["jax_bf16"] = _spawn("jax", tk, args, bf16=True)
        rows.append(entry)
        # incremental write so a late OOM doesn't lose earlier rows.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n=== peak VRAM vs length (steps={args.steps} "
          f"recycling={args.recycling} fp32 matmul=default) ===")
    hdr = (f"{'tokens':>7}{'atoms':>7}  "
           f"{'JAX peak':>10}{'JAX smi':>9}  "
           f"{'T alloc':>9}{'T resv':>9}{'T smi':>9}  "
           f"{'ws_gap':>9}{'smi_gap':>9}")
    print(hdr)
    for e in rows:
        jx, to, g = e["jax_fp32"], e["torch_fp32"], e["gaps"]
        if "error" in jx or "error" in to:
            print(f"{e['tokens']:>7}{e['atoms']:>7}  "
                  f"ERROR jax={'error' in jx} torch={'error' in to}")
            continue
        print(f"{e['tokens']:>7}{e['atoms']:>7}  "
              f"{jx['peak_bytes_in_use_mib']:>10.0f}"
              f"{jx['smi_process_peak_mib']:>9.0f}  "
              f"{to['max_memory_allocated_mib']:>9.0f}"
              f"{to['max_memory_reserved_mib']:>9.0f}"
              f"{to['smi_process_peak_mib']:>9.0f}  "
              f"{g['working_set_gap_mib']:>+9.0f}"
              f"{g['smi_total_gap_mib']:>+9.0f}")
        if "jax_bf16" in e and "error" not in e["jax_bf16"]:
            jb = e["jax_bf16"]
            print(f"{'  bf16':>7}{'':>7}  "
                  f"{jb['peak_bytes_in_use_mib']:>10.0f}"
                  f"{jb['smi_process_peak_mib']:>9.0f}")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
