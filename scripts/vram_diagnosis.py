"""TRUE 3-way peak-VRAM comparison: JAX vs PyTorch Boltz-2 full-graph sampler.

Each framework runs in its OWN subprocess with XLA_PYTHON_CLIENT_PREALLOCATE=false
so that process-level nvidia-smi reflects just that framework + its CUDA context.

Three metrics per framework:
  - JAX:   device.memory_stats()['peak_bytes_in_use']  (working set, no context)
  - torch: max_memory_allocated  (live tensors)  AND  max_memory_reserved (caching pool)
  - both:  process-level nvidia-smi peak for the python PID
           (= working set + CUDA context + cudnn/cublas workspace + reservation)

The point: split the apparent JAX>torch gap into metric-definition (context /
workspace / pool) vs real working-set.

Usage:
    uv run python scripts/vram_diagnosis.py --steps 50 \
        --features-pt outputs/real_features/1UBQ_A.pt outputs/real_features/1US0_A.pt
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
# nvidia-smi process-level peak sampler (runs in-process, background thread)   #
# --------------------------------------------------------------------------- #
def _smi_used_for_pid(pid: int) -> int:
    """Return MiB used by `pid` on the GPU (0 if not present)."""
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
    """Background thread polling nvidia-smi for this PID's peak used MiB."""

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
def _run_jax(args: argparse.Namespace, features_pt: str) -> dict:
    from functools import partial

    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import (  # noqa: E402
        _jax_memory_stats,
        _load_features_pt,
        _tree_to_jax,
    )

    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
    from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    smi = SmiPeakSampler()
    smi.start()

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    jax_params = map_boltz2_graph_state_dict(
        state_cpu, num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers, token_transformer_heads=16,
    )
    feats_np, record_id = _load_features_pt(Path(features_pt))
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    jax_feats = _tree_to_jax(feats_np)

    base = partial(
        boltz2_sample_forward,
        num_sampling_steps=args.steps, recycling_steps=args.recycling,
        token_layers=args.token_layers, augmentation=False,
        alignment_reverse_diff=True, use_scan=True,
    )
    # donate_argnums=(1,) donates the `feats` pytree (largest sampler input
    # buffers: atom/token features). params (0) is reused across iters so it
    # cannot be donated. With donation the input feats buffer is freed for
    # reuse inside the graph instead of held alongside intermediates.
    sampler = jax.jit(base, donate_argnums=(1,)) if args.donate \
        else jax.jit(base)

    def call(seed):
        # When donating feats, pass a fresh copy each call (donated buffer is
        # consumed by the graph and must not be reused).
        feats_in = _tree_to_jax(feats_np) if args.donate else jax_feats
        return sampler(jax_params, feats_in, jax.random.PRNGKey(seed))[
            "sample_atom_coords"]

    call(0).block_until_ready()
    for i in range(args.iters):
        call(i + 1).block_until_ready()

    mem = _jax_memory_stats() or {}
    smi_peak = smi.stop()
    return {
        "framework": "jax", "record_id": record_id, "n_atoms": n_atoms,
        "peak_bytes_in_use_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "bytes_in_use_mib": mem.get("bytes_in_use", 0) / 1024**2,
        "smi_process_peak_mib": smi_peak,
        "donate": args.donate,
    }


# --------------------------------------------------------------------------- #
# PyTorch child                                                                #
# --------------------------------------------------------------------------- #
def _run_torch(args: argparse.Namespace, features_pt: str) -> dict:
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_boltz2_graph import (  # noqa: E402
        _load_features_pt,
        _load_torch_graph,
        _tree_to_torch,
    )
    from compare_sampling_rmsd import _torch_sample  # noqa: E402

    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict

    smi = SmiPeakSampler()
    smi.start()

    device = "cuda"
    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_model = _load_torch_graph(
        state_cpu, args.msa_layers, args.pairformer_layers,
        args.token_layers, device)
    feats_np, record_id = _load_features_pt(Path(features_pt))
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    torch_feats = _tree_to_torch(feats_np, device)
    rng = np.random.default_rng(0)

    def one():
        init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
        step_noises = rng.standard_normal(
            (args.steps, 1, n_atoms, 3)).astype(np.float32)
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
        "framework": "torch", "record_id": record_id, "n_atoms": n_atoms,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "smi_process_peak_mib": smi_peak,
    }


# --------------------------------------------------------------------------- #
# Parent: dispatch child subprocesses and tabulate                             #
# --------------------------------------------------------------------------- #
def _child_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--features-pt", type=str, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--msa-layers", type=int, default=4)
    p.add_argument("--pairformer-layers", type=int, default=64)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--framework", choices=["jax", "torch"], required=True)
    p.add_argument("--donate", action="store_true")
    p.add_argument("--result-fd", type=int, required=True)
    a = p.parse_args(sys.argv[2:])
    res = _run_jax(a, a.features_pt) if a.framework == "jax" \
        else _run_torch(a, a.features_pt)
    os.write(a.result_fd, json.dumps(res).encode())


def _spawn(framework: str, features_pt: str, args: argparse.Namespace,
           donate: bool = False) -> dict:
    r, w = os.pipe()
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    cmd = [sys.executable, __file__, "--child",
           "--framework", framework, "--features-pt", features_pt,
           "--steps", str(args.steps), "--recycling", str(args.recycling),
           "--iters", str(args.iters), "--checkpoint", str(args.checkpoint),
           "--result-fd", str(w)]
    if donate:
        cmd.append("--donate")
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
    p.add_argument("--features-pt", nargs="+", type=str,
                   default=["outputs/real_features/1UBQ_A.pt",
                            "outputs/real_features/1US0_A.pt"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--donate", action="store_true",
                   help="Also measure JAX with donate_inputs=True.")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/vram_diagnosis.json"))
    args = p.parse_args()

    rows = []
    for fp in args.features_pt:
        jx = _spawn("jax", fp, args)
        to = _spawn("torch", fp, args)
        entry = {"features_pt": fp, "jax": jx, "torch": to}
        if args.donate:
            entry["jax_donate"] = _spawn("jax", fp, args, donate=True)
        rows.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n=== TRUE 3-way peak VRAM (steps={args.steps} "
          f"recycling={args.recycling}) ===")
    hdr = (f"{'record':<10}{'n_atoms':>8}  "
           f"{'JAX peak_in_use':>16}{'JAX smi':>10}  "
           f"{'T alloc':>10}{'T reserved':>12}{'T smi':>10}")
    print(hdr)
    for e in rows:
        jx, to = e["jax"], e["torch"]
        print(f"{jx['record_id'] or '?':<10}{jx['n_atoms']:>8}  "
              f"{jx['peak_bytes_in_use_mib']:>16.0f}"
              f"{jx['smi_process_peak_mib']:>10.0f}  "
              f"{to['max_memory_allocated_mib']:>10.0f}"
              f"{to['max_memory_reserved_mib']:>12.0f}"
              f"{to['smi_process_peak_mib']:>10.0f}")
        if "jax_donate" in e:
            jd = e["jax_donate"]
            print(f"{'  +donate':<10}{'':>8}  "
                  f"{jd['peak_bytes_in_use_mib']:>16.0f}"
                  f"{jd['smi_process_peak_mib']:>10.0f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main()
    else:
        main()
