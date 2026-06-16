"""Benchmark PyTorch and JAX micro Pairformer/Structure modules."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

from boltz_jax.models.micro_modules import (
    MicroConfig,
    init_micro_inputs,
    init_micro_params,
    jax_pairformer_forward,
    jax_structure_forward,
    to_jax_tree,
    to_torch_tree,
    torch_pairformer_forward,
    torch_structure_forward,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residues", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--token-s", type=int, default=256)
    parser.add_argument("--token-z", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--atoms-per-residue", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/microbench.json"))
    args = parser.parse_args()

    torch = _import_torch()
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    jax_devices = [str(device) for device in jax.devices()]
    rows = []

    for residues in args.residues:
        config = MicroConfig(
            batch=args.batch,
            residues=residues,
            token_s=args.token_s,
            token_z=args.token_z,
            heads=args.heads,
            pairformer_blocks=args.blocks,
            structure_steps=args.steps,
            atoms_per_residue=args.atoms_per_residue,
        )
        params_np = init_micro_params(config, seed=args.seed)
        inputs_np = init_micro_inputs(config, seed=args.seed + 1)

        torch_params = to_torch_tree(params_np, torch_device)
        torch_inputs = to_torch_tree(inputs_np, torch_device)
        jax_params = to_jax_tree(params_np)
        jax_inputs = to_jax_tree(inputs_np)

        pair_jit = jax.jit(jax_pairformer_forward, donate_argnums=())
        struct_jit = jax.jit(jax_structure_forward, static_argnames=("steps",))

        def torch_pair() -> Any:
            return torch_pairformer_forward(
                torch_params["pairformer"],
                torch_inputs["s"],
                torch_inputs["z"],
                torch_inputs["mask"],
                torch_inputs["pair_mask"],
            )

        def jax_pair() -> Any:
            return pair_jit(
                jax_params["pairformer"],
                jax_inputs["s"],
                jax_inputs["z"],
                jax_inputs["mask"],
                jax_inputs["pair_mask"],
            )

        torch_s, torch_z = torch_pair()
        jax_s, jax_z = _block_until_ready(jax_pair())
        pair_rmse = _rmse(
            torch_s.detach().cpu().numpy(),
            np.asarray(jax_s),
        ) + _rmse(torch_z.detach().cpu().numpy(), np.asarray(jax_z))

        def torch_struct() -> Any:
            return torch_structure_forward(
                torch_params["structure"],
                torch_s,
                torch_z,
                torch_inputs["coords"],
                torch_inputs["atom_token"],
                config.structure_steps,
            )

        def jax_struct() -> Any:
            return struct_jit(
                jax_params["structure"],
                jax_s,
                jax_z,
                jax_inputs["coords"],
                jax_inputs["atom_token"],
                steps=config.structure_steps,
            )

        torch_coords = torch_struct()
        jax_coords = _block_until_ready(jax_struct())
        structure_rmse = _rmse(
            torch_coords.detach().cpu().numpy(),
            np.asarray(jax_coords),
        )

        rows.append(
            {
                "residues": residues,
                "atoms": config.atoms,
                "batch": config.batch,
                "token_s": config.token_s,
                "token_z": config.token_z,
                "heads": config.heads,
                "blocks": config.pairformer_blocks,
                "steps": config.structure_steps,
                "torch_device": torch_device,
                "jax_devices": jax_devices,
                "pairformer_rmse_sum": pair_rmse,
                "structure_rmse": structure_rmse,
                "torch_pairformer": _bench_torch(
                    torch_pair, torch, torch_device, args.warmup, args.iters
                ),
                "jax_pairformer": _bench_jax(jax_pair, args.warmup, args.iters),
                "torch_structure": _bench_torch(
                    torch_struct, torch, torch_device, args.warmup, args.iters
                ),
                "jax_structure": _bench_jax(jax_struct, args.warmup, args.iters),
                "jax_memory_stats": _jax_memory_stats(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def _bench_torch(
    fn: Callable[[], Any], torch: Any, device: str, warmup: int, iters: int
) -> dict[str, Any]:
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            out = fn()
            if device == "cuda":
                torch.cuda.synchronize()

        times = []
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                start.record()
                out = fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
        else:
            for _ in range(iters):
                begin = time.perf_counter()
                out = fn()
                _consume_torch(out)
                times.append((time.perf_counter() - begin) * 1000)
            peak = None
            reserved = None

    return _summary(times, peak, reserved)


def _bench_jax(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, Any]:
    compile_begin = time.perf_counter()
    _block_until_ready(fn())
    compile_ms = (time.perf_counter() - compile_begin) * 1000

    for _ in range(warmup):
        _block_until_ready(fn())

    times = []
    for _ in range(iters):
        begin = time.perf_counter()
        _block_until_ready(fn())
        times.append((time.perf_counter() - begin) * 1000)
    result = _summary(times, None, None)
    result["first_call_compile_plus_run_ms"] = compile_ms
    return result


def _summary(
    times: list[float], peak: int | None, reserved: int | None
) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
    }


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _consume_torch(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _consume_torch(item)
        return
    if hasattr(value, "sum"):
        float(value.sum())


def _jax_memory_stats() -> dict[str, Any]:
    device = jax.devices()[0]
    stats_fn = getattr(device, "memory_stats", None)
    if stats_fn is None:
        return {}
    return stats_fn() or {}


def _rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    return float(np.sqrt(np.mean(delta**2)))


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = "Install torch support with: uv sync --extra dev --extra torch-bridge"
        raise SystemExit(msg) from exc
    return torch


if __name__ == "__main__":
    main()
