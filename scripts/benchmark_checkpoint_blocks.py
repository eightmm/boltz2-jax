"""Benchmark checkpoint-compatible PyTorch/JAX Boltz subblocks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from math import pi
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as functional

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import (
    map_attention_pair_bias_state_dict,
    map_diffusion_transformer_layer_state_dict,
    map_single_conditioning_state_dict,
    map_transition_state_dict,
    map_triangle_multiplication_state_dict,
)
from boltz_jax.models.diffusion.diffusion_transformer import (
    diffusion_transformer_layer_forward,
)
from boltz_jax.models.primitives.attention import attention_pair_bias_forward
from boltz_jax.models.primitives.transition import transition_forward
from boltz_jax.models.triangle.triangle import triangle_multiplication_forward
from boltz_jax.models.trunk_blocks.conditioning import single_conditioning_forward

ATTENTION_PREFIX = "pairformer_module.layers.0.attention"
TRI_OUT_PREFIX = "pairformer_module.layers.0.tri_mul_out"
TRI_IN_PREFIX = "pairformer_module.layers.0.tri_mul_in"
TRANSITION_S_PREFIX = "pairformer_module.layers.0.transition_s"
TRANSITION_Z_PREFIX = "pairformer_module.layers.0.transition_z"
SINGLE_CONDITIONING_PREFIX = "structure_module.score_model.single_conditioner"
DIFFUSION_TRANSFORMER_LAYER_PREFIX = (
    "structure_module.score_model.token_transformer.layers.0"
)


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"),
    )
    parser.add_argument("--residues", nargs="+", type=int, default=[117, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/checkpoint_blocks_bench.json"),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    state_torch = _state_to_torch_device(state_cpu, torch_device)
    params = _jax_params(state_cpu)
    rows = []

    for residues in args.residues:
        inputs = _make_inputs(residues, args.seed)
        rows.extend(
            _benchmark_residue_size(
                residues=residues,
                inputs=inputs,
                state_torch=state_torch,
                params=params,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "torch_device": torch_device,
        "jax_devices": [str(device) for device in jax.devices()],
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _state_to_torch_device(
    state: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    prefixes = (
        ATTENTION_PREFIX,
        TRI_OUT_PREFIX,
        TRI_IN_PREFIX,
        TRANSITION_S_PREFIX,
        TRANSITION_Z_PREFIX,
        SINGLE_CONDITIONING_PREFIX,
        DIFFUSION_TRANSFORMER_LAYER_PREFIX,
    )
    return {
        key: value.to(device=device)
        for key, value in state.items()
        if key.startswith(prefixes)
    }


def _jax_params(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "attention": map_attention_pair_bias_state_dict(state, ATTENTION_PREFIX),
        "tri_out": map_triangle_multiplication_state_dict(state, TRI_OUT_PREFIX),
        "tri_in": map_triangle_multiplication_state_dict(state, TRI_IN_PREFIX),
        "transition_s": map_transition_state_dict(state, TRANSITION_S_PREFIX),
        "transition_z": map_transition_state_dict(state, TRANSITION_Z_PREFIX),
        "single_conditioning": map_single_conditioning_state_dict(
            state, SINGLE_CONDITIONING_PREFIX
        ),
        "diffusion_transformer_layer": map_diffusion_transformer_layer_state_dict(
            state, DIFFUSION_TRANSFORMER_LAYER_PREFIX, num_heads=8
        ),
    }


def _make_inputs(residues: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + residues)
    return {
        "s": rng.standard_normal((1, residues, 384), dtype=np.float32),
        "z": rng.standard_normal((1, residues, residues, 128), dtype=np.float32),
        "mask": np.ones((1, residues), dtype=np.float32),
        "pair_mask": np.ones((1, residues, residues), dtype=np.float32),
        "times": np.asarray([0.17], dtype=np.float32),
        "s_trunk": rng.standard_normal((1, residues, 384), dtype=np.float32),
        "s_inputs": rng.standard_normal((1, residues, 384), dtype=np.float32),
        "a": rng.standard_normal((1, residues, 768), dtype=np.float32),
        "single_cond": rng.standard_normal((1, residues, 768), dtype=np.float32),
        "token_bias": rng.standard_normal((1, residues, residues, 8), dtype=np.float32),
    }


def _benchmark_residue_size(
    residues: int,
    inputs: dict[str, np.ndarray],
    state_torch: dict[str, torch.Tensor],
    params: dict[str, Any],
    torch_device: str,
    warmup: int,
    iters: int,
) -> list[dict[str, Any]]:
    torch_inputs = {
        name: torch.as_tensor(value, device=torch_device)
        for name, value in inputs.items()
    }
    jax_inputs = {name: jnp.asarray(value) for name, value in inputs.items()}

    specs = [
        _BenchSpec(
            name="transition_s",
            torch_fn=lambda: _torch_transition(
                state_torch, TRANSITION_S_PREFIX, torch_inputs["s"]
            ),
            jax_fn=jax.jit(
                lambda x: transition_forward(params["transition_s"], x)
            ),
            jax_args=(jax_inputs["s"],),
        ),
        _BenchSpec(
            name="transition_z",
            torch_fn=lambda: _torch_transition(
                state_torch, TRANSITION_Z_PREFIX, torch_inputs["z"]
            ),
            jax_fn=jax.jit(
                lambda x: transition_forward(params["transition_z"], x)
            ),
            jax_args=(jax_inputs["z"],),
        ),
        _BenchSpec(
            name="attention",
            torch_fn=lambda: _torch_attention(
                state_torch,
                torch_inputs["s"],
                torch_inputs["z"],
                torch_inputs["mask"],
                torch_inputs["s"],
            ),
            jax_fn=jax.jit(
                lambda s, z, mask: attention_pair_bias_forward(
                    params["attention"], s, z, mask, k_in=s
                )
            ),
            jax_args=(jax_inputs["s"], jax_inputs["z"], jax_inputs["mask"]),
        ),
        _BenchSpec(
            name="tri_mul_out",
            torch_fn=lambda: _torch_triangle(
                state_torch,
                TRI_OUT_PREFIX,
                torch_inputs["z"],
                torch_inputs["pair_mask"],
                "outgoing",
            ),
            jax_fn=jax.jit(
                lambda z, pair_mask: triangle_multiplication_forward(
                    params["tri_out"], z, pair_mask, "outgoing"
                )
            ),
            jax_args=(jax_inputs["z"], jax_inputs["pair_mask"]),
        ),
        _BenchSpec(
            name="tri_mul_in",
            torch_fn=lambda: _torch_triangle(
                state_torch,
                TRI_IN_PREFIX,
                torch_inputs["z"],
                torch_inputs["pair_mask"],
                "incoming",
            ),
            jax_fn=jax.jit(
                lambda z, pair_mask: triangle_multiplication_forward(
                    params["tri_in"], z, pair_mask, "incoming"
                )
            ),
            jax_args=(jax_inputs["z"], jax_inputs["pair_mask"]),
        ),
        _BenchSpec(
            name="single_conditioning",
            torch_fn=lambda: _torch_single_conditioning(
                state_torch,
                torch_inputs["times"],
                torch_inputs["s_trunk"],
                torch_inputs["s_inputs"],
            )[0],
            jax_fn=jax.jit(
                lambda times, s_trunk, s_inputs: single_conditioning_forward(
                    params["single_conditioning"], times, s_trunk, s_inputs
                )[0]
            ),
            jax_args=(
                jax_inputs["times"],
                jax_inputs["s_trunk"],
                jax_inputs["s_inputs"],
            ),
        ),
        _BenchSpec(
            name="diffusion_transformer_layer",
            torch_fn=lambda: _torch_diffusion_transformer_layer(
                state_torch,
                torch_inputs["a"],
                torch_inputs["single_cond"],
                torch_inputs["token_bias"],
                torch_inputs["mask"],
            ),
            jax_fn=jax.jit(
                lambda a, s, bias, mask: diffusion_transformer_layer_forward(
                    params["diffusion_transformer_layer"], a, s, bias, mask
                )
            ),
            jax_args=(
                jax_inputs["a"],
                jax_inputs["single_cond"],
                jax_inputs["token_bias"],
                jax_inputs["mask"],
            ),
        ),
    ]

    rows = []
    for spec in specs:
        torch_out = spec.torch_fn()
        jax_out = _block_until_ready(spec.jax_fn(*spec.jax_args))
        rows.append(
            {
                "residues": residues,
                "block": spec.name,
                "rmse": _rmse(torch_out.detach().cpu().numpy(), np.asarray(jax_out)),
                "torch": _bench_torch(spec.torch_fn, torch_device, warmup, iters),
                "jax": _bench_jax(
                    lambda spec=spec: spec.jax_fn(*spec.jax_args),
                    warmup,
                    iters,
                ),
            }
        )
    return rows


class _BenchSpec:
    def __init__(
        self,
        name: str,
        torch_fn: Callable[[], torch.Tensor],
        jax_fn: Callable[..., jnp.ndarray],
        jax_args: tuple[jnp.ndarray, ...],
    ) -> None:
        self.name = name
        self.torch_fn = torch_fn
        self.jax_fn = jax_fn
        self.jax_args = jax_args


def _torch_transition(
    state: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
) -> torch.Tensor:
    x = functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=state[f"{prefix}.norm.weight"],
        bias=state[f"{prefix}.norm.bias"],
        eps=1e-5,
    )
    hidden = functional.silu(functional.linear(x, state[f"{prefix}.fc1.weight"]))
    hidden = hidden * functional.linear(x, state[f"{prefix}.fc2.weight"])
    return functional.linear(hidden, state[f"{prefix}.fc3.weight"])


def _torch_attention(
    state: dict[str, torch.Tensor],
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    k_in: torch.Tensor,
) -> torch.Tensor:
    batch = s.shape[0]
    num_heads = state[f"{ATTENTION_PREFIX}.proj_z.1.weight"].shape[0]
    head_dim = s.shape[-1] // num_heads
    q = functional.linear(
        s,
        state[f"{ATTENTION_PREFIX}.proj_q.weight"],
        state[f"{ATTENTION_PREFIX}.proj_q.bias"],
    ).reshape(batch, -1, num_heads, head_dim)
    k = functional.linear(k_in, state[f"{ATTENTION_PREFIX}.proj_k.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    v = functional.linear(k_in, state[f"{ATTENTION_PREFIX}.proj_v.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    bias = functional.layer_norm(
        z,
        normalized_shape=(z.shape[-1],),
        weight=state[f"{ATTENTION_PREFIX}.proj_z.0.weight"],
        bias=state[f"{ATTENTION_PREFIX}.proj_z.0.bias"],
        eps=1e-5,
    )
    bias = functional.linear(bias, state[f"{ATTENTION_PREFIX}.proj_z.1.weight"])
    bias = bias.permute(0, 3, 1, 2)
    gate = torch.sigmoid(
        functional.linear(s, state[f"{ATTENTION_PREFIX}.proj_g.weight"])
    )
    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (head_dim**0.5) + bias.float()
    attn = attn + (1 - mask[:, None, None].float()) * -1e6
    attn = attn.softmax(dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    out = out.reshape(batch, -1, s.shape[-1])
    return functional.linear(gate * out, state[f"{ATTENTION_PREFIX}.proj_o.weight"])


def _torch_triangle(
    state: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    mask: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    x = functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=state[f"{prefix}.norm_in.weight"],
        bias=state[f"{prefix}.norm_in.bias"],
        eps=1e-5,
    )
    x_in = x
    projected = functional.linear(x, state[f"{prefix}.p_in.weight"])
    projected = projected * torch.sigmoid(
        functional.linear(x, state[f"{prefix}.g_in.weight"])
    )
    projected = projected * mask.unsqueeze(-1)
    a, b = torch.chunk(projected.float(), 2, dim=-1)
    if direction == "outgoing":
        out = torch.einsum("bikd,bjkd->bijd", a, b)
    elif direction == "incoming":
        out = torch.einsum("bkid,bkjd->bijd", a, b)
    else:
        raise AssertionError(direction)
    out = functional.layer_norm(
        out,
        normalized_shape=(out.shape[-1],),
        weight=state[f"{prefix}.norm_out.weight"],
        bias=state[f"{prefix}.norm_out.bias"],
        eps=1e-5,
    )
    out = functional.linear(out, state[f"{prefix}.p_out.weight"])
    return out * torch.sigmoid(functional.linear(x_in, state[f"{prefix}.g_out.weight"]))


def _torch_single_conditioning(
    state: dict[str, torch.Tensor],
    times: torch.Tensor,
    s_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = SINGLE_CONDITIONING_PREFIX
    s = torch.cat((s_trunk, s_inputs), dim=-1)
    s = functional.layer_norm(
        s,
        normalized_shape=(s.shape[-1],),
        weight=state[f"{prefix}.norm_single.weight"],
        bias=state[f"{prefix}.norm_single.bias"],
        eps=1e-5,
    )
    s = functional.linear(
        s,
        state[f"{prefix}.single_embed.weight"],
        state[f"{prefix}.single_embed.bias"],
    )
    fourier = torch.cos(
        2
        * pi
        * functional.linear(
            times[:, None],
            state[f"{prefix}.fourier_embed.proj.weight"],
            state[f"{prefix}.fourier_embed.proj.bias"],
        )
    )
    normed_fourier = functional.layer_norm(
        fourier,
        normalized_shape=(fourier.shape[-1],),
        weight=state[f"{prefix}.norm_fourier.weight"],
        bias=state[f"{prefix}.norm_fourier.bias"],
        eps=1e-5,
    )
    s = s + functional.linear(
        normed_fourier,
        state[f"{prefix}.fourier_to_single.weight"],
    )[:, None, :]
    for index in range(2):
        transition_prefix = f"{prefix}.transitions.{index}"
        s = s + _torch_transition(state, transition_prefix, s)
    return s, normed_fourier


def _torch_diffusion_transformer_layer(
    state: dict[str, torch.Tensor],
    a: torch.Tensor,
    s: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prefix = DIFFUSION_TRANSFORMER_LAYER_PREFIX
    b = _torch_adaln(state, f"{prefix}.adaln", a, s)
    b = _torch_attention_no_proj_z(state, f"{prefix}.pair_bias_attn", b, bias, mask, b)
    b = torch.sigmoid(
        functional.linear(
            s,
            state[f"{prefix}.output_projection_linear.weight"],
            state[f"{prefix}.output_projection_linear.bias"],
        )
    ) * b
    a = a + b
    a = a + _torch_conditioned_transition_block(state, f"{prefix}.transition", a, s)
    return a


def _torch_adaln(
    state: dict[str, torch.Tensor],
    prefix: str,
    a: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    a = functional.layer_norm(a, normalized_shape=(a.shape[-1],), eps=1e-5)
    s = functional.layer_norm(
        s,
        normalized_shape=(s.shape[-1],),
        weight=state[f"{prefix}.s_norm.weight"],
        bias=None,
        eps=1e-5,
    )
    return torch.sigmoid(
        functional.linear(
            s,
            state[f"{prefix}.s_scale.weight"],
            state[f"{prefix}.s_scale.bias"],
        )
    ) * a + functional.linear(s, state[f"{prefix}.s_bias.weight"])


def _torch_conditioned_transition_block(
    state: dict[str, torch.Tensor],
    prefix: str,
    a: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    a = _torch_adaln(state, f"{prefix}.adaln", a, s)
    swish, gates = torch.chunk(
        functional.linear(a, state[f"{prefix}.swish_gate.0.weight"]), 2, dim=-1
    )
    b = (functional.silu(gates) * swish) * functional.linear(
        a, state[f"{prefix}.a_to_b.weight"]
    )
    out = functional.linear(b, state[f"{prefix}.b_to_a.weight"])
    gate = torch.sigmoid(
        functional.linear(
            s,
            state[f"{prefix}.output_projection.0.weight"],
            state[f"{prefix}.output_projection.0.bias"],
        )
    )
    return gate * out


def _torch_attention_no_proj_z(
    state: dict[str, torch.Tensor],
    prefix: str,
    s: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    k_in: torch.Tensor,
) -> torch.Tensor:
    batch = s.shape[0]
    num_heads = bias.shape[-1]
    head_dim = s.shape[-1] // num_heads
    q = functional.linear(
        s,
        state[f"{prefix}.proj_q.weight"],
        state[f"{prefix}.proj_q.bias"],
    ).reshape(batch, -1, num_heads, head_dim)
    k = functional.linear(k_in, state[f"{prefix}.proj_k.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    v = functional.linear(k_in, state[f"{prefix}.proj_v.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    gate = torch.sigmoid(functional.linear(s, state[f"{prefix}.proj_g.weight"]))
    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (head_dim**0.5) + bias.permute(0, 3, 1, 2).float()
    attn = attn + (1 - mask[:, None, None].float()) * -1e6
    attn = attn.softmax(dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    out = out.reshape(batch, -1, s.shape[-1])
    return functional.linear(gate * out, state[f"{prefix}.proj_o.weight"])


def _bench_torch(
    fn: Callable[[], torch.Tensor],
    device: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = fn()
            if device == "cuda":
                torch.cuda.synchronize()

        times = []
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                start.record()
                _ = fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        else:
            for _ in range(iters):
                begin = time.perf_counter()
                out = fn()
                float(out.sum())
                times.append((time.perf_counter() - begin) * 1000)
            peak_allocated = None
            peak_reserved = None
    return _summary(times, peak_allocated, peak_reserved)


def _bench_jax(
    fn: Callable[[], jnp.ndarray],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    jax.clear_caches()
    compile_begin = time.perf_counter()
    _block_until_ready(fn())
    compile_ms = (time.perf_counter() - compile_begin) * 1000
    before_stats = _jax_memory_stats()

    for _ in range(warmup):
        _block_until_ready(fn())

    times = []
    for _ in range(iters):
        begin = time.perf_counter()
        _block_until_ready(fn())
        times.append((time.perf_counter() - begin) * 1000)
    after_stats = _jax_memory_stats()
    result = _summary(
        times,
        _first_int(after_stats, ("peak_bytes_in_use", "bytes_in_use")),
        _first_int(after_stats, ("bytes_limit",)),
    )
    result["first_call_compile_plus_run_ms"] = compile_ms
    result["jax_memory_stats_before"] = before_stats
    result["jax_memory_stats_after"] = after_stats
    return result


def _summary(
    times: list[float],
    peak_allocated: int | None,
    peak_reserved: int | None,
) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_or_limit_bytes": peak_reserved,
    }


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _jax_memory_stats() -> dict[str, Any]:
    stats_fn = getattr(jax.devices()[0], "memory_stats", None)
    if stats_fn is None:
        return {}
    return stats_fn() or {}


def _first_int(stats: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, int):
            return value
    return None


def _rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    return float(np.sqrt(np.mean(delta**2)))


if __name__ == "__main__":
    main()
