"""Benchmark checkpoint-compatible Boltz diffusion score forward."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_diffusion_score_model_state_dict
from boltz_jax.models.atom import get_indexing_matrix, single_to_keys
from boltz_jax.models.diffusion import diffusion_score_model_forward

PREFIX = "structure_module.score_model"
BOLTZ_SRC = Path("../boltz/src")


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
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diffusion_score_bench.json"),
    )
    args = parser.parse_args()

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _load_torch_score_model(state_cpu, args.token_layers, torch_device)
    jax_params = map_diffusion_score_model_state_dict(
        state_cpu,
        PREFIX,
        num_token_layers=args.token_layers,
    )
    inputs = _make_inputs(args.tokens, args.atoms, args.token_layers)
    torch_inputs = _tree_to_torch(inputs, torch_device)
    jax_inputs = _tree_to_jax(inputs)
    torch_to_keys, jax_to_keys = _to_keys_fns(args.atoms, torch_device)

    torch_conditioning = {
        "q": torch_inputs["q"],
        "c": torch_inputs["c"],
        "to_keys": torch_to_keys,
        "atom_enc_bias": torch_inputs["atom_enc_bias"],
        "atom_dec_bias": torch_inputs["atom_dec_bias"],
        "token_trans_bias": torch_inputs["token_trans_bias"],
    }
    jax_conditioning = {
        "q": jax_inputs["q"],
        "c": jax_inputs["c"],
        "to_keys": jax_to_keys,
        "atom_enc_bias": jax_inputs["atom_enc_bias"],
        "atom_dec_bias": jax_inputs["atom_dec_bias"],
        "token_trans_bias": jax_inputs["token_trans_bias"],
    }

    def torch_fn() -> torch.Tensor:
        return torch_model(
            s_inputs=torch_inputs["s_inputs"],
            s_trunk=torch_inputs["s_trunk"],
            r_noisy=torch_inputs["r_noisy"],
            times=torch_inputs["times"],
            feats=torch_inputs["feats"],
            diffusion_conditioning=torch_conditioning,
            multiplicity=1,
        )

    jax_fn = jax.jit(
        lambda s_inputs, s_trunk, r_noisy, times: diffusion_score_model_forward(
            jax_params,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            times=times,
            feats=jax_inputs["feats"],
            diffusion_conditioning=jax_conditioning,
            multiplicity=1,
        )
    )

    with torch.no_grad():
        torch_out = torch_fn()
    jax_out = _block_until_ready(
        jax_fn(
            jax_inputs["s_inputs"],
            jax_inputs["s_trunk"],
            jax_inputs["r_noisy"],
            jax_inputs["times"],
        )
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "tokens": args.tokens,
        "atoms": args.atoms,
        "token_layers": args.token_layers,
        "torch_device": torch_device,
        "jax_devices": [str(device) for device in jax.devices()],
        "rmse": _rmse(torch_out.detach().cpu().numpy(), np.asarray(jax_out)),
        "torch": _bench_torch(torch_fn, torch_device, args.warmup, args.iters),
        "jax": _bench_jax(
            lambda: jax_fn(
                jax_inputs["s_inputs"],
                jax_inputs["s_trunk"],
                jax_inputs["r_noisy"],
                jax_inputs["times"],
            ),
            args.warmup,
            args.iters,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


class _ScoreModel(torch.nn.Module):
    def __init__(self, token_layers: int) -> None:
        super().__init__()
        sys.path.insert(0, str(BOLTZ_SRC))
        from boltz.model.modules.encodersv2 import (  # noqa: PLC0415
            AtomAttentionDecoder,
            AtomAttentionEncoder,
            SingleConditioning,
        )
        from boltz.model.modules.transformersv2 import (  # noqa: PLC0415
            DiffusionTransformer,
        )
        from boltz.model.modules.utils import LinearNoBias  # noqa: PLC0415

        self.single_conditioner = SingleConditioning(16, 384, 256, 2)
        self.atom_attention_encoder = AtomAttentionEncoder(128, 384, 32, 128)
        self.s_to_a_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            LinearNoBias(768, 768),
        )
        self.token_transformer = DiffusionTransformer(
            dim=768,
            dim_single_cond=768,
            depth=token_layers,
            heads=8,
        )
        self.a_norm = torch.nn.LayerNorm(768)
        self.atom_attention_decoder = AtomAttentionDecoder(128, 384, 32, 128)

    def forward(
        self,
        s_inputs,
        s_trunk,
        r_noisy,
        times,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        s, _ = self.single_conditioner(
            times,
            s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs.repeat_interleave(multiplicity, 0),
        )
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            multiplicity=multiplicity,
        )
        a = a + self.s_to_a_linear(s)
        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)
        return self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )


def _load_torch_score_model(
    state: dict[str, torch.Tensor],
    token_layers: int,
    device: str,
) -> torch.nn.Module:
    model = _ScoreModel(token_layers).eval().to(device)
    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local_key = key.removeprefix(f"{PREFIX}.")
        if local_key.startswith("token_transformer.layers."):
            index = int(local_key.split(".")[2])
            if index >= token_layers:
                continue
        module_state[local_key] = value.to(device=device)
    model.load_state_dict(module_state)
    return model


def _make_inputs(tokens: int, atoms: int, token_layers: int) -> dict[str, Any]:
    windows = atoms // 32
    atom_to_token = np.zeros((1, atoms, tokens), dtype=np.float32)
    atom_to_token[0, np.arange(atoms), np.arange(atoms) % tokens] = 1.0
    atom_mask = np.ones((1, atoms), dtype=np.float32)
    atom_mask[:, -3:] = 0.0
    token_mask = np.ones((1, tokens), dtype=np.float32)
    token_mask[:, -1:] = 0.0
    return {
        "s_inputs": np.linspace(-0.2, 0.2, num=tokens * 384, dtype=np.float32).reshape(
            1, tokens, 384
        ),
        "s_trunk": np.linspace(0.25, -0.25, num=tokens * 384, dtype=np.float32).reshape(
            1, tokens, 384
        ),
        "r_noisy": np.linspace(0.15, -0.15, num=atoms * 3, dtype=np.float32).reshape(
            1, atoms, 3
        ),
        "times": np.asarray([0.17], dtype=np.float32),
        "feats": {
            "ref_pos": np.linspace(-0.3, 0.3, num=atoms * 3, dtype=np.float32).reshape(
                1, atoms, 3
            ),
            "atom_pad_mask": atom_mask,
            "atom_to_token": atom_to_token,
            "token_pad_mask": token_mask,
        },
        "q": np.linspace(-0.2, 0.2, num=atoms * 128, dtype=np.float32).reshape(
            1, atoms, 128
        ),
        "c": np.linspace(0.25, -0.25, num=atoms * 128, dtype=np.float32).reshape(
            1, atoms, 128
        ),
        "atom_enc_bias": np.linspace(
            -0.1, 0.1, num=windows * 32 * 128 * 12, dtype=np.float32
        ).reshape(1, windows, 32, 128, 12),
        "atom_dec_bias": np.linspace(
            0.1, -0.1, num=windows * 32 * 128 * 12, dtype=np.float32
        ).reshape(1, windows, 32, 128, 12),
        "token_trans_bias": np.linspace(
            -0.05,
            0.05,
            num=tokens * tokens * token_layers * 8,
            dtype=np.float32,
        ).reshape(1, tokens, tokens, token_layers * 8),
    }


def _to_keys_fns(atoms: int, torch_device: str):
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (  # noqa: PLC0415
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    keys = torch_index(K=atoms // 32, W=32, H=128, device=torch_device)
    torch_fn = partial(torch_to_keys, indexing_matrix=keys, W=32, H=128)
    jax_keys = get_indexing_matrix(k=atoms // 32, w=32, h_keys=128)
    return torch_fn, lambda x: single_to_keys(x, jax_keys, w=32, h_keys=128)


def _tree_to_torch(value: Any, device: str) -> Any:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    if isinstance(value, dict):
        return {key: _tree_to_torch(item, device) for key, item in value.items()}
    return value


def _tree_to_jax(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, dict):
        return {key: _tree_to_jax(item) for key, item in value.items()}
    return value


def _bench_torch(fn, device: str, warmup: int, iters: int) -> dict[str, Any]:
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


def _bench_jax(fn, warmup: int, iters: int) -> dict[str, Any]:
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


def _summary(times, peak_allocated, peak_reserved) -> dict[str, Any]:
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
