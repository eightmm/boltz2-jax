"""Benchmark non-template Boltz-2 trunk plus conditioned score graph."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk import boltz2_graph_score_forward

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
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--msa-rows", type=int, default=8)
    parser.add_argument("--msa-layers", type=int, default=4)
    parser.add_argument("--pairformer-layers", type=int, default=64)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--recycling-steps", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/boltz2_graph_bench.json"),
    )
    args = parser.parse_args()

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _load_torch_graph(
        state_cpu,
        args.msa_layers,
        args.pairformer_layers,
        args.token_layers,
        torch_device,
    )
    jax_params = map_boltz2_graph_state_dict(
        state_cpu,
        num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers,
        token_transformer_heads=16,
    )
    feats = _make_feats(args.tokens, args.atoms, args.msa_rows)
    torch_feats = _tree_to_torch(feats, torch_device)
    jax_feats = _tree_to_jax(feats)
    r_noisy_np = np.linspace(0.15, -0.15, num=args.atoms * 3, dtype=np.float32)
    r_noisy_np = r_noisy_np.reshape(1, args.atoms, 3)
    times_np = np.asarray([0.17], dtype=np.float32)
    torch_r_noisy = torch.as_tensor(r_noisy_np, device=torch_device)
    torch_times = torch.as_tensor(times_np, device=torch_device)
    jax_r_noisy = jnp.asarray(r_noisy_np)
    jax_times = jnp.asarray(times_np)

    def torch_fn() -> torch.Tensor:
        return torch_model(
            torch_feats,
            torch_r_noisy,
            torch_times,
            recycling_steps=args.recycling_steps,
        )

    jax_fn = jax.jit(
        lambda feats_arg, r_arg, t_arg: boltz2_graph_score_forward(
            jax_params,
            feats_arg,
            r_arg,
            t_arg,
            recycling_steps=args.recycling_steps,
            token_layers=args.token_layers,
            multiplicity=1,
        )
    )

    with torch.no_grad():
        torch_out = torch_fn()
    jax_out = _block_until_ready(jax_fn(jax_feats, jax_r_noisy, jax_times))

    payload = {
        "checkpoint": str(args.checkpoint),
        "tokens": args.tokens,
        "atoms": args.atoms,
        "msa_rows": args.msa_rows,
        "msa_layers": args.msa_layers,
        "pairformer_layers": args.pairformer_layers,
        "token_layers": args.token_layers,
        "recycling_steps": args.recycling_steps,
        "torch_device": torch_device,
        "jax_devices": [str(device) for device in jax.devices()],
        "rmse": _rmse(torch_out.detach().cpu().numpy(), np.asarray(jax_out)),
        "torch": _bench_torch(torch_fn, torch_device, args.warmup, args.iters),
        "jax": _bench_jax(
            lambda: jax_fn(jax_feats, jax_r_noisy, jax_times),
            args.warmup,
            args.iters,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


class _TorchGraph(torch.nn.Module):
    def __init__(
        self,
        msa_layers: int,
        pairformer_layers: int,
        token_layers: int,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(BOLTZ_SRC))
        from boltz.model.layers.pairformer import PairformerModule  # noqa: PLC0415
        from boltz.model.modules.diffusion_conditioning import (  # noqa: PLC0415
            DiffusionConditioning,
        )
        from boltz.model.modules.encodersv2 import (  # noqa: PLC0415
            RelativePositionEncoder,
        )
        from boltz.model.modules.trunkv2 import (  # noqa: PLC0415
            ContactConditioning,
            InputEmbedder,
            MSAModule,
        )

        self.input_embedder = InputEmbedder(
            atom_s=128,
            atom_z=16,
            token_s=384,
            token_z=128,
            atoms_per_window_queries=32,
            atoms_per_window_keys=128,
            atom_feature_dim=388,
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            add_method_conditioning=True,
            add_modified_flag=True,
            add_cyclic_flag=True,
            add_mol_type_feat=True,
        )
        self.s_init = torch.nn.Linear(384, 384, bias=False)
        self.z_init_1 = torch.nn.Linear(384, 128, bias=False)
        self.z_init_2 = torch.nn.Linear(384, 128, bias=False)
        self.rel_pos = RelativePositionEncoder(128)
        self.token_bonds = torch.nn.Linear(1, 128, bias=False)
        self.token_bonds_type = torch.nn.Embedding(7, 128)
        self.contact_conditioning = ContactConditioning(128, 4.0, 20.0)
        self.s_norm = torch.nn.LayerNorm(384)
        self.z_norm = torch.nn.LayerNorm(128)
        self.s_recycle = torch.nn.Linear(384, 384, bias=False)
        self.z_recycle = torch.nn.Linear(128, 128, bias=False)
        self.msa_module = MSAModule(
            msa_s=64,
            token_z=128,
            token_s=384,
            msa_blocks=msa_layers,
            msa_dropout=0.15,
            z_dropout=0.25,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            use_paired_feature=True,
        )
        self.pairformer_module = PairformerModule(
            token_s=384,
            token_z=128,
            num_blocks=pairformer_layers,
            num_heads=16,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            v2=True,
        )
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=384,
            token_z=128,
            atom_s=128,
            atom_z=16,
            atoms_per_window_queries=32,
            atoms_per_window_keys=128,
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            token_transformer_depth=token_layers,
            token_transformer_heads=16,
            atom_decoder_depth=3,
            atom_decoder_heads=4,
            atom_feature_dim=388,
        )
        self.score_model = _ScoreModel(token_layers)

    def forward(self, feats, r_noisy, times, recycling_steps=0):
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        for _ in range(recycling_steps + 1):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))
            z = z + self.msa_module(z, s_inputs, feats, use_kernels=False)
            s, z = self.pairformer_module(
                s,
                z,
                mask=mask,
                pair_mask=pair_mask,
                use_kernels=False,
            )
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.diffusion_conditioning(
                s,
                z,
                relative_position_encoding,
                feats,
            )
        )
        return self.score_model(
            s_inputs=s_inputs,
            s_trunk=s,
            r_noisy=r_noisy,
            times=times,
            feats=feats,
            diffusion_conditioning={
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            },
            multiplicity=1,
        )


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
            heads=16,
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


def _load_torch_graph(
    state: dict[str, torch.Tensor],
    msa_layers: int,
    pairformer_layers: int,
    token_layers: int,
    device: str,
) -> torch.nn.Module:
    module = _TorchGraph(msa_layers, pairformer_layers, token_layers).eval().to(device)
    module.load_state_dict(
        _filter_state(state, msa_layers, pairformer_layers, token_layers)
    )
    return module


def _filter_state(
    state: dict[str, torch.Tensor],
    msa_layers: int,
    pairformer_layers: int,
    token_layers: int,
) -> dict[str, torch.Tensor]:
    prefixes = (
        "input_embedder.",
        "s_init.",
        "z_init_1.",
        "z_init_2.",
        "rel_pos.",
        "token_bonds.",
        "token_bonds_type.",
        "contact_conditioning.",
        "s_norm.",
        "z_norm.",
        "s_recycle.",
        "z_recycle.",
        "msa_module.",
        "pairformer_module.",
        "diffusion_conditioning.",
    )
    module_state = {}
    for key, value in state.items():
        if key.startswith("structure_module.score_model."):
            local_key = key.replace("structure_module.score_model.", "score_model.")
            if local_key.startswith("score_model.token_transformer.layers."):
                index = int(local_key.split(".")[3])
                if index >= token_layers:
                    continue
            module_state[local_key] = value
            continue
        if not key.startswith(prefixes):
            continue
        if key.startswith("msa_module.layers."):
            index = int(key.split(".")[2])
            if index >= msa_layers:
                continue
        if key.startswith("pairformer_module.layers."):
            index = int(key.split(".")[2])
            if index >= pairformer_layers:
                continue
        if key.startswith("diffusion_conditioning.token_trans_proj_z."):
            index = int(key.split(".")[2])
            if index >= token_layers:
                continue
        module_state[key] = value
    return module_state


def _make_feats(tokens: int, atoms: int, msa_rows: int) -> dict[str, np.ndarray]:
    if atoms % 32 != 0:
        msg = "--atoms must be a multiple of 32"
        raise SystemExit(msg)
    atom_to_token = np.zeros((1, atoms, tokens), dtype=np.float32)
    atom_to_token[0, np.arange(atoms), np.arange(atoms) % tokens] = 1.0
    ref_element = np.zeros((1, atoms, 128), dtype=np.float32)
    ref_element[0, np.arange(atoms), np.arange(atoms) % 128] = 1.0
    chars = np.zeros((1, atoms, 4, 64), dtype=np.float32)
    for index in range(4):
        chars[0, np.arange(atoms), index, (np.arange(atoms) + index) % 64] = 1.0
    res_type = np.zeros((1, tokens, 33), dtype=np.float32)
    res_type[0, np.arange(tokens), np.arange(tokens) % 33] = 1.0
    profile = np.zeros((1, tokens, 33), dtype=np.float32)
    profile[0, np.arange(tokens), (np.arange(tokens) + 3) % 33] = 1.0
    contact_conditioning = np.zeros((1, tokens, tokens, 5), dtype=np.float32)
    i_grid, j_grid = np.meshgrid(np.arange(tokens), np.arange(tokens), indexing="ij")
    contact_conditioning[0, i_grid, j_grid, 2 + ((i_grid + j_grid) % 3)] = 1.0
    return {
        "ref_pos": np.linspace(-0.3, 0.3, num=atoms * 3, dtype=np.float32).reshape(
            1, atoms, 3
        ),
        "atom_pad_mask": np.ones((1, atoms), dtype=np.float32),
        "ref_space_uid": (np.arange(atoms) // 8).reshape(1, atoms).astype(np.int64),
        "ref_charge": np.linspace(-0.5, 0.5, num=atoms, dtype=np.float32).reshape(
            1, atoms
        ),
        "ref_element": ref_element,
        "ref_atom_name_chars": chars,
        "atom_to_token": atom_to_token,
        "res_type": res_type,
        "profile": profile,
        "deletion_mean": np.linspace(0.0, 1.0, num=tokens, dtype=np.float32).reshape(
            1, tokens
        ),
        "method_feature": (np.arange(tokens) % 12).reshape(1, tokens).astype(np.int64),
        "modified": (np.arange(tokens) % 2).reshape(1, tokens).astype(np.int64),
        "cyclic_period": np.zeros((1, tokens), dtype=np.float32),
        "mol_type": (np.arange(tokens) % 4).reshape(1, tokens).astype(np.int64),
        "asym_id": np.zeros((1, tokens), dtype=np.int64),
        "residue_index": np.arange(tokens).reshape(1, tokens).astype(np.int64),
        "entity_id": np.zeros((1, tokens), dtype=np.int64),
        "token_index": np.arange(tokens).reshape(1, tokens).astype(np.int64),
        "sym_id": np.zeros((1, tokens), dtype=np.int64),
        "token_bonds": np.eye(tokens, dtype=np.float32).reshape(1, tokens, tokens, 1),
        "type_bonds": (np.arange(tokens * tokens) % 7).reshape(1, tokens, tokens),
        "contact_conditioning": contact_conditioning,
        "contact_threshold": np.linspace(
            4.0, 20.0, num=tokens * tokens, dtype=np.float32
        ).reshape(1, tokens, tokens),
        "msa": (np.arange(msa_rows * tokens) % 33).reshape(1, msa_rows, tokens),
        "has_deletion": np.zeros((1, msa_rows, tokens), dtype=np.float32),
        "deletion_value": np.linspace(
            0.0, 1.0, num=msa_rows * tokens, dtype=np.float32
        ).reshape(1, msa_rows, tokens),
        "msa_paired": np.ones((1, msa_rows, tokens), dtype=np.float32),
        "msa_mask": np.ones((1, msa_rows, tokens), dtype=np.float32),
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
    }


def _tree_to_torch(tree: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in tree.items()}


def _tree_to_jax(tree: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in tree.items()}


def _bench_torch(fn, device: str, warmup: int, iters: int) -> dict[str, Any]:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
        _torch_sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _torch_sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "times_ms": times,
        "peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }


def _bench_jax(fn, warmup: int, iters: int) -> dict[str, Any]:
    start_compile = time.perf_counter()
    _block_until_ready(fn())
    compile_run_ms = (time.perf_counter() - start_compile) * 1000.0
    for _ in range(warmup):
        _block_until_ready(fn())
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "compile_plus_first_run_ms": compile_run_ms,
        "mean_ms": statistics.mean(times),
        "times_ms": times,
        "memory_stats": _jax_memory_stats(),
    }


def _block_until_ready(value):
    return jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _torch_sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _jax_memory_stats() -> dict[str, Any] | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return None
    return dict(stats) if stats is not None else None


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    main()
