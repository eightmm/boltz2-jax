"""Parity + VRAM bench for token-attention query-chunking (35 GiB OOM fix).

Foreground only. STEP 0 finding: the tokens=2048 ~35 GiB blocker is the
[B, heads, N, N] fp32 token self-attention score buffer emitted by
diffusion_transformer._attention_pair_bias_no_proj_z_forward (op
``while/body/closed_call/bihd,bjhd->bhij/dot_general``). bf16 did not shrink it
because that path force-cast q/k/v/scores to fp32 unconditionally.

Fix: query-axis chunking of the token attention (``token_attention_chunk``) so
the score buffer is never materialized for the full N, plus a dtype gate so the
score matmul honors compute_dtype under bf16. fp32 default is bit-exact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def _run(sampler, params, feats, key, label):
    import jax
    import numpy as np

    out = sampler(params, feats, key)["sample_atom_coords"].block_until_ready()
    t0 = time.perf_counter()
    out = sampler(params, feats, jax.random.PRNGKey(7))[
        "sample_atom_coords"
    ].block_until_ready()
    steady = (time.perf_counter() - t0) * 1000.0
    out_np = np.asarray(out)
    mem = dict(jax.devices()[0].memory_stats() or {})
    return {
        "label": label,
        "steady_ms": steady,
        "peak_mib": mem.get("peak_bytes_in_use", 0) / 1024**2,
        "finite": bool(np.isfinite(out_np).all()),
    }, out_np


def parity():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    jax.config.update("jax_default_matmul_precision", "highest")
    feats_np = dict(np.load(REPO / "outputs/real_features/1UBQ_A.npz"))
    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}
    atoms = feats["ref_pos"].shape[1]
    params = load_params(str(REPO / "outputs/native_weights/boltz2_conf.safetensors"))

    rng = np.random.default_rng(0)
    init_noise = jnp.asarray(rng.standard_normal((1, atoms, 3)).astype(np.float32))
    step_noises = jnp.asarray(
        rng.standard_normal((20, 1, atoms, 3)).astype(np.float32)
    )

    def mk(chunk):
        return jax.jit(
            partial(
                boltz2_sample_forward,
                recycling_steps=3,
                num_sampling_steps=20,
                multiplicity=1,
                augmentation=False,
                alignment_reverse_diff=True,
                use_scan=True,
                chunk_size=128,
                token_attention_chunk=chunk,
                matmul_precision="highest",
                attention_backend="xla",
                triangle_backend="xla",
                compute_dtype=jnp.float32,
                init_noise=init_noise,
                step_noises=step_noises,
            )
        )

    key = jax.random.PRNGKey(0)
    base = np.asarray(mk(None)(params, feats, key)["sample_atom_coords"])
    chunked = np.asarray(mk(64)(params, feats, key)["sample_atom_coords"])
    diff = float(np.max(np.abs(base - chunked)))
    # aligned RMSD (no rotation needed; identical noise -> same frame)
    rmsd = float(np.sqrt(np.mean((base - chunked) ** 2)))
    return {"max_abs_diff": diff, "rmsd": rmsd, "atoms": int(atoms)}


def bench(tokens_list, dtypes, chunk):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from benchmark_boltz2_graph import _make_feats

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    jax.config.update("jax_default_matmul_precision", "highest")
    params = load_params(str(REPO / "outputs/native_weights/boltz2_conf.safetensors"))
    rows = []
    for tokens in tokens_list:
        atoms = tokens * 8
        feats = {k: jnp.asarray(v) for k, v in _make_feats(tokens, atoms, 1).items()}
        rng = np.random.default_rng(0)
        init_noise = jnp.asarray(
            rng.standard_normal((1, atoms, 3)).astype(np.float32)
        )
        step_noises = jnp.asarray(
            rng.standard_normal((20, 1, atoms, 3)).astype(np.float32)
        )
        for dt in dtypes:
            cd = jnp.float32 if dt == "fp32" else jnp.bfloat16
            sampler = jax.jit(
                partial(
                    boltz2_sample_forward,
                    recycling_steps=3,
                    num_sampling_steps=20,
                    multiplicity=1,
                    augmentation=False,
                    alignment_reverse_diff=True,
                    use_scan=True,
                    chunk_size=128,
                    token_attention_chunk=chunk,
                    matmul_precision="highest",
                    attention_backend="xla",
                    triangle_backend="xla",
                    compute_dtype=cd,
                    init_noise=init_noise,
                    step_noises=step_noises,
                )
            )
            try:
                r, _ = _run(
                    sampler, params, feats, jax.random.PRNGKey(0),
                    f"tokens={tokens} {dt} chunk={chunk}",
                )
                r.update(tokens=tokens, atoms=atoms, dtype=dt, ok=True)
            except Exception as exc:  # noqa: BLE001
                r = {
                    "tokens": tokens, "atoms": atoms, "dtype": dt, "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            print(json.dumps(r))
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["parity", "bench", "both"], default="both")
    p.add_argument("--tokens", default="1536,2048,3072,4096")
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--out", default="outputs/af3_superblock_bench.json")
    args = p.parse_args()

    result = {
        "profile": {
            "blocker": "[B, heads=16, N, N] fp32 token self-attention scores",
            "hlo_op": "while/body/closed_call/bihd,bjhd->bhij/dot_general",
            "source": "diffusion_transformer._attention_pair_bias_no_proj_z_forward",
            "temp_buffer_bytes_N512": 3270197024,
            "memory_analysis_temp_GB_N2048": 37.90,
            "why_bf16_failed": (
                "q/k/v/scores were force-cast to jnp.float32 unconditionally, "
                "so the dominant [B,heads,N,N] score buffer stayed fp32 "
                "regardless of compute_dtype."
            ),
            "secondary": (
                "held token_trans_bias [1,N,N,24*16] ~6.4 GiB fp32; "
                "halves under bf16; not the 35 GiB blocker."
            ),
            "fix": "query-axis chunking of token attention + dtype gate",
        }
    }
    if args.mode in ("parity", "both"):
        result["parity"] = parity()
        print("PARITY", json.dumps(result["parity"]))
    if args.mode in ("bench", "both"):
        result["bench"] = bench(
            [int(t) for t in args.tokens.split(",")],
            args.dtypes.split(","),
            args.chunk,
        )
    Path(REPO / args.out).write_text(json.dumps(result, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
