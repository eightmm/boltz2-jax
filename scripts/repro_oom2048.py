"""Standalone repro for the tokens=2048 sampler OOM. Foreground only."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--atoms-per-token", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--triangle-backend", default="xla")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--atom-chunk", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from benchmark_boltz2_graph import _make_feats

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")

    tokens = args.tokens
    atoms = tokens * args.atoms_per_token
    feats_np = _make_feats(tokens, atoms, 1)
    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}
    params = load_params(str(REPO / "outputs/native_weights/boltz2_conf.safetensors"))

    rng = np.random.default_rng(0)
    init_noise = jnp.asarray(
        rng.standard_normal((args.multiplicity, atoms, 3)).astype(np.float32)
    )
    step_noises = jnp.asarray(
        rng.standard_normal(
            (args.steps, args.multiplicity, atoms, 3)
        ).astype(np.float32)
    )

    extra = {}
    if args.atom_chunk > 0:
        extra["atom_attention_chunk"] = args.atom_chunk

    sampler = jax.jit(
        partial(
            boltz2_sample_forward,
            recycling_steps=args.recycling,
            num_sampling_steps=args.steps,
            multiplicity=args.multiplicity,
            augmentation=False,
            alignment_reverse_diff=True,
            use_scan=True,
            chunk_size=args.chunk_size,
            matmul_precision="highest",
            attention_backend="xla",
            triangle_backend=args.triangle_backend,
            compute_dtype=jnp.float32,
            init_noise=init_noise,
            step_noises=step_noises,
            **extra,
        )
    )

    result = {"tokens": tokens, "atoms": atoms, "atom_chunk": args.atom_chunk}
    try:
        t0 = time.perf_counter()
        out = sampler(params, feats, jax.random.PRNGKey(0))[
            "sample_atom_coords"
        ].block_until_ready()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        out = sampler(params, feats, jax.random.PRNGKey(1))[
            "sample_atom_coords"
        ].block_until_ready()
        steady_ms = (time.perf_counter() - t1) * 1000.0
        out_np = np.asarray(out)
        mem = dict(jax.devices()[0].memory_stats() or {})
        result.update(
            ok=True,
            compile_plus_first_ms=compile_ms,
            steady_ms=steady_ms,
            peak_mib=mem.get("peak_bytes_in_use", 0) / 1024**2,
            finite=bool(np.isfinite(out_np).all()),
        )
    except Exception as exc:  # noqa: BLE001
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())

    print(json.dumps(result, indent=2)[:4000])
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
