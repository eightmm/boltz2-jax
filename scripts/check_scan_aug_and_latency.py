"""Verify scan/eager augmentation equivalence and measure use_scan latency.

CHANGE-1: use_scan toggles BOTH the sampling loop and the layer-stack scans.
CHANGE-2: augmentation is supported inside the compiled scan path.

Run:
    uv run python scripts/check_scan_aug_and_latency.py
"""

from __future__ import annotations

import statistics
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import (  # noqa: E402
    _jax_memory_stats,
    _load_features_pt,
    _tree_to_jax,
)

CKPT = Path("../boltz/.cache/boltz/boltz2_conf.ckpt")
FEATS = Path("outputs/real_features/1UBQ_A.pt")
MSA_LAYERS, PF_LAYERS, TOK_LAYERS = 4, 64, 24


def _build():
    state = load_checkpoint_state_dict(CKPT)
    params = map_boltz2_graph_state_dict(
        state, num_msa_layers=MSA_LAYERS, num_pairformer_layers=PF_LAYERS,
        num_token_layers=TOK_LAYERS, token_transformer_heads=16,
    )
    feats_np, rid = _load_features_pt(FEATS)
    return params, _tree_to_jax(feats_np), feats_np, rid


def _time(fn, iters=5):
    fn(0).block_until_ready()  # not counted: compile already done by caller
    ts = []
    for i in range(iters):
        s = time.perf_counter()
        fn(i + 1).block_until_ready()
        ts.append((time.perf_counter() - s) * 1000.0)
    return statistics.mean(ts), ts


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    params, feats, feats_np, rid = _build()
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    steps, recycling = 200, 3
    print(f"record={rid} n_atoms={n_atoms} steps={steps} recycling={recycling} "
          f"device={jax.default_backend()}")

    # ---- Equivalence: eager vs scan with augmentation=True, same key ----
    # Both paths consume an IDENTICAL per-step key schedule (init split then
    # per-step aug-split/noise-split), so injecting the same init_noise and the
    # same PRNG key makes the random augmentation rotations identical.
    key = jax.random.PRNGKey(42)
    rng = np.random.default_rng(0)
    init_noise = jnp.asarray(rng.standard_normal((1, n_atoms, 3)), dtype=jnp.float32)
    eq_kw = dict(
        num_sampling_steps=steps, recycling_steps=recycling,
        token_layers=TOK_LAYERS, augmentation=True, alignment_reverse_diff=True,
        init_noise=init_noise,
    )
    eager = jax.jit(partial(boltz2_sample_forward, use_scan=False, **eq_kw))
    scan = jax.jit(partial(boltz2_sample_forward, use_scan=True, **eq_kw))
    ce = eager(params, feats, key)["sample_atom_coords"].block_until_ready()
    cs = scan(params, feats, key)["sample_atom_coords"].block_until_ready()
    diff = float(jnp.max(jnp.abs(ce - cs)))
    print(f"[aug equivalence] eager vs scan max|diff| = {diff:.3e} "
          f"(target < 1e-4): {'PASS' if diff < 1e-4 else 'FAIL'}")

    # ---- Latency: unrolled (use_scan=False) vs scan (use_scan=True) ----
    lat_kw = dict(
        num_sampling_steps=steps, recycling_steps=recycling,
        token_layers=TOK_LAYERS, augmentation=False, alignment_reverse_diff=True,
    )
    for label, flag in [("unrolled (use_scan=False, default)", False),
                        ("scan    (use_scan=True)", True)]:
        sampler = jax.jit(partial(boltz2_sample_forward, use_scan=flag, **lat_kw))

        def call(seed, _s=sampler):
            return _s(params, feats, jax.random.PRNGKey(seed))["sample_atom_coords"]

        s = time.perf_counter()
        call(0).block_until_ready()
        compile_first = (time.perf_counter() - s) * 1000.0
        steady, _ = _time(call)
        mem = _jax_memory_stats()
        peak = (mem.get("peak_bytes_in_use", 0) / 1024**2) if mem else None
        print(f"[latency] {label}: compile+first={compile_first:9.1f} ms  "
              f"steady={steady:8.1f} ms  peak={peak and round(peak)} MiB")


if __name__ == "__main__":
    main()
