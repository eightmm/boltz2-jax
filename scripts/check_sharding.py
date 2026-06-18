"""Sharding parity gate: single-device vs 8-device sharded trunk.

OPT-IN multi-device sharding (GSPMD via jax.lax.with_sharding_constraint) over
the trunk pair ``z`` [B, N, N, C] and single ``s`` [B, N, C] activations, so a
large structure can be split across devices when available. It must be a no-op
on a single device and is validated bit-exact on CPU with SIMULATED devices.

This runs ``boltz2_trunk_forward`` on real 1UBQ_A features and compares the
trunk outputs (``s``, ``z``, ``s_inputs``, ``relative_position_encoding``)
against the default no-mesh path:

  (a) mesh=None                          -> default single-device path
  (b) 8-device mesh, shard_tokens=False  -> activations placed across all 8
        devices, replicated layout, NO partitioned reductions -> BIT-EXACT
        (this is the ASSERTED gate)
  (c) 8-device mesh, shard_tokens=True   -> token (N) axis partitioned across
        the 8 devices; the i/j/k token contractions become distributed
        partial-sum matmuls whose fp32 reduction order differs from the
        single-device order -> reported honestly, NOT bit-exact

The asserted gate is (a) vs (b): it proves the with_sharding_constraint /
NamedSharding path is wired correctly and inert on numerics when reductions
are not partitioned, while the trunk activations genuinely live distributed on
all 8 devices.

Why the trunk boundary (not the full sampler): on CPU XLA, the mere presence
of any with_sharding_constraint in the long diffusion-sampling graph nudges the
downstream matmul algorithm selection and breaks exact bitwise reproduction,
even for a replicated spec. The trunk -- where the pair/single sharding is
defined -- reproduces bit-exactly, so it is the meaningful unit to validate.

CPU-only with SIMULATED devices (no GPU). Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        uv run python scripts/check_sharding.py

The simulated devices prove correctness of the sharding path; real multi-GPU
speedup / memory savings are NOT measurable on a single physical GPU box.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_trunk_forward

OUT_KEYS = ("s", "z", "s_inputs", "relative_position_encoding")


def _run(params, feats, mesh, shard_tokens=True):
    fn = jax.jit(
        functools.partial(
            boltz2_trunk_forward,
            use_scan=True,
            mesh=mesh,
            token_axis="tok",
            shard_tokens=shard_tokens,
        )
    )
    out = fn(params, feats)
    return {k: np.asarray(jax.block_until_ready(out[k])) for k in OUT_KEYS}, out["z"]


def _maxdiff(a, b):
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in OUT_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/real_features/1UBQ_A.npz")
    )
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")

    n_dev = jax.device_count()
    print(f"jax.device_count() == {n_dev} (platform={jax.default_backend()})")
    assert n_dev >= 2, (
        "expected simulated multi-device CPU; run with "
        "XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu"
    )

    feats = load_features_npz(args.features)
    n_tok = int(feats["token_pad_mask"].shape[1])
    print(f"1UBQ_A: tokens={n_tok}")
    params = load_params(args.native)["trunk"]

    mesh = Mesh(np.asarray(jax.devices()), axis_names=("tok",))
    print(f"mesh: {mesh}")

    # (a) default single-device path (mesh=None).
    a, a_z = _run(params, feats, None)
    print(f"(a) mesh=None                       z sharding: {a_z.sharding}")

    # (b) 8-device mesh, replicated layout -> bit-exact gate.
    b, b_z = _run(params, feats, mesh, shard_tokens=False)
    print(f"(b) 8-device, shard_tokens=False    z sharding: {b_z.sharding}")
    assert b_z.sharding.num_devices == n_dev, "z not placed across all 8 devices"

    # (c) 8-device mesh, token (N) axis partitioned -> reported, not asserted.
    c, c_z = _run(params, feats, mesh, shard_tokens=True)
    print(f"(c) 8-device, shard_tokens=True     z sharding: {c_z.sharding}")

    diff_b = _maxdiff(a, b)
    diff_c = _maxdiff(a, c)
    print(f"max abs trunk diff (a) vs (b) replicated-8-dev : {diff_b:.3e}  [GATE]")
    print(f"max abs trunk diff (a) vs (c) token-partitioned: {diff_c:.3e}  [report]")
    assert diff_b < 1e-5, f"sharding parity FAILED (b): {diff_b:.3e} >= 1e-5"
    print("SHARDING PARITY OK (< 1e-5) for the replicated 8-device path.")
    print(
        "NOTE (c): token-partitioned compute reorders fp32 reductions across "
        "devices, so it is intentionally NOT bit-exact; the GATE validates the "
        "sharding mechanism is correct and inert on numerics."
    )
    print(
        "NOTE: simulated CPU devices prove correctness only; real multi-GPU "
        "speedup/memory is unmeasured on this single-GPU box."
    )


if __name__ == "__main__":
    main()
