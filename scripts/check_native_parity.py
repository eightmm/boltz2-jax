"""Parity gate: native-weights path vs torch-mapping path.

Runs ``boltz2_sample_forward`` twice with IDENTICAL injected init/step noise and
augmentation OFF: (a) params from the torch-mapping converter, (b) params from
``load_params`` of the native weights. Asserts max abs coord diff < 1e-5.

Uses torch ONLY to build path (a); this is a dev-time check, not the runtime.

Run:
    uv run python scripts/check_native_parity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

MSA_LAYERS = 4
PAIRFORMER_LAYERS = 64
TOKEN_LAYERS = 24
HEADS = 16


def _run(params, feats, init_noise, step_noises, steps):
    out = boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(0),
        num_sampling_steps=steps,
        token_layers=TOKEN_LAYERS,
        augmentation=False,
        use_scan=True,
        init_noise=jnp.asarray(init_noise),
        step_noises=jnp.asarray(step_noises),
    )
    return np.asarray(jax.block_until_ready(out["sample_atom_coords"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=Path, default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt")
    )
    parser.add_argument(
        "--native", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/real_features/1UBQ_A.npz")
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")

    feats = load_features_npz(args.features)
    n_atoms = int(feats["atom_pad_mask"].shape[1])
    rng = np.random.default_rng(args.seed)
    init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
    step_noises = rng.standard_normal((args.steps, 1, n_atoms, 3)).astype(np.float32)

    state = load_checkpoint_state_dict(args.ckpt)
    torch_params = map_boltz2_graph_state_dict(
        state,
        num_msa_layers=MSA_LAYERS,
        num_pairformer_layers=PAIRFORMER_LAYERS,
        num_token_layers=TOKEN_LAYERS,
        token_transformer_heads=HEADS,
    )
    native_params = load_params(args.native)

    a = _run(torch_params, feats, init_noise, step_noises, args.steps)
    b = _run(native_params, feats, init_noise, step_noises, args.steps)

    max_diff = float(np.max(np.abs(a - b)))
    print(f"max abs coord diff (torch-map vs native): {max_diff:.3e}")
    assert max_diff < 1e-5, f"parity FAILED: {max_diff:.3e} >= 1e-5"
    print("PARITY OK (< 1e-5)")


if __name__ == "__main__":
    main()
