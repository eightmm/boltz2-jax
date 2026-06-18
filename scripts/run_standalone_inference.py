"""Standalone boltz_jax inference: NO torch, NO boltz at runtime.

Loads native weights + numpy features and runs the structure sampler. To PROVE
the runtime path is torch/boltz-free, this asserts that neither ``torch`` nor
``boltz`` is present in ``sys.modules`` after importing boltz_jax.

Run:
    uv run python scripts/run_standalone_inference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import boltz_jax  # noqa: F401
from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.predict import boltz2_predict

# Fail loudly if the runtime path dragged in torch/boltz.
assert "torch" not in sys.modules, "torch leaked into the runtime import path"
assert "boltz" not in sys.modules, "boltz leaked into the runtime import path"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/real_features/1UBQ_A.npz")
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")

    params = load_params(args.weights)
    feats = load_features_npz(args.features)

    out = boltz2_predict(
        params,
        feats,
        jax.random.PRNGKey(args.seed),
        recycling_steps=0,
        num_sampling_steps=args.steps,
        token_layers=args.token_layers,
        augmentation=False,
        run_confidence=True,
        run_distogram=True,
        run_bfactor=True,
    )
    coords = jax.block_until_ready(out["sample_atom_coords"])

    finite = bool(jnp.all(jnp.isfinite(coords)))
    plddt = np.asarray(out["plddt"])
    ptm = float(np.asarray(out["ptm"]).reshape(-1)[0])
    iptm = float(np.asarray(out["iptm"]).reshape(-1)[0])
    complex_plddt = float(np.asarray(out["complex_plddt"]).reshape(-1)[0])
    print(f"sample_atom_coords shape: {coords.shape}")
    print(f"coords all finite: {finite}")
    print(f"pdistogram shape: {np.asarray(out['pdistogram']).shape}")
    print(f"pbfactor shape: {np.asarray(out['pbfactor']).shape}")
    print(f"pTM: {ptm:.4f}")
    print(f"ipTM: {iptm:.4f}")
    print(f"complex_plddt: {complex_plddt:.4f}")
    print(
        f"pLDDT (per-token): min={plddt.min():.4f} "
        f"mean={plddt.mean():.4f} max={plddt.max():.4f}"
    )
    print("torch in sys.modules:", "torch" in sys.modules)
    print("boltz in sys.modules:", "boltz" in sys.modules)
    assert finite, "non-finite coordinates produced"
    assert "torch" not in sys.modules and "boltz" not in sys.modules
    print("STANDALONE OK (no torch, no boltz)")


if __name__ == "__main__":
    main()
