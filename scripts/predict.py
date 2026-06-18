"""Unified boltz_jax predict CLI: raw YAML -> structure file (PDB/mmCIF).

Self-contained (no `import boltz`): preprocess + featurize via boltz_jax.data,
run the JAX structure sampler with native weights, then write the predicted
structure with the vendored writers.

Run:
  uv run --extra torch-bridge python scripts/predict.py \
      --input X.yaml \
      --weights outputs/native_weights/boltz2_conf \
      --mols ../boltz/.cache/boltz/mols \
      --out-dir outputs/predictions \
      [--steps 200 --recycling 3 --compute-dtype float32|bfloat16 --fmt pdb|cif]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Featurization (torch side) lives in the standalone preprocessor.
from preprocess_standalone import featurize_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="raw input YAML")
    p.add_argument(
        "--weights", type=Path, default=ROOT / "outputs/native_weights/boltz2_conf"
    )
    p.add_argument(
        "--mols", type=Path, default=ROOT.parent / "boltz/.cache/boltz/mols"
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/predictions")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument(
        "--compute-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    p.add_argument("--fmt", choices=["pdb", "cif"], default="pdb")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert args.input.exists(), f"missing input: {args.input}"

    # --- Step 1: YAML -> processed tree + features (torch side) ---
    prep_dir = args.out_dir / "prep" / args.input.stem
    prep_dir.mkdir(parents=True, exist_ok=True)
    feats_np, manifest, struct_dir = featurize_yaml(
        args.input, prep_dir, args.mols
    )
    records = manifest.records
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    record_id = records[0].id
    print(f"featurized record: {record_id}")

    # --- Step 2: run the JAX sampler (no torch/boltz at runtime here) ---
    import jax
    import jax.numpy as jnp

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.predict import boltz2_predict

    jax.config.update("jax_default_matmul_precision", "highest")
    compute_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.compute_dtype]

    params = load_params(args.weights)
    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}

    out = boltz2_predict(
        params,
        feats,
        jax.random.PRNGKey(args.seed),
        recycling_steps=args.recycling,
        num_sampling_steps=args.steps,
        augmentation=False,
        run_confidence=True,
        run_distogram=True,
        run_bfactor=True,
        compute_dtype=compute_dtype,
        use_scan=True,
    )
    coords = np.asarray(jax.block_until_ready(out["sample_atom_coords"]))
    plddt = np.asarray(out["plddt"]).reshape(-1)
    assert np.all(np.isfinite(coords)), "non-finite coordinates produced"

    # --- Step 3: write structure file ---
    from boltz_jax.data.write.structure import write_prediction

    atom_pad_mask = feats_np["atom_pad_mask"].reshape(-1)
    out_path = args.out_dir / f"{record_id}.{args.fmt}"
    written = write_prediction(
        structure_npz=struct_dir / f"{record_id}.npz",
        coords=coords,
        atom_pad_mask=atom_pad_mask,
        out_path=out_path,
        plddts=plddt,
        fmt=args.fmt,
    )

    print(f"sample_atom_coords shape: {coords.shape}")
    print(
        f"pLDDT: min={plddt.min():.4f} mean={plddt.mean():.4f} "
        f"max={plddt.max():.4f}"
    )
    print(f"WROTE {written}")


if __name__ == "__main__":
    main()
