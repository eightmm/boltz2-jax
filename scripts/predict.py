"""boltz_jax predict CLI: raw YAML -> structure file (PDB/mmCIF).

Featurizes the input via boltz_jax.data, runs the JAX structure sampler with
native weights, and writes the predicted structure.

Run:
  uv run --extra torch-bridge python scripts/predict.py \
      --input X.yaml \
      --weights outputs/native_weights/boltz2_conf \
      --mols .cache/boltz/mols \
      --out-dir outputs/predictions \
      [--steps 200 --recycling 3 --compute-dtype float32|bfloat16 --fmt pdb|cif]
      [--use-msa-server]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from featurize import featurize_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="raw input YAML")
    p.add_argument(
        "--seq", action="append", default=[],
        help="protein sequence (repeatable); builds a job YAML when --input is "
        "omitted",
    )
    p.add_argument("--dna", action="append", default=[], help="DNA sequence")
    p.add_argument("--rna", action="append", default=[], help="RNA sequence")
    p.add_argument(
        "--ligand-ccd", action="append", default=[], help="ligand CCD code"
    )
    p.add_argument(
        "--ligand-smiles", action="append", default=[], help="ligand SMILES"
    )
    p.add_argument(
        "--weights", type=Path, default=ROOT / "outputs/native_weights/boltz2_conf"
    )
    p.add_argument(
        "--mols", type=Path, default=ROOT / ".cache/boltz/mols"
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/predictions")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument(
        "--compute-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    p.add_argument("--fmt", choices=["pdb", "cif"], default="pdb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--use-msa-server",
        action="store_true",
        help="generate MSAs via the colabfold mmseqs2 server for protein chains "
        "that request one (msa not 'empty' and no precomputed a3m/csv)",
    )
    p.add_argument("--msa-server-url", default="https://api.colabfold.com")
    p.add_argument(
        "--msa-pairing-strategy", choices=["greedy", "complete"], default="greedy"
    )
    p.add_argument(
        "--compile-cache",
        type=Path,
        default=ROOT / "outputs/compile_cache",
        help="persistent XLA compilation cache dir (reuse compiles across runs); "
        "on by default, pass a different path to relocate",
    )
    p.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "outputs/feature_cache",
        help="persistent featurization cache dir (memoize features by input "
        "digest; on by default, pass a different path to relocate)",
    )
    p.add_argument(
        "--bucket",
        action="store_true",
        help="pad token/atom dims to a shape ladder so the compile cache hits "
        "across different-length targets (serving). Pads FLOPs (single-run "
        "loss, multi-run win); shifts real coords by ~1e-4 A (fp reassociation "
        "from padded reductions), biologically negligible.",
    )
    args = p.parse_args()

    if args.input is None:
        any_entity = (
            args.seq or args.dna or args.rna or args.ligand_ccd or args.ligand_smiles
        )
        assert any_entity, "provide --input YAML or --seq/--dna/--rna/--ligand-*"
        from make_job_yaml import build_job_yaml

        gen_dir = args.out_dir / "prep" / "job"
        gen_dir.mkdir(parents=True, exist_ok=True)
        args.input = gen_dir / "job.yaml"
        args.input.write_text(
            build_job_yaml(
                args.seq,
                args.ligand_ccd,
                dna=args.dna,
                rna=args.rna,
                ligands_smiles=args.ligand_smiles,
                use_msa_server=args.use_msa_server,
            )
        )
        print(f"built job YAML: {args.input}")
    assert args.input.exists(), f"missing input: {args.input}"

    # --- Step 1: YAML -> processed tree + features (torch side) ---
    prep_dir = args.out_dir / "prep" / args.input.stem
    prep_dir.mkdir(parents=True, exist_ok=True)
    feats_np, manifest, struct_dir = featurize_yaml(
        args.input,
        prep_dir,
        args.mols,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
        cache_dir=args.feature_cache,
    )
    if manifest is not None:
        records = manifest.records
        assert len(records) == 1, f"expected 1 record, got {len(records)}"
        record_id = records[0].id
    else:  # cache hit: record id is the structure npz filename
        npzs = sorted(struct_dir.glob("*.npz"))
        assert len(npzs) == 1, f"expected 1 cached structure, got {len(npzs)}"
        record_id = npzs[0].stem
        print("(feature cache hit)")
    print(f"featurized record: {record_id}")

    # --- Step 2: run the JAX sampler (no torch/boltz at runtime here) ---
    import jax
    import jax.numpy as jnp

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.predict import boltz2_predict

    jax.config.update("jax_default_matmul_precision", "highest")
    if args.compile_cache is not None:
        cache = args.compile_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        print(f"compile cache: {cache}")
    compute_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.compute_dtype]

    params = load_params(args.weights)

    if args.bucket:
        from boltz_jax.data.bucket import pad_feats

        token_ladder = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
        n_tok = int(feats_np["token_pad_mask"].shape[-1])
        n_atom = int(feats_np["atom_pad_mask"].shape[-1])
        tgt_tok = next((b for b in token_ladder if b >= n_tok), n_tok)
        tgt_atom = ((n_atom + 31) // 32) * 32  # atom-window (32) multiple
        feats_np, _log = pad_feats(feats_np, tgt_tok, tgt_atom)
        print(f"bucket: tokens {n_tok}->{tgt_tok} atoms {n_atom}->{tgt_atom}")

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
