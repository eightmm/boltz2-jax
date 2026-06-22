"""High-level Python API: sequences / YAML -> predicted structure.

Wraps featurization + the JAX sampler so other code (incl. other JAX models)
can call Boltz-2 inference without the CLI. Heavy deps (jax, torch) are imported
lazily, so ``import boltz_jax`` stays cheap.

  import boltz_jax
  out = boltz_jax.predict(seq=["MKQLED..."], ligand_ccd=["ATP"],
                          weights="outputs/native_weights/boltz2_conf",
                          mols=".cache/boltz/mols")
  out["coords"]       # (n_atom, 3) np.ndarray
  out["plddt"]        # (n_atom,) np.ndarray

For composing with other JAX models, use the lower-level
``boltz_jax.boltz2_predict`` (a pure JAX function over params + feature pytree)
and ``boltz_jax.load_params`` directly.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from boltz_jax.data.featurize import featurize_yaml
from boltz_jax.data.job_yaml import build_job_yaml

_TOKEN_LADDER = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def featurize(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    mols: str | Path,
    out_dir: str | Path | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    feature_cache: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], str, Path]:
    """Featurize a YAML path or bare entities.

    Returns ``(feats, record_id, struct_dir)``.
    """
    work = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    if input is None:
        if not (seq or dna or rna or ligand_ccd or ligand_smiles):
            raise ValueError("provide `input` YAML or sequences/ligands")
        work.mkdir(parents=True, exist_ok=True)
        input = work / "job.yaml"
        Path(input).write_text(
            build_job_yaml(
                seq, ligand_ccd, dna=dna, rna=rna,
                ligands_smiles=ligand_smiles, use_msa_server=use_msa_server,
            )
        )
    feats, manifest, struct_dir = featurize_yaml(
        Path(input), work, Path(mols),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        cache_dir=Path(feature_cache) if feature_cache is not None else None,
    )
    if manifest is not None:
        record_id = manifest.records[0].id
    else:
        record_id = sorted(struct_dir.glob("*.npz"))[0].stem
    return feats, record_id, struct_dir


def predict(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    weights: str | Path,
    mols: str | Path,
    out_dir: str | Path | None = None,
    steps: int = 200,
    recycling: int = 3,
    seed: int = 0,
    compute_dtype: str = "float32",
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    feature_cache: str | Path | None = None,
    compile_cache: str | Path | None = None,
    bucket: bool = False,
    write_fmt: str | None = None,
) -> dict[str, Any]:
    """Run end-to-end Boltz-2 inference.

    Pass either ``input`` (a job YAML) or bare ``seq``/``dna``/``rna``/
    ``ligand_ccd``/``ligand_smiles``. Returns a dict with ``coords`` (n_atom, 3),
    ``plddt``, ``record_id``, ``raw`` (full model output), and ``out_path`` (if
    ``write_fmt`` is "pdb"/"cif"). Defaults match the Boltz-2 reference.
    """
    import jax
    import jax.numpy as jnp

    from boltz_jax.bridge.native import load_params
    from boltz_jax.models.predict import boltz2_predict

    feats_np, record_id, struct_dir = featurize(
        input=input, seq=seq, dna=dna, rna=rna,
        ligand_ccd=ligand_ccd, ligand_smiles=ligand_smiles,
        mols=mols, out_dir=out_dir, use_msa_server=use_msa_server,
        msa_server_url=msa_server_url, msa_pairing_strategy=msa_pairing_strategy,
        feature_cache=feature_cache,
    )

    jax.config.update("jax_default_matmul_precision", "highest")
    if compile_cache is not None:
        cache = Path(compile_cache).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[compute_dtype]
    params = load_params(Path(weights))

    if bucket:
        from boltz_jax.data.bucket import pad_feats

        n_tok = int(feats_np["token_pad_mask"].shape[-1])
        n_atom = int(feats_np["atom_pad_mask"].shape[-1])
        tgt_tok = next((b for b in _TOKEN_LADDER if b >= n_tok), n_tok)
        tgt_atom = ((n_atom + 31) // 32) * 32
        feats_np, _ = pad_feats(feats_np, tgt_tok, tgt_atom)

    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}
    out = boltz2_predict(
        params, feats, jax.random.PRNGKey(seed),
        recycling_steps=recycling, num_sampling_steps=steps,
        augmentation=False, run_confidence=True, run_distogram=True,
        run_bfactor=True, compute_dtype=dtype, use_scan=True,
    )
    coords = np.asarray(jax.block_until_ready(out["sample_atom_coords"]))
    plddt = np.asarray(out["plddt"]).reshape(-1)
    assert np.all(np.isfinite(coords)), "non-finite coordinates produced"

    result: dict[str, Any] = {
        "coords": coords, "plddt": plddt, "record_id": record_id, "raw": out,
    }
    if write_fmt is not None:
        from boltz_jax.data.write.structure import write_prediction

        dest = Path(out_dir) if out_dir is not None else struct_dir.parent
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / f"{record_id}.{write_fmt}"
        result["out_path"] = write_prediction(
            structure_npz=struct_dir / f"{record_id}.npz",
            coords=coords,
            atom_pad_mask=feats_np["atom_pad_mask"].reshape(-1),
            out_path=out_path,
            plddts=plddt,
            fmt=write_fmt,
        )
    return result
