"""Generate Boltz-2 features for the protein+ligand ``affinity.yaml`` example.

Mirrors ``make_multimer_feats.py``: ``download_boltz2`` -> ``check_inputs`` ->
``process_inputs`` -> ``PredictionDataset.__getitem__`` (tokenizer + molecule
load + ``Boltz2Featurizer.process``). Uses ``msa: empty`` (no MSA server). The
input is boltz/examples/affinity.yaml (protein A + ligand B, affinity binder B),
so ``affinity_token_mask`` must have nonzero (ligand) entries.

Dump to outputs/real_features/affinity_complex.{pt,npz}.

  JAX_PLATFORMS=cpu uv run --extra torch-bridge python scripts/make_affinity_feats.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from boltz.data.module.inferencev2 import PredictionDataset
from boltz.data.types import Manifest
from boltz.main import check_inputs, download_boltz2, process_inputs

ROOT = Path(__file__).resolve().parent.parent
BOLTZ_ROOT = ROOT.parent / "boltz"
YAML = ROOT / "outputs" / "prep" / "affinity_no_msa.yaml"
CACHE = BOLTZ_ROOT / ".cache" / "boltz"
OUT_DIR = ROOT / "outputs" / "real_features"
PREP_DIR = ROOT / "outputs" / "prep" / "affinity"


def main() -> None:
    assert YAML.exists(), f"missing input yaml: {YAML}"
    CACHE.mkdir(parents=True, exist_ok=True)

    download_boltz2(CACHE)
    ccd_path = CACHE / "ccd.pkl"
    mol_dir = CACHE / "mols"
    assert mol_dir.exists(), (
        f"BLOCKER: mol dir missing after download_boltz2: {mol_dir}"
    )

    data = check_inputs(YAML)
    assert data, f"check_inputs returned nothing for {YAML}"

    process_inputs(
        data=data,
        out_dir=PREP_DIR,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    processed = PREP_DIR / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=mol_dir,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
    )
    assert len(dataset) >= 1, "empty affinity dataset"
    features = dataset[0]

    feats_t: dict[str, torch.Tensor] = {}
    feats_np: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if key.startswith("_") or key == "record":
            continue
        if not torch.is_tensor(value):
            continue
        batched = value.unsqueeze(0)
        feats_t[key] = batched
        feats_np[key] = batched.detach().cpu().numpy()

    feats_t["_record_id"] = manifest.records[0].id

    assert "affinity_token_mask" in feats_np, "missing affinity_token_mask"
    n_lig = int(feats_np["affinity_token_mask"].astype(bool).sum())
    assert n_lig > 0, (
        f"BLOCKER: affinity_token_mask all zero (no ligand tokens) for {YAML}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = OUT_DIR / "affinity_complex.pt"
    npz_path = OUT_DIR / "affinity_complex.npz"
    torch.save(feats_t, pt_path)
    np.savez(npz_path, **feats_np)

    n_tokens = int(feats_np["token_pad_mask"].shape[-1])
    n_atoms = int(feats_np["atom_pad_mask"].shape[-1])
    print(
        f"affinity feats: tokens={n_tokens} atoms={n_atoms} "
        f"n_ligand_tokens={n_lig} ({len(feats_np)} tensors)"
    )
    print(f"wrote {pt_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
