"""Generate Boltz-2 features for the multi-chain ``multimer.yaml`` example.

Reproduces the exact preprocessing the original ``1UBQ_A.pt`` bundle came from:
``download_boltz2`` -> ``check_inputs`` -> ``process_inputs`` (writes the
``processed/`` tree) -> ``PredictionDataset.__getitem__`` (tokenizer + molecule
load + ``Boltz2Featurizer.process``). The resulting feats dict (a single,
un-collated example, batched to leading dim 1) is dumped to
``outputs/real_features/multimer.pt`` and ``.npz``.

Dev-time torch is allowed here (the conversion happens once, off the inference
path). Run with the torch-bridge extra:

  JAX_PLATFORMS=cpu uv run --extra torch-bridge python scripts/make_multimer_feats.py
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
# Same two chains as boltz/examples/multimer.yaml, with `msa: empty` per chain
# so no MSA server is needed (matches the no-MSA dump path the 1UBQ_A bundle
# used). We never edit the boltz repo, so this lives under our outputs/.
YAML = ROOT / "outputs" / "prep" / "multimer_no_msa.yaml"
CACHE = BOLTZ_ROOT / ".cache" / "boltz"
OUT_DIR = ROOT / "outputs" / "real_features"
PREP_DIR = ROOT / "outputs" / "prep" / "multimer"


def main() -> None:
    assert YAML.exists(), f"missing input yaml: {YAML}"
    CACHE.mkdir(parents=True, exist_ok=True)

    # Ensure CCD/mols are present. download_boltz2 fetches ccd.pkl + mols if
    # absent; it is a no-op when the cache is already populated.
    download_boltz2(CACHE)
    ccd_path = CACHE / "ccd.pkl"
    mol_dir = CACHE / "mols"
    assert mol_dir.exists(), (
        f"BLOCKER: mol dir missing after download_boltz2: {mol_dir}"
    )

    data = check_inputs(YAML)
    assert data, f"check_inputs returned nothing for {YAML}"

    # process_inputs (rank_zero_only) writes processed/manifest.json but returns
    # None, so we load the manifest back from disk.
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
    assert len(dataset) >= 1, "empty multimer dataset"
    features = dataset[0]  # single un-collated example

    # The PredictionDataset returns per-example tensors WITHOUT the leading
    # batch dim that collate() adds. Match the 1UBQ_A bundle layout (leading
    # dim 1) by unsqueezing every tensor.
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = OUT_DIR / "multimer.pt"
    npz_path = OUT_DIR / "multimer.npz"
    torch.save(feats_t, pt_path)
    np.savez(npz_path, **feats_np)

    n_tokens = int(feats_np["token_pad_mask"].shape[-1])
    n_atoms = int(feats_np["atom_pad_mask"].shape[-1])
    n_chains = int(np.unique(feats_np["asym_id"][0]).size)
    print(
        f"multimer feats: tokens={n_tokens} atoms={n_atoms} chains={n_chains} "
        f"({len(feats_np)} tensors)"
    )
    print(f"wrote {pt_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
