"""Verify boltz_jax.data pipeline for MSA / ligand / template modalities.

Self-contained (no `import boltz`). Builds a protein-only baseline and each
modality variant, then prints feature-dict evidence proving the modality
changed the features.

Run:
  JAX_PLATFORMS=cpu uv run --extra torch-bridge \
    python scripts/test_modalities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

assert "boltz" not in sys.modules, "runtime must not import boltz"

from featurize import featurize_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BOLTZ_ROOT = ROOT.parent / "boltz"
MOL_DIR = BOLTZ_ROOT / ".cache" / "boltz" / "mols"
CIF_1UBQ = (
    BOLTZ_ROOT
    / "benchmark_results"
    / "real_pdb_eval_expanded_2026-06-04"
    / "references"
    / "1UBQ.cif"
)
WORK = ROOT / "outputs" / "prep" / "modalities"
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def write(name: str, body: str) -> Path:
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(body)
    return p


def run(name: str, body: str) -> dict[str, np.ndarray]:
    p = write(name, body)
    feats, _, _ = featurize_yaml(p, WORK / name, MOL_DIR)
    return feats


def summary(feats: dict[str, np.ndarray]) -> dict:
    msa = feats["msa"][0]
    return {
        "n_tokens": int(feats["token_index"].shape[1]),
        "n_atoms_present": int(feats["atom_pad_mask"].sum()),
        "msa_depth": int(msa.shape[0]),
        "deletion_value_sum": float(feats["deletion_value"].sum()),
        "profile_nonzero": int((feats["profile"][0] != 0).sum()),
        "msa_mask_sum": float(feats["msa_mask"].sum()),
        "mol_type_unique": np.unique(feats["mol_type"][0]).tolist(),
        "template_mask_sum": float(feats.get("template_mask", np.zeros(1)).sum()),
        "template_frame_rot_nonzero": int(
            (feats.get("template_frame_rot", np.zeros(1)) != 0).sum()
        ),
        "template_mask_frame_sum": float(
            feats.get("template_mask_frame", np.zeros(1)).sum()
        ),
    }


def main() -> None:
    # ---- baseline: protein, msa empty
    base = run(
        "base",
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {SEQ}\n      msa: empty\n",
    )

    # ---- MSA: build a tiny a3m (query + 3 aligned seqs)
    a3m = WORK / "msa" / "q.a3m"
    a3m.parent.mkdir(parents=True, exist_ok=True)
    seqs = [SEQ]
    for i, pos in enumerate((5, 20, 40)):
        m = list(SEQ)
        m[pos] = "A" if m[pos] != "A" else "G"
        # introduce a lowercase insertion (deletion column) in one
        s = "".join(m)
        if i == 1:
            s = s[:10] + "a" + s[10:]
        seqs.append(s)
    a3m.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
    msa = run(
        "msa",
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {SEQ}\n      msa: {a3m}\n",
    )

    # ---- ligand: protein + SMILES ligand + CCD ligand
    lig = run(
        "lig",
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {SEQ}\n      msa: empty\n"
        "  - ligand:\n      id: B\n      smiles: 'CC(=O)O'\n",
    )
    lig_ccd = run(
        "lig_ccd",
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {SEQ}\n      msa: empty\n"
        "  - ligand:\n      id: B\n      ccd: ATP\n",
    )

    # ---- template
    tmpl_status = "BLOCKED: 1UBQ.cif not found"
    tmpl = None
    if CIF_1UBQ.exists():
        tmpl = run(
            "tmpl",
            "version: 1\nsequences:\n  - protein:\n      id: A\n"
            f"      sequence: {SEQ}\n      msa: empty\n"
            f"templates:\n  - cif: {CIF_1UBQ}\n",
        )
        tmpl_status = "ran"

    print("=== BASELINE (protein, msa empty) ===")
    bs = summary(base)
    for k, v in bs.items():
        print(f"  {k}: {v}")

    print("\n=== MSA (precomputed a3m, depth 4) ===")
    for k, v in summary(msa).items():
        print(f"  {k}: {v}")

    print("\n=== LIGAND (SMILES CC(=O)O) ===")
    for k, v in summary(lig).items():
        print(f"  {k}: {v}")

    print("\n=== LIGAND (CCD ATP) ===")
    for k, v in summary(lig_ccd).items():
        print(f"  {k}: {v}")

    print(f"\n=== TEMPLATE ({tmpl_status}) ===")
    if tmpl is not None:
        for k, v in summary(tmpl).items():
            print(f"  {k}: {v}")

    # ---- verdicts
    print("\n=== VERDICTS ===")
    print(
        "MSA:",
        "DONE"
        if summary(msa)["msa_depth"] > 1 > 0
        and summary(msa)["msa_depth"] != bs["msa_depth"]
        else "CHECK",
    )
    print(
        "LIGAND(SMILES):",
        "DONE"
        if summary(lig)["n_atoms_present"] > bs["n_atoms_present"]
        and 3 in summary(lig)["mol_type_unique"]
        else "CHECK",
    )
    print(
        "LIGAND(CCD):",
        "DONE"
        if summary(lig_ccd)["n_atoms_present"] > bs["n_atoms_present"]
        and 3 in summary(lig_ccd)["mol_type_unique"]
        else "CHECK",
    )
    if tmpl is not None:
        print(
            "TEMPLATE:",
            "DONE" if summary(tmpl)["template_mask_sum"] > 0 else "CHECK",
        )


if __name__ == "__main__":
    main()
