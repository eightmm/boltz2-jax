"""Build a Boltz-2 job YAML from bare sequences / ligand CCD codes.

Convenience for callers who don't want to hand-write the YAML schema. Chains
are auto-assigned IDs A, B, C, ... in order (proteins first, then ligands).

  # write a YAML
  uv run python scripts/make_job_yaml.py --seq MKQLED... --ligand-ccd ATP \
    --out job.yaml
  # or pipe straight into predict via predict.py --seq (no file needed)
"""

from __future__ import annotations

import argparse
import string
from collections.abc import Sequence
from pathlib import Path


def _chain_ids(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... for n chains."""
    a = string.ascii_uppercase
    out, i = [], 0
    while len(out) < n:
        q, r = divmod(i, 26)
        out.append((a[q - 1] if q else "") + a[r])
        i += 1
    return out


def build_job_yaml(
    proteins: Sequence[str] = (),
    ligands_ccd: Sequence[str] = (),
    *,
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligands_smiles: Sequence[str] = (),
    use_msa_server: bool = False,
) -> str:
    """Return a Boltz-2 job YAML string for mixed entities.

    proteins / dna / rna: sequences (one chain each).
    ligands_ccd: ligand CCD codes (e.g. ``ATP``); ligands_smiles: SMILES strings.
    use_msa_server: if False, each protein gets ``msa: empty`` (single-sequence);
        if True, the ``msa`` key is omitted so predict's ``--use-msa-server`` can
        generate one. (Only proteins use MSA.)
    """
    n = len(proteins) + len(dna) + len(rna) + len(ligands_ccd) + len(ligands_smiles)
    if n == 0:
        raise ValueError("need at least one protein/dna/rna sequence or ligand")
    ids = iter(_chain_ids(n))
    lines = ["version: 1", "sequences:"]
    for seq in proteins:
        seq = seq.strip().upper()
        if not seq.isalpha():
            raise ValueError(f"protein sequence must be alphabetic: {seq!r}")
        lines += ["  - protein:", f"      id: {next(ids)}", f"      sequence: {seq}"]
        if not use_msa_server:
            lines.append("      msa: empty")
    for kind, seqs in (("dna", dna), ("rna", rna)):
        for seq in seqs:
            seq = seq.strip().upper()
            if not seq.isalpha():
                raise ValueError(f"{kind} must be alphabetic: {seq!r}")
            lines += [f"  - {kind}:", f"      id: {next(ids)}"]
            lines.append(f"      sequence: {seq}")
    for ccd in ligands_ccd:
        lines += ["  - ligand:", f"      id: {next(ids)}"]
        lines.append(f"      ccd: {ccd.strip()}")
    for smi in ligands_smiles:
        lines += ["  - ligand:", f"      id: {next(ids)}"]
        lines.append(f"      smiles: {smi.strip()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seq", action="append", default=[], help="protein sequence (repeatable)"
    )
    p.add_argument("--dna", action="append", default=[], help="DNA sequence")
    p.add_argument("--rna", action="append", default=[], help="RNA sequence")
    p.add_argument(
        "--ligand-ccd", action="append", default=[], help="ligand CCD code"
    )
    p.add_argument(
        "--ligand-smiles", action="append", default=[], help="ligand SMILES"
    )
    p.add_argument("--use-msa-server", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    text = build_job_yaml(
        args.seq,
        args.ligand_ccd,
        dna=args.dna,
        rna=args.rna,
        ligands_smiles=args.ligand_smiles,
        use_msa_server=args.use_msa_server,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
