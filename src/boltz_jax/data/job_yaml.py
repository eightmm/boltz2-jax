"""Build a Boltz-2 job YAML from bare sequences / ligand codes.

Convenience for callers who don't want to hand-write the YAML schema (upstream
Boltz parses user-written .yaml/.fasta; it does not generate them). Chains get
auto IDs A, B, ..., Z, AA, ... (proteins, then dna, rna, ligands).
"""

from __future__ import annotations

import string
from collections.abc import Sequence


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
        if True, the ``msa`` key is omitted so an MSA can be generated. (Only
        proteins use MSA.)
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
