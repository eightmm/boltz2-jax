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


def build_job_yaml(
    proteins: Sequence[str],
    ligands_ccd: Sequence[str] = (),
    *,
    use_msa_server: bool = False,
) -> str:
    """Return a Boltz-2 job YAML string.

    proteins: protein sequences (one chain each).
    ligands_ccd: ligand CCD codes (e.g. ``ATP``).
    use_msa_server: if False, each protein gets ``msa: empty`` (single-sequence);
        if True, the ``msa`` key is omitted so predict's ``--use-msa-server`` can
        generate one.
    """
    if not proteins and not ligands_ccd:
        raise ValueError("need at least one protein sequence or ligand CCD code")
    ids = iter(string.ascii_uppercase)
    lines = ["version: 1", "sequences:"]
    for seq in proteins:
        seq = seq.strip().upper()
        if not seq.isalpha():
            raise ValueError(f"protein sequence must be alphabetic: {seq!r}")
        lines += ["  - protein:", f"      id: {next(ids)}", f"      sequence: {seq}"]
        if not use_msa_server:
            lines.append("      msa: empty")
    for ccd in ligands_ccd:
        lines += ["  - ligand:", f"      id: {next(ids)}", f"      ccd: {ccd.strip()}"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seq", action="append", default=[], help="protein sequence (repeatable)"
    )
    p.add_argument(
        "--ligand-ccd", action="append", default=[], help="ligand CCD code (repeatable)"
    )
    p.add_argument("--use-msa-server", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    text = build_job_yaml(
        args.seq, args.ligand_ccd, use_msa_server=args.use_msa_server
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
