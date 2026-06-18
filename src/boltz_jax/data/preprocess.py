"""Self-contained input preprocessing for the protein (no-MSA) Boltz path.

Extracted from ``boltz/src/boltz/main.py`` (functions ``check_inputs``,
``process_input``, ``process_inputs``) with the excluded paths removed or
guarded: model download, prediction/CLI, affinity, MSA-server search, and
templates. The protein / ``msa: empty`` path runs end-to-end here with no
``import boltz``.

``process_inputs`` writes the standard ``out_dir/processed/`` tree
(``structures/``, ``records/``, ``msa/``, ``constraints/``, ``templates/``,
``mols/`` + ``manifest.json``) consumable by
``boltz_jax.data.module.inferencev2.PredictionDataset``.
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

from boltz_jax.data import const
from boltz_jax.data.mol import load_canonicals
from boltz_jax.data.parse.fasta import parse_fasta
from boltz_jax.data.parse.yaml import parse_yaml
from boltz_jax.data.types import Manifest, Record


def check_inputs(data: Path) -> list[Path]:
    """Validate the input path and return the list of input files."""
    if data.is_dir():
        files: list[Path] = list(data.glob("*"))
        for d in files:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
        return files
    return [data]


def process_input(
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    """Parse a single input file and dump its processed structure/record tree.

    Protein / ``msa: empty`` only: any protein chain that still requests an MSA
    (``msa_id == 0``) raises, since the MSA-server search path is excluded.
    """
    # Parse data
    if path.suffix.lower() in (".fa", ".fas", ".fasta"):
        target = parse_fasta(path, ccd, mol_dir, boltz2)
    elif path.suffix.lower() in (".yml", ".yaml"):
        target = parse_yaml(path, ccd, mol_dir, boltz2)
    else:
        msg = (
            f"Unable to parse filetype {path.suffix}, "
            "please provide a .fasta or .yaml file."
        )
        raise RuntimeError(msg)

    target_id = target.record.id

    # Decide whether any MSA generation would be needed (excluded path).
    to_generate = {}
    prot_id = const.chain_type_ids["PROTEIN"]
    for chain in target.record.chains:
        if (chain.mol_type == prot_id) and (chain.msa_id == 0):
            entity_id = chain.entity_id
            msa_id = f"{target_id}_{entity_id}"
            to_generate[msa_id] = target.sequences[entity_id]
            chain.msa_id = msa_dir / f"{msa_id}.csv"
        elif chain.msa_id == 0:
            chain.msa_id = -1

    if to_generate and not use_msa_server:
        msg = (
            "Missing MSA's in input and MSA-server search is not supported in "
            "boltz_jax.data.preprocess (protein no-MSA path only). Use "
            "`msa: empty` per protein chain."
        )
        raise RuntimeError(msg)

    # Parse MSA data (csv/a3m). For the no-MSA path this list is empty.
    msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
    msa_id_map = {}
    for msa_idx, msa_id in enumerate(msas):
        msa_path = Path(msa_id)
        if not msa_path.exists():
            msg = f"MSA file {msa_path} not found."
            raise FileNotFoundError(msg)
        processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
        msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
        if not processed.exists():
            if msa_path.suffix == ".a3m":
                from boltz_jax.data.parse.a3m import parse_a3m

                msa = parse_a3m(msa_path, taxonomy=None, max_seqs=8192)
            elif msa_path.suffix == ".csv":
                from boltz_jax.data.parse.csv import parse_csv

                msa = parse_csv(msa_path, max_seqs=8192)
            else:
                msg = f"MSA file {msa_path} not supported, only a3m or csv."
                raise RuntimeError(msg)
            msa.dump(processed)

    for c in target.record.chains:
        if (c.msa_id != -1) and (c.msa_id in msa_id_map):
            c.msa_id = msa_id_map[c.msa_id]

    # Dump templates (empty for the protein no-MSA path).
    for template_id, template in target.templates.items():
        name = f"{target.record.id}_{template_id}.npz"
        template.dump(processed_templates_dir / name)

    # Dump constraints.
    constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
    target.residue_constraints.dump(constraints_path)

    # Dump extra molecules.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
        pickle.dump(target.extra_mols, f)

    # Dump structure and record.
    target.structure.dump(structure_dir / f"{target.record.id}.npz")
    target.record.dump(records_dir / f"{target.record.id}.json")


def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    use_msa_server: bool = False,
    boltz2: bool = True,
) -> Manifest:
    """Process the input data, writing the ``processed/`` tree + manifest.

    Returns the loaded :class:`Manifest`.
    """
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    for d in (
        out_dir,
        msa_dir,
        records_dir,
        structure_dir,
        processed_msa_dir,
        processed_constraints_dir,
        processed_templates_dir,
        processed_mols_dir,
        predictions_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Load CCD / canonical molecules.
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    for path in tqdm(data):
        process_input_partial(path)

    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    return manifest
