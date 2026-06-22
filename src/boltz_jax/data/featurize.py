"""Featurization: raw YAML job -> model feature dict (no ``import boltz``).

``featurize_yaml`` runs check_inputs -> process_inputs -> PredictionDataset over
one input and returns the batched feature dict, the manifest, and the processed
structures dir. Requires torch (featurization is torch-side); the JAX runtime
does not import this. ``cache_dir`` memoizes features by a content digest of the
input (YAML + referenced MSA/template files + options).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from boltz_jax.data.module.inferencev2 import PredictionDataset
from boltz_jax.data.preprocess import check_inputs, process_inputs


def _input_digest(yaml_path: Path, mol_dir: Path, opts: tuple) -> str:
    """Content digest of a featurization request.

    Hashes the YAML text, every existing file it references (MSA a3m/csv,
    template cif/pdb), the mol dir path, and the options. Output-identical
    inputs -> identical digest -> cache hit.
    """
    h = hashlib.sha256()
    text = yaml_path.read_bytes()
    h.update(text)
    h.update(str(mol_dir).encode())
    h.update(repr(opts).encode())
    # Fold in the content of any referenced file that exists on disk. Only
    # consider path-like tokens (contain '/' or a file extension) and short
    # enough to be a real path -- avoids treating long sequence strings as paths.
    for tok in re.findall(rb"[\w./@-]+", text):
        s = tok.decode("utf-8", "ignore")
        if len(s) > 255 or ("/" not in s and "." not in s):
            continue
        try:
            p = Path(s)
            if not p.is_file():
                continue
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            continue
    return h.hexdigest()[:16]


def featurize_yaml(
    yaml_path: Path,
    out_dir: Path,
    mol_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    cache_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], object, Path]:
    """Run preprocessing + featurization for one YAML.

    Returns ``(feats_np, manifest, processed_structures_dir)`` where
    ``feats_np`` maps feature name -> batched numpy array (leading batch dim 1),
    and the structures dir holds the processed ``{id}.npz`` files used for
    structure-file writing. ``manifest`` is None on a cache hit (callers use
    record_id from the structure-dir filename, which is the cached record).
    """
    import torch

    assert mol_dir.exists(), f"missing mols dir: {mol_dir}"

    opts = (use_msa_server, msa_server_url, msa_pairing_strategy)
    cache_entry = None
    if cache_dir is not None:
        digest = _input_digest(yaml_path, mol_dir, opts)
        cache_entry = Path(cache_dir) / digest
        feats_file = cache_entry / "feats.npz"
        struct_dir = cache_entry / "structures"
        if feats_file.is_file() and struct_dir.is_dir():
            loaded = np.load(feats_file)
            feats_np = {k: loaded[k] for k in loaded.files}
            return feats_np, None, struct_dir

    data = check_inputs(yaml_path)
    assert data, "check_inputs returned nothing"

    manifest = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=out_dir / "unused_ccd.pkl",  # boltz2=True uses canonicals
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        boltz2=True,
    )

    processed = out_dir / "processed"
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=mol_dir,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
    )
    features = dataset[0]

    feats_np: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if key.startswith("_") or key == "record":
            continue
        if not torch.is_tensor(value):
            continue
        feats_np[key] = value.unsqueeze(0).detach().cpu().numpy()

    struct_dir = processed / "structures"
    if cache_entry is not None:
        import shutil

        out_struct = cache_entry / "structures"
        out_struct.mkdir(parents=True, exist_ok=True)
        for rec in manifest.records:
            src = struct_dir / f"{rec.id}.npz"
            if src.is_file():
                shutil.copy2(src, out_struct / f"{rec.id}.npz")
        np.savez(cache_entry / "feats.npz", **feats_np)
        struct_dir = out_struct
    return feats_np, manifest, struct_dir
