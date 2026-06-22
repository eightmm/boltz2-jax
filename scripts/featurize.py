"""Featurization via boltz_jax.data: raw YAML -> model feature dict.

Provides ``featurize_yaml`` (used by ``scripts/predict.py``): runs
check_inputs -> process_inputs -> PredictionDataset over one input, returning the
feature dict, manifest, and processed-structures dir. The featurization runs
without ``import boltz``. Running this module directly featurizes 1UBQ_A and
checks it against outputs/real_features/1UBQ_A.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from boltz_jax.data.module.inferencev2 import PredictionDataset  # noqa: E402
from boltz_jax.data.preprocess import check_inputs, process_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOL_DIR = ROOT / ".cache" / "boltz" / "mols"
REF_NPZ = ROOT / "outputs" / "real_features" / "1UBQ_A.npz"
RECORD_ID = "1UBQ_A"

PREP_DIR = ROOT / "outputs" / "prep" / "1UBQ_A"
YAML_PATH = PREP_DIR / "1UBQ_A.yaml"
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # noqa: E501


def write_input_yaml() -> Path:
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: A\n"
        f"      sequence: {SEQUENCE}\n"
        "      msa: empty\n"
    )
    return YAML_PATH


def _input_digest(yaml_path: Path, mol_dir: Path, opts: tuple) -> str:
    """Content digest of a featurization request.

    Hashes the YAML text, every existing file it references (MSA a3m/csv,
    template cif/pdb), the mol dir path, and the options. Output-identical
    inputs -> identical digest -> cache hit.
    """
    import hashlib
    import re

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
    structure-file writing.

    ``cache_dir`` (opt-in): memoize the final features + structure npz keyed by a
    content digest of the input (YAML + referenced MSA/template files + options).
    A hit skips preprocessing + featurization entirely and returns bit-identical
    arrays. ``manifest`` is None on a cache hit (callers use record_id from the
    structure-dir filename, which is the cached record).
    """
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


def build_features() -> dict[str, np.ndarray]:
    write_input_yaml()
    feats_np, manifest, _ = featurize_yaml(YAML_PATH, PREP_DIR, MOL_DIR)
    records = [r for r in manifest.records if r.id == RECORD_ID]
    assert len(records) == 1, f"record {RECORD_ID} not in manifest"
    return feats_np


def _ref_pos_rigid_match(produced: dict, ref: dict) -> tuple[bool, str]:
    rp_r, rp_o = ref["ref_pos"][0], produced["ref_pos"][0]
    if rp_r.shape != rp_o.shape:
        return False, f"shape {rp_o.shape} vs {rp_r.shape}"
    uid = ref["ref_space_uid"][0]
    res = ref["atom_resolved_mask"][0].astype(bool)

    def pdist(x):
        d = x[:, None, :] - x[None, :, :]
        return np.sqrt((d * d).sum(-1))

    worst = 0.0
    for u in np.unique(uid):
        idx = np.where((uid == u) & res)[0]
        if len(idx) < 2:
            continue
        worst = max(worst, float(np.abs(pdist(rp_r[idx]) - pdist(rp_o[idx])).max()))
    ok = worst < 1e-3
    return ok, f"rigid-invariant: intra-block dist max|d|={worst:.2e} (RNG aug)"


def compare(produced: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> bool:
    pk, rk = set(produced), set(ref)
    missing = sorted(rk - pk)
    extra = sorted(pk - rk)

    rows = []
    overall = True
    for k in sorted(rk & pk):
        if k == "ref_pos":
            ok, detail = _ref_pos_rigid_match(produced, ref)
            rows.append((k, "PASS" if ok else "FAIL", detail))
            overall = overall and ok
            continue
        a, b = produced[k], ref[k]
        if a.shape != b.shape:
            rows.append((k, "FAIL", f"shape {a.shape} vs {b.shape}"))
            overall = False
            continue
        if np.issubdtype(b.dtype, np.floating):
            close = np.allclose(a, b, atol=1e-4, equal_nan=True)
            detail = "" if close else f"max|d|={np.nanmax(np.abs(a - b)):.3e}"
        else:
            close = np.array_equal(a, b)
            detail = "" if close else "int mismatch"
        rows.append((k, "PASS" if close else "FAIL", detail))
        overall = overall and close

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'key':<{w}}  RESULT  detail")
    print("-" * (w + 25))
    for k, res, detail in rows:
        print(f"{k:<{w}}  {res:<6}  {detail}")

    if missing:
        overall = False
        print(f"\nMISSING keys (in ref, not produced): {missing}")
    if extra:
        print(f"\nEXTRA keys (produced, not in ref): {extra}")

    n_pass = sum(1 for _, r, _ in rows if r == "PASS")
    print(
        f"\n{n_pass}/{len(rows)} matched keys PASS; "
        f"{len(missing)} missing, {len(extra)} extra"
    )
    return overall and not missing


def main() -> None:
    assert REF_NPZ.exists(), f"missing reference npz: {REF_NPZ}"
    ref = dict(np.load(REF_NPZ))
    produced = build_features()
    print(f"produced {len(produced)} keys; reference {len(ref)} keys\n")
    overall = compare(produced, ref)
    print("\nOVERALL:", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
