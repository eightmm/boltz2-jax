"""Crystal CA RMSD: JAX Boltz-2 prediction for 1UBQ vs the RCSB deposited PDB.

The bundled feats (outputs/real_features/1UBQ_A.npz) carry an all-zero
``coords`` placeholder (sequence-only prediction dump), so the crystal ground
truth must come from the deposited structure, NOT the feats. We fetch
https://files.rcsb.org/download/1UBQ.pdb, parse chain A CA atoms (ATOM records,
residue order) -> (76, 3), and Kabsch-align the predicted CA onto it.

The feats have 76 protein tokens (mol_type == 0) with residue_index 0..75 in
order, i.e. token i corresponds to ubiquitin residue i+1, matching the PDB
residue numbering 1..76.

Run on CPU:
  JAX_PLATFORMS=cpu uv run python scripts/crystal_rmsd_pdb.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import jax
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "outputs" / "real_features" / "1UBQ_A.npz"
WEIGHTS_PATH = ROOT / "outputs" / "native_weights" / "boltz2_conf.safetensors"
OUT_JSON = ROOT / "outputs" / "crystal_rmsd_pdb.json"

PDB_ID = "1UBQ"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
N_UBQ_RES = 76  # ubiquitin monomer, residues 1..76

NUM_SAMPLING_STEPS = 200
FALLBACK_SAMPLING_STEPS = 100
SECOND_FALLBACK_STEPS = 50
TIME_BUDGET_S = 600.0
RECYCLING_STEPS = 3
SEEDS = (0, 1, 2)
PROTEIN_MOL_TYPE = 0


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of a onto b after optimal rigid alignment (no scaling)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = a @ rot.T
    return float(np.sqrt(((a_rot - b) ** 2).sum(1).mean()))


def fetch_deposited_ca() -> tuple[np.ndarray, list[int]]:
    """Fetch RCSB PDB, parse chain A protein CA coords in residue-number order.

    Returns (coords (N,3), residue_seq_numbers). Fixed-column PDB ATOM parse:
    atom name cols 13-16, chain col 22, resSeq cols 23-26, x/y/z cols 31-54.
    Takes the first CA per residue (altLoc A or blank wins by first occurrence).
    """
    with urllib.request.urlopen(PDB_URL, timeout=60) as resp:  # noqa: S310
        text = resp.read().decode("ascii", errors="replace")

    ca_by_res: dict[int, np.ndarray] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16]
        chain = line[21:22]
        if atom_name != " CA " or chain != "A":
            continue
        res_seq = int(line[22:26])
        if res_seq in ca_by_res:
            continue  # keep first altLoc
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        ca_by_res[res_seq] = np.array([x, y, z], dtype=np.float64)

    res_order = sorted(ca_by_res)
    coords = np.stack([ca_by_res[r] for r in res_order], axis=0)
    return coords, res_order


def main() -> None:
    feats = load_features_npz(FEATS_PATH)
    params = load_params(WEIGHTS_PATH)

    a0 = int(feats["atom_pad_mask"].shape[-1])
    atom_resolved = np.asarray(feats["atom_resolved_mask"])[0].astype(bool)
    mol_type = np.asarray(feats["mol_type"])[0]
    token_pad = np.asarray(feats["token_pad_mask"])[0].astype(bool)
    token_resolved = np.asarray(feats["token_resolved_mask"])[0].astype(bool)
    residue_index = np.asarray(feats["residue_index"])[0]
    t2c = np.asarray(feats["token_to_center_atom"])[0]  # (T, A) one-hot

    # --- deposited crystal CA from RCSB ---
    crystal_ca, res_order = fetch_deposited_ca()
    print(
        f"fetched {PDB_ID}: {crystal_ca.shape[0]} CA, "
        f"resSeq {res_order[:3]}..{res_order[-3:]}"
    )
    assert crystal_ca.shape[0] == N_UBQ_RES, (
        f"expected {N_UBQ_RES} deposited CA, got {crystal_ca.shape[0]}"
    )

    # --- predicted CA: center atom of each protein token, in token order ---
    center_atom_idx = t2c.argmax(axis=1)  # (T,)
    has_center = t2c.sum(axis=1) > 0
    prot_mask = (
        token_pad & token_resolved & has_center & (mol_type == PROTEIN_MOL_TYPE)
    )
    # token order == residue_index order (already 0..75 ascending). Sort to be safe.
    prot_tokens = np.nonzero(prot_mask)[0]
    prot_tokens = prot_tokens[np.argsort(residue_index[prot_tokens])]
    ca_idx = center_atom_idx[prot_tokens]
    ca_idx = ca_idx[atom_resolved[ca_idx]]
    n_ca = int(ca_idx.size)
    print(f"a0={a0} predicted protein CA tokens={n_ca}")
    assert n_ca == N_UBQ_RES, f"feats give {n_ca} CA, expected {N_UBQ_RES}"

    def _sample(steps: int, seed: int) -> np.ndarray:
        out = boltz2_sample_forward(
            params,
            feats,
            jax.random.PRNGKey(seed),
            recycling_steps=RECYCLING_STEPS,
            num_sampling_steps=steps,
            augmentation=True,
            alignment_reverse_diff=True,
            use_scan=False,
        )
        return np.asarray(out["sample_atom_coords"])[0][:a0]

    # Probe latency (also JIT-warms), then pick a step count under budget.
    probe_steps = 5
    t0 = time.perf_counter()
    _sample(probe_steps, SEEDS[0])
    per_step = (time.perf_counter() - t0) / probe_steps
    proj200 = per_step * NUM_SAMPLING_STEPS
    if proj200 <= TIME_BUDGET_S:
        steps = NUM_SAMPLING_STEPS
    elif per_step * FALLBACK_SAMPLING_STEPS <= TIME_BUDGET_S:
        steps = FALLBACK_SAMPLING_STEPS
    else:
        steps = SECOND_FALLBACK_STEPS
    print(f"probe ~{per_step:.2f}s/step, proj 200={proj200:.0f}s -> {steps} steps")

    per_seed = {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        pred = _sample(steps, seed)
        rmsd = _kabsch_rmsd(pred[ca_idx], crystal_ca)
        per_seed[str(seed)] = rmsd
        print(f"seed {seed}: CA RMSD = {rmsd:.3f} A  ({time.perf_counter()-t0:.0f}s)")

    mean_rmsd = float(np.mean(list(per_seed.values())))
    print(f"mean CA RMSD = {mean_rmsd:.3f} A over seeds {SEEDS}")

    payload = {
        "pdb_id": PDB_ID,
        "source": "RCSB deposited",
        "pdb_url": PDB_URL,
        "feats": str(FEATS_PATH),
        "weights": str(WEIGHTS_PATH),
        "n_ca": n_ca,
        "steps": steps,
        "recycling_steps": RECYCLING_STEPS,
        "augmentation": True,
        "alignment_reverse_diff": True,
        "platform": "cpu",
        "seeds": list(SEEDS),
        "ca_rmsd_A": {"per_seed": per_seed, "mean": mean_rmsd},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
