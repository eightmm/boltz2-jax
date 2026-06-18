"""Crystal CA RMSD: JAX Boltz-2 prediction for 1UBQ_A vs bundled crystal coords.

The features bundle (outputs/real_features/1UBQ_A.npz, mirror of the .pt) carries
the deposited experimental structure under key ``coords`` (1, 1, A, 3) -- the
boltz record crystal pose, distinct from ``ref_pos`` (per-atom reference
conformer). We use ``coords`` as the crystal ground truth.

CA atoms are the per-token center atoms (``token_to_center_atom``, one-hot
(b, T, A)) for resolved PROTEIN tokens (mol_type == 0). RMSD is Kabsch-aligned.

Sampling uses the default serving config: augmentation=True, alignment on,
recycling_steps=3, num_sampling_steps=200 (falls back to 50 on CPU if 200
exceeds the time budget; the actual count is recorded in the output JSON).

Run on CPU:
  JAX_PLATFORMS=cpu uv run python scripts/crystal_rmsd.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "outputs" / "real_features" / "1UBQ_A.npz"
WEIGHTS_PATH = ROOT / "outputs" / "native_weights" / "boltz2_conf.safetensors"
OUT_JSON = ROOT / "outputs" / "crystal_rmsd.json"

NUM_SAMPLING_STEPS = 200
FALLBACK_SAMPLING_STEPS = 50
TIME_BUDGET_S = 600.0  # if a 200-step run is projected to exceed this, use 50
RECYCLING_STEPS = 3
SEED = 0
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


def _raw_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def main() -> None:
    feats = load_features_npz(FEATS_PATH)
    params = load_params(WEIGHTS_PATH)

    a0 = int(feats["atom_pad_mask"].shape[-1])
    atom_pad = np.asarray(feats["atom_pad_mask"])[0].astype(bool)  # (A,)
    atom_resolved = np.asarray(feats["atom_resolved_mask"])[0].astype(bool)
    mol_type = np.asarray(feats["mol_type"])[0]  # (T,)
    token_pad = np.asarray(feats["token_pad_mask"])[0].astype(bool)
    token_resolved = np.asarray(feats["token_resolved_mask"])[0].astype(bool)
    t2c = np.asarray(feats["token_to_center_atom"])[0]  # (T, A) one-hot

    # Crystal ground truth from the boltz record.
    crystal = np.asarray(feats["coords"])[0, 0]  # (A, 3)

    # Guard: these feats were dumped in pure prediction mode (sequence-only
    # yaml), so ``coords`` is an all-zero placeholder -- there is NO deposited
    # structure to compare against. Detect and refuse to emit a bogus RMSD.
    resolved_real = atom_pad & atom_resolved
    if np.allclose(crystal[resolved_real], 0.0):
        msg = (
            "feats['coords'] is all-zero on resolved real atoms: this features "
            "bundle carries no deposited/experimental structure (dumped in "
            "prediction mode from a sequence-only yaml). Crystal RMSD is "
            "undefined. Re-dump feats from an input that includes the deposited "
            "structure (CIF/PDB record) to obtain real ground truth."
        )
        print("BLOCKED: " + msg)
        OUT_JSON.write_text(json.dumps({
            "blocked": True,
            "reason": msg,
            "feats": str(FEATS_PATH),
            "n_resolved_real_atoms": int(resolved_real.sum()),
        }, indent=2))
        print(f"wrote {OUT_JSON}")
        return

    # CA = center atom index for each resolved protein token.
    center_atom_idx = t2c.argmax(axis=1)  # (T,)
    has_center = t2c.sum(axis=1) > 0
    prot_tok = (
        token_pad & token_resolved & has_center & (mol_type == PROTEIN_MOL_TYPE)
    )
    ca_idx = center_atom_idx[prot_tok]
    # Keep only CAs that are resolved atoms (crystal coord present).
    ca_idx = np.array([i for i in ca_idx if atom_resolved[i]])
    n_ca = int(ca_idx.size)

    real_atoms = atom_pad & atom_resolved  # (A,)
    n_real = int(real_atoms.sum())
    print(f"a0={a0} n_real_atoms={n_real} n_CA={n_ca}")

    # ---- run JAX sampler (default serving: augmentation on, alignment on) ----
    # Probe latency with a short run, then decide 200 vs 50 steps so a slow CPU
    # cannot blow past the budget.
    def _sample(steps: int) -> np.ndarray:
        out = boltz2_sample_forward(
            params,
            feats,
            jax.random.PRNGKey(SEED),
            recycling_steps=RECYCLING_STEPS,
            num_sampling_steps=steps,
            augmentation=True,
            alignment_reverse_diff=True,
            use_scan=False,
        )
        return np.asarray(out["sample_atom_coords"])[0][:a0]  # (A, 3)

    probe_steps = 5
    t0_probe = time.perf_counter()
    _sample(probe_steps)  # also triggers JIT compile
    per_step = (time.perf_counter() - t0_probe) / probe_steps
    projected_200 = per_step * NUM_SAMPLING_STEPS
    steps = NUM_SAMPLING_STEPS if projected_200 <= TIME_BUDGET_S else (
        FALLBACK_SAMPLING_STEPS
    )
    print(
        f"probe: ~{per_step:.2f}s/step, projected 200-step={projected_200:.0f}s "
        f"-> using {steps} sampling steps"
    )
    t0_run = time.perf_counter()
    pred = _sample(steps)
    print(f"sampling ({steps} steps) took {time.perf_counter() - t0_run:.0f}s")

    # ---- CA RMSD ----
    ca_pred = pred[ca_idx]
    ca_ref = crystal[ca_idx]
    ca_raw = _raw_rmsd(ca_pred, ca_ref)
    ca_aligned = _kabsch_rmsd(ca_pred, ca_ref)

    # ---- all-atom RMSD (resolved real atoms) ----
    aa_pred = pred[real_atoms]
    aa_ref = crystal[real_atoms]
    aa_raw = _raw_rmsd(aa_pred, aa_ref)
    aa_aligned = _kabsch_rmsd(aa_pred, aa_ref)

    print(f"CA  : raw={ca_raw:.3f}  aligned={ca_aligned:.3f} A")
    print(f"all : raw={aa_raw:.3f}  aligned={aa_aligned:.3f} A")

    payload = {
        "feats": str(FEATS_PATH),
        "weights": str(WEIGHTS_PATH),
        "reference": (
            "bundled crystal coords (feats['coords'][0,0], boltz record "
            "deposited structure for 1UBQ_A) -- NOT ref_pos conformer"
        ),
        "num_sampling_steps": steps,
        "augmentation": True,
        "recycling_steps": RECYCLING_STEPS,
        "seed": SEED,
        "platform": "cpu",
        "n_real_atoms": n_real,
        "n_CA_atoms": n_ca,
        "ca_rmsd_raw_A": ca_raw,
        "ca_rmsd_aligned_A": ca_aligned,
        "all_atom_rmsd_raw_A": aa_raw,
        "all_atom_rmsd_aligned_A": aa_aligned,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
