"""Geometry / clash sanity check for a JAX-sampled Boltz-2 structure.

Samples a structure with ``boltz2_sample_forward`` (deterministic serving mode,
augmentation off) and computes basic geometry sanity metrics over the REAL
(unmasked) atoms:

  - number of clashing atom pairs (min-distance below a threshold),
  - global minimum inter-atomic distance,
  - consecutive-CA distance distribution if a backbone is identifiable.

This needs no crystal ground truth. Crystal RMSD vs PDB would require wiring a
reference structure + atom correspondence, which is out of scope here; we note
that explicitly rather than fabricate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import _load_features_pt, _tree_to_jax  # noqa: E402


def _clash_stats(coords: np.ndarray, thresholds: list[float]) -> dict:
    """Count close non-self atom pairs below each threshold; report min dist."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    out = {
        "min_interatomic_distance_angstrom": float(pair_dist.min()),
        "n_atom_pairs": int(pair_dist.size),
        "clash_counts": {
            f"<{t}A": int(np.sum(pair_dist < t)) for t in thresholds
        },
    }
    return out


def _ca_stats(coords: np.ndarray, ca_idx: np.ndarray) -> dict | None:
    """Consecutive-CA distance distribution (ideal ~3.8 A)."""
    if ca_idx is None or ca_idx.size < 2:
        return None
    ca = coords[ca_idx]
    d = np.sqrt(np.sum((ca[1:] - ca[:-1]) ** 2, axis=-1))
    return {
        "n_ca": int(ca_idx.size),
        "n_consecutive_pairs": int(d.size),
        "mean_ca_ca_angstrom": float(d.mean()),
        "std_ca_ca_angstrom": float(d.std()),
        "min_ca_ca_angstrom": float(d.min()),
        "max_ca_ca_angstrom": float(d.max()),
        "frac_within_3.4_4.2A": float(np.mean((d > 3.4) & (d < 4.2))),
    }


def _find_ca_indices(feats: dict, real_idx: np.ndarray) -> np.ndarray | None:
    """Best-effort CA atom indices restricted to real atoms.

    Uses ``ref_atom_name_chars`` if present (4-char one-hot/byte encoding of the
    atom name); otherwise returns None and CA stats are skipped.
    """
    names_key = None
    for k in ("ref_atom_name_chars", "atom_name_chars", "ref_element"):
        if k in feats:
            names_key = k
            break
    if names_key != "ref_atom_name_chars" and "ref_atom_name_chars" not in feats:
        return None
    arr = np.asarray(feats["ref_atom_name_chars"])  # (1, n_atoms, 4, 64) one-hot
    if arr.ndim != 4:
        return None
    # decode 4 chars: argmax over last dim -> ascii offset (Boltz uses c-32).
    idx = arr.argmax(axis=-1)[0]  # (n_atoms, 4)
    chars = (idx + 32).astype(np.uint8)
    names = ["".join(chr(c) for c in row).strip() for row in chars]
    ca_mask = np.array([nm == "CA" for nm in names])
    ca_idx_full = np.where(ca_mask)[0]
    # restrict to real atoms and remap to position within real_idx array
    real_set = set(real_idx.tolist())
    ca_in_real = np.array([i for i in ca_idx_full if i in real_set])
    if ca_in_real.size < 2:
        return None
    pos = {a: p for p, a in enumerate(real_idx.tolist())}
    return np.array([pos[a] for a in ca_in_real])


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"))
    p.add_argument("--features-pt", type=Path,
                   default=Path("outputs/real_features/1UBQ_A.pt"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--token-layers", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/geometry_check.json"))
    args = p.parse_args()

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    jax_params = map_boltz2_graph_state_dict(
        state_cpu, num_msa_layers=4, num_pairformer_layers=64,
        num_token_layers=args.token_layers, token_transformer_heads=16,
    )
    feats_np, record_id = _load_features_pt(args.features_pt)
    jax_feats = _tree_to_jax(feats_np)

    out = boltz2_sample_forward(
        jax_params, jax_feats, jax.random.PRNGKey(args.seed),
        num_sampling_steps=args.steps, recycling_steps=args.recycling,
        token_layers=args.token_layers, augmentation=False,
        alignment_reverse_diff=True,
    )
    coords = np.asarray(out["sample_atom_coords"])[0]  # (n_atoms, 3)

    atom_mask = np.asarray(feats_np["atom_pad_mask"]).reshape(-1).astype(bool)
    real_idx = np.where(atom_mask)[0]
    real_coords = coords[real_idx]

    thresholds = [0.5, 1.0, 1.2, 1.5]
    clash = _clash_stats(real_coords, thresholds)
    ca_pos = _find_ca_indices(feats_np, real_idx)
    ca_stats = _ca_stats(real_coords, ca_pos)

    payload = {
        "record_id": record_id,
        "features_pt": str(args.features_pt),
        "steps": args.steps,
        "recycling": args.recycling,
        "augmentation": False,
        "mode": "deterministic serving mode (augmentation off)",
        "n_atoms_total": int(atom_mask.size),
        "n_real_atoms": int(real_idx.size),
        "clash": clash,
        "ca_backbone": ca_stats,
        "crystal_rmsd_vs_pdb": None,
        "crystal_rmsd_note": (
            "Skipped: needs ground-truth PDB + atom correspondence wiring, "
            "which is not available in this benchmark harness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"=== Geometry check: {record_id} (steps={args.steps}, "
          f"recycling={args.recycling}, aug=off) ===")
    print(f"real atoms: {real_idx.size}")
    print(f"min interatomic distance: "
          f"{clash['min_interatomic_distance_angstrom']:.3f} A")
    for t in thresholds:
        print(f"  clashing pairs <{t}A: {clash['clash_counts'][f'<{t}A']}")
    if ca_stats:
        print(f"CA-CA: n={ca_stats['n_ca']} mean="
              f"{ca_stats['mean_ca_ca_angstrom']:.3f} A "
              f"(within 3.4-4.2A: {ca_stats['frac_within_3.4_4.2A']*100:.1f}%)")
    else:
        print("CA backbone: not identifiable from features (skipped)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
