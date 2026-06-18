"""Padding-invariance check on a MULTI-CHAIN input (multimer, 2 protein chains).

Same contract as scripts/check_padding_invariance.py, but on multimer feats so
the per-chain features (asym_id / entity_id / sym_id) and cross-chain pair
masking are exercised. Padding a real input up to a larger (token, atom) bucket
must not change the real (unmasked) atom outputs.

Reuses ``pad_feats`` from check_padding_invariance.py.

Run on CPU:
  JAX_PLATFORMS=cpu uv run python scripts/check_padding_invariance_multichain.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from check_padding_invariance import (  # noqa: E402  (sibling script import)
    _kabsch_rmsd,
    _raw_rmsd,
    pad_feats,
)

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward, boltz2_trunk_forward

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "outputs" / "real_features" / "multimer.npz"
WEIGHTS_PATH = ROOT / "outputs" / "native_weights" / "boltz2_conf.safetensors"
OUT_JSON = ROOT / "outputs" / "padding_invariance_multichain.json"

# Multimer has ~3x the atoms of the single-chain case; fewer sampling steps keep
# the CPU run tractable. Invariance does not depend on step count.
NUM_SAMPLING_STEPS = 10
RECYCLING_STEPS = 3
SEED = 0


def run_config(params, feats, init_noise, step_noises):
    out = boltz2_sample_forward(
        params,
        feats,
        jax.random.PRNGKey(0),
        recycling_steps=RECYCLING_STEPS,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        augmentation=False,
        alignment_reverse_diff=True,
        use_scan=False,
        init_noise=jnp.asarray(init_noise),
        step_noises=jnp.asarray(step_noises),
    )
    return np.asarray(out["sample_atom_coords"])


def run_trunk(params, feats):
    trunk = boltz2_trunk_forward(
        params["trunk"], feats, recycling_steps=RECYCLING_STEPS, eps=1e-5
    )
    return np.asarray(trunk["s"]), np.asarray(trunk["z"])


def pad_noise(noise: np.ndarray, target_atoms: int) -> np.ndarray:
    atom_axis = noise.ndim - 2
    out = np.zeros(
        noise.shape[:atom_axis] + (target_atoms,) + noise.shape[atom_axis + 1 :],
        dtype=noise.dtype,
    )
    sl = [slice(None)] * noise.ndim
    sl[atom_axis] = slice(0, noise.shape[atom_axis])
    out[tuple(sl)] = noise
    return out


def main() -> None:
    feats = load_features_npz(FEATS_PATH)
    params = load_params(WEIGHTS_PATH)

    t0 = int(feats["token_pad_mask"].shape[-1])
    a0 = int(feats["atom_pad_mask"].shape[-1])
    atom_mask0 = np.asarray(feats["atom_pad_mask"])[0].astype(bool)
    n_real_atoms = int(atom_mask0.sum())
    n_chains = int(np.unique(np.asarray(feats["asym_id"])[0]).size)
    print(
        f"baseline: t0={t0} a0={a0} n_real_atoms={n_real_atoms} chains={n_chains}"
    )

    rng = np.random.default_rng(SEED)
    init_noise_real = rng.standard_normal((1, a0, 3)).astype(np.float32)
    step_noises_real = rng.standard_normal(
        (NUM_SAMPLING_STEPS, 1, a0, 3)
    ).astype(np.float32)

    # Pick buckets strictly larger than the real (228, 1792) shape. Kept modest
    # so the CPU run completes; padding by a few tokens / a few hundred atoms is
    # enough to exercise the masking (any nonzero pad would leak if buggy).
    configs = [
        ("baseline", t0, a0),
        ("pad_256_1920", 256, 1920),
    ]

    base_coords = run_config(params, feats, init_noise_real, step_noises_real)
    base_real = base_coords[0][:a0][atom_mask0]
    base_s, base_z = run_trunk(params, feats)
    base_s_real = base_s[0][:t0]
    base_z_real = base_z[0][:t0, :t0]

    pad_log_recorded = None
    results = {}
    for name, tt, ta in configs:
        if name == "baseline":
            results[name] = {
                "real_atom_raw_rmsd": 0.0,
                "real_atom_aligned_rmsd": 0.0,
                "real_atom_max_abs_diff": 0.0,
                "trunk_s_max_abs_diff": 0.0,
                "trunk_z_max_abs_diff": 0.0,
                "n_real_atoms": n_real_atoms,
                "shape": [tt, ta],
            }
            continue

        pfeats, plog = pad_feats(feats, tt, ta)
        if pad_log_recorded is None:
            pad_log_recorded = plog
        pinit = pad_noise(init_noise_real, ta)
        pstep = pad_noise(step_noises_real, ta)
        assert np.array_equal(pinit[:, :a0], init_noise_real)
        assert np.array_equal(pstep[:, :, :a0], step_noises_real)

        pcoords = run_config(params, pfeats, pinit, pstep)
        preal = pcoords[0][:a0][atom_mask0]

        raw = _raw_rmsd(preal, base_real)
        aligned = _kabsch_rmsd(preal, base_real)
        maxabs = float(np.abs(preal - base_real).max())

        ps, pz = run_trunk(params, pfeats)
        s_diff = float(np.abs(ps[0][:t0] - base_s_real).max())
        z_diff = float(np.abs(pz[0][:t0, :t0] - base_z_real).max())

        results[name] = {
            "real_atom_raw_rmsd": raw,
            "real_atom_aligned_rmsd": aligned,
            "real_atom_max_abs_diff": maxabs,
            "trunk_s_max_abs_diff": s_diff,
            "trunk_z_max_abs_diff": z_diff,
            "n_real_atoms": n_real_atoms,
            "shape": [tt, ta],
        }

    print("\n=== PADDED KEYS (first padded config) ===")
    for line in pad_log_recorded:
        print("  " + line)

    print("\n=== MULTI-CHAIN PADDING INVARIANCE (real-atom slice) ===")
    cols = ["config", "raw_rmsd", "aligned_rmsd", "max_abs", "trunk_s", "trunk_z"]
    hdr = (
        f"{cols[0]:14s} {cols[1]:>12s} {cols[2]:>14s} "
        f"{cols[3]:>12s} {cols[4]:>12s} {cols[5]:>12s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, _, _ in configs:
        r = results[name]
        print(
            f"{name:14s} {r['real_atom_raw_rmsd']:12.3e} "
            f"{r['real_atom_aligned_rmsd']:14.3e} {r['real_atom_max_abs_diff']:12.3e} "
            f"{r['trunk_s_max_abs_diff']:12.3e} {r['trunk_z_max_abs_diff']:12.3e}"
        )

    payload = {
        "feats": str(FEATS_PATH),
        "weights": str(WEIGHTS_PATH),
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "recycling_steps": RECYCLING_STEPS,
        "seed": SEED,
        "platform": "cpu",
        "n_chains": n_chains,
        "n_real_atoms": n_real_atoms,
        "configs": results,
        "padded_keys": pad_log_recorded,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_JSON}")

    worst_raw = max(
        results[n]["real_atom_raw_rmsd"] for n, _, _ in configs if n != "baseline"
    )
    print(f"\nworst real-atom raw RMSD over padded configs: {worst_raw:.3e} A")
    verdict = "SAFE (padding-invariant)" if worst_raw < 1e-2 else "LEAK (NOT invariant)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
