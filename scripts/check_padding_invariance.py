"""Verify the JAX Boltz-2 sampler is PADDING-INVARIANT.

Padding a real input up to a larger (token, atom) shape bucket must NOT change
the real (unmasked) atom outputs. This validates the shape-bucketing +
precompile-cache serving strategy.

Run on CPU:  JAX_PLATFORMS=cpu uv run python scripts/check_padding_invariance.py

Only adds a script; does not touch src/. If a leak is found, it is reported,
not fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward, boltz2_trunk_forward

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "outputs" / "real_features" / "1UBQ_A.npz"
WEIGHTS_PATH = ROOT / "outputs" / "native_weights" / "boltz2_conf.safetensors"
OUT_JSON = ROOT / "outputs" / "padding_invariance.json"

NUM_SAMPLING_STEPS = 25
RECYCLING_STEPS = 3
SEED = 0

# Per-axis padding semantics. We inspect every key and classify it by which
# axis (token T or atom A) each dimension corresponds to, using the native
# sizes (t0=76, a0=608). A few keys are index-arrays into the atom axis
# (frames_idx) and are handled specially.
#
# Classification is done dynamically by matching dim sizes against t0 / a0.


def _pad_to(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    """Zero-pad ``arr`` along ``axis`` up to ``target`` (no-op if already >=)."""
    cur = arr.shape[axis]
    if cur >= target:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, target - cur)
    return np.pad(arr, pad, mode="constant", constant_values=0)


def pad_feats(
    feats: dict[str, jnp.ndarray],
    target_tokens: int,
    target_atoms: int,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    """Zero-pad every per-token / per-atom / per-pair array to the target sizes.

    Returns the new feats dict and a human-readable log of how each key was
    padded.

    Strategy
    --------
    * All padded entries are 0. Pad masks (token_pad_mask, atom_pad_mask) are
      therefore 0 on padded positions and unchanged (1) on real ones.
    * atom<->token one-hot maps (atom_to_token (b,A,T), token_to_rep_atom /
      token_to_center_atom / r_set_to_rep_atom (b,T,A)) are zero-padded on both
      axes. Because they are one-hot, padded atoms/tokens get all-zero rows AND
      columns, so no real index points into the padded region and vice-versa.
    * frames_idx (b,1,T,3) holds integer atom indices. Real tokens keep their
      real atom indices; padded tokens get 0 (a valid real atom index, but the
      frame is masked via frame_resolved_mask which is zero-padded to False).
    * msa arrays (b,depth,T): pad the token axis, keep msa depth.
    * pair arrays (b,T,T,...): pad both token axes.
    """
    t0 = int(feats["token_pad_mask"].shape[-1])  # 76
    a0 = int(feats["atom_pad_mask"].shape[-1])  # 608
    log: list[str] = []

    out: dict[str, np.ndarray] = {}
    for k, v in feats.items():
        arr = np.asarray(v)
        orig_dtype = arr.dtype

        if k == "frames_idx":
            # (b, 1, T, 3) integer atom indices. Pad token axis with 0.
            new = _pad_to(arr, 2, target_tokens)
            out[k] = new.astype(orig_dtype)
            log.append(
                f"{k}: frames_idx, pad token axis(2) {t0}->{target_tokens} with 0"
            )
            continue

        # Generic dim-by-dim classification: a dim equal to a0 is an atom axis,
        # a dim equal to t0 is a token axis. Atom axis takes priority when
        # t0==a0 (not the case here). Dims equal to neither are left alone.
        new = arr
        actions = []
        for ax, dim in enumerate(arr.shape):
            if dim == a0:
                new = _pad_to(new, ax, target_atoms)
                actions.append(f"atom@{ax}:{a0}->{target_atoms}")
            elif dim == t0:
                new = _pad_to(new, ax, target_tokens)
                actions.append(f"token@{ax}:{t0}->{target_tokens}")
        out[k] = new.astype(orig_dtype)
        if actions:
            log.append(f"{k}: {arr.shape} -> {new.shape} [{', '.join(actions)}]")
        else:
            log.append(f"{k}: {arr.shape} unchanged (no token/atom axis)")

    return {k: jnp.asarray(v) for k, v in out.items()}, log


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of a onto b after optimal rigid alignment (no scaling)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T
    a_rot = a @ rot.T
    return float(np.sqrt(((a_rot - b) ** 2).sum(1).mean()))


def _raw_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


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


def main() -> None:
    feats = load_features_npz(FEATS_PATH)
    params = load_params(WEIGHTS_PATH)

    t0 = int(feats["token_pad_mask"].shape[-1])
    a0 = int(feats["atom_pad_mask"].shape[-1])
    atom_mask0 = np.asarray(feats["atom_pad_mask"])[0].astype(bool)  # (a0,)
    n_real_atoms = int(atom_mask0.sum())
    print(f"baseline: t0={t0} a0={a0} n_real_atoms(atom_pad_mask==1)={n_real_atoms}")

    # ---- identical injected noise on real atoms ----
    rng = np.random.default_rng(SEED)
    init_noise_real = rng.standard_normal((1, a0, 3)).astype(np.float32)
    step_noises_real = rng.standard_normal(
        (NUM_SAMPLING_STEPS, 1, a0, 3)
    ).astype(np.float32)

    def pad_noise(noise: np.ndarray, target_atoms: int) -> np.ndarray:
        atom_axis = noise.ndim - 2  # second to last
        out = np.zeros(
            noise.shape[:atom_axis] + (target_atoms,) + noise.shape[atom_axis + 1 :],
            dtype=noise.dtype,
        )
        sl = [slice(None)] * noise.ndim
        sl[atom_axis] = slice(0, noise.shape[atom_axis])
        out[tuple(sl)] = noise
        return out

    configs = [
        ("baseline", t0, a0),
        ("pad_128_1024", 128, 1024),
        ("pad_256_2048", 256, 2048),
    ]

    # baseline
    base_feats = feats
    base_coords = run_config(params, base_feats, init_noise_real, step_noises_real)
    base_real = base_coords[0][:a0][atom_mask0]  # (n_real, 3)
    base_s, base_z = run_trunk(params, base_feats)
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
        # sanity: padded noise real slice identical
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

    # ---- report ----
    print("\n=== PADDED KEYS (pad_128_1024) ===")
    for line in pad_log_recorded:
        print("  " + line)

    print("\n=== PADDING INVARIANCE (real-atom slice, atom_pad_mask==1) ===")
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
