"""Shape-bucketing helpers: zero-pad a feature dict to (token, atom) buckets.

Padding a real input up to a larger ``(token, atom)`` shape bucket does not
change the real (unmasked) atom outputs, so padded inputs can share a single
compiled program / persistent-cache entry across different-length targets
(serving). Pad masks are zero on padded positions, so masked reductions ignore
them. Used by ``scripts/predict.py --bucket``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


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
    t0 = int(feats["token_pad_mask"].shape[-1])
    a0 = int(feats["atom_pad_mask"].shape[-1])
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
        # a dim equal to t0 is a token axis. Dims equal to neither are left alone.
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
