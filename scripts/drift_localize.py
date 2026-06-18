"""Localize bf16-vs-fp32 drift WITHOUT sampling-loop chaos.

Compares single forward passes (trunk, single score pass) at IDENTICAL inputs,
fp32 vs bf16, reporting relative error. This is the per-op precision floor; the
200-step e2e RMSD is this compounded + amplified by the diffusion trajectory.
"""
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_trunk_forward, boltz2_graph_score_forward

ROOT = Path(__file__).resolve().parent.parent
jax.config.update("jax_default_matmul_precision", "highest")
P = load_params(ROOT / "outputs/native_weights/boltz2_conf")
F = load_features_npz(ROOT / "outputs/real_features/1UBQ_A.npz")

def relerr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-9))

# --- trunk only (no diffusion) ---
def trunk(dt):
    P2 = jax.tree.map(lambda x: x.astype(dt) if (hasattr(x,'dtype') and jnp.issubdtype(x.dtype, jnp.floating)) else x, P)
    F2 = {k: (v.astype(dt) if jnp.issubdtype(v.dtype, jnp.floating) else v) for k, v in F.items()}
    fn = lambda p, f: boltz2_trunk_forward(p["trunk"], f, recycling_steps=0, use_scan=False)
    o = fn(P2, F2); jax.block_until_ready(o)
    return np.asarray(o["s"], np.float32), np.asarray(o["z"], np.float32)

s32, z32 = trunk(jnp.float32)
s16, z16 = trunk(jnp.bfloat16)
print(f"TRUNK  s relerr {relerr(s32,s16):.3e} | z relerr {relerr(z32,z16):.3e}")

# --- single score pass at identical r_noisy/times (no sampling loop) ---
n_atoms = int(F["atom_pad_mask"].sum()) if "atom_pad_mask" in F else s32.shape[1]
rng = np.random.default_rng(0)
A = F["coords"].shape[-2] if "coords" in F else 608
r_noisy = jnp.asarray(rng.standard_normal((1, A, 3)) * 10.0, dtype=jnp.float32)
times = jnp.asarray([[1.0]], dtype=jnp.float32)

def score(dt):
    P2 = jax.tree.map(lambda x: x.astype(dt) if (hasattr(x,'dtype') and jnp.issubdtype(x.dtype, jnp.floating)) else x, P)
    F2 = {k: (v.astype(dt) if jnp.issubdtype(v.dtype, jnp.floating) else v) for k, v in F.items()}
    fn = lambda p, f, r, t: boltz2_graph_score_forward(p, f, r.astype(dt), t.astype(dt), recycling_steps=0, use_scan=False)
    o = fn(P2, F2, r_noisy, times); jax.block_until_ready(o)
    return np.asarray(o, np.float32)

try:
    sc32 = score(jnp.float32); sc16 = score(jnp.bfloat16)
    print(f"SCORE  single-pass coord relerr {relerr(sc32,sc16):.3e} | RMSD-ish max|d| {float(np.max(np.abs(sc32-sc16))):.4f}")
except Exception as e:
    print("SCORE failed:", type(e).__name__, str(e)[:120])
