"""Run ONE sampler config, save coords to .npy. One sampler per process (robust)."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

ROOT = Path(__file__).resolve().parent.parent
DT = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}

p = argparse.ArgumentParser()
p.add_argument("--dtype", default="fp32")
p.add_argument("--ab", default="xla")
p.add_argument("--tb", default="xla")
p.add_argument("--gb", default="xla")
p.add_argument("--steps", type=int, default=25)
p.add_argument("--align", action="store_true")
p.add_argument("--out", required=True)
a = p.parse_args()

jax.config.update("jax_default_matmul_precision", "highest")
params = load_params(ROOT / "outputs/native_weights/boltz2_conf")
feats = load_features_npz(ROOT / "outputs/real_features/1UBQ_A.npz")
fn = jax.jit(lambda P, F, K: boltz2_sample_forward(
    P, F, K, recycling_steps=0, num_sampling_steps=a.steps, augmentation=False,
    multiplicity=1, compute_dtype=DT[a.dtype], use_scan=True,
    attention_backend=a.ab, triangle_backend=a.tb, glu_backend=a.gb,
    alignment_reverse_diff=a.align,
)["sample_atom_coords"])
out = np.asarray(jax.block_until_ready(fn(params, feats, jax.random.PRNGKey(0))), np.float32)
np.save(a.out, out)
print(f"SAVED {a.out} shape={out.shape} finite={bool(np.isfinite(out).all())}", flush=True)
