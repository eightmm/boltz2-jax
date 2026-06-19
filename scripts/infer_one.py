"""Single standalone inference of one config: time + peak VRAM + CIF output.

Loads a precomputed feature .pt (fixed MSA/template), runs boltz2_predict for
the requested precision/backend at the given steps, writes a CIF, and reports
wall-clock (compile+first / steady) and peak device memory. One config per
process so VRAM is clean and a crash can't take down the others.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from boltz_jax.bridge.native import load_params
from boltz_jax.data.write.structure import write_prediction
from boltz_jax.models.predict import boltz2_predict

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    p = argparse.ArgumentParser()
    p.add_argument("--features-pt", required=True, type=Path)
    p.add_argument("--structure-npz", required=True, type=Path)
    p.add_argument("--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--compute-dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--attention-backend", default="xla")
    p.add_argument("--triangle-backend", default="xla")
    p.add_argument("--glu-backend", default="xla")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-heads", action="store_true",
                   help="disable confidence/distogram/bfactor heads (structure only)")
    a = p.parse_args()

    f = torch.load(a.features_pt)
    feats = {k: jnp.asarray(v.numpy()) for k, v in f.items() if torch.is_tensor(v)}
    params = load_params(a.weights)
    dt = DTYPES[a.compute_dtype]
    heads = not a.no_heads
    kw = dict(
        recycling_steps=a.recycling, num_sampling_steps=a.steps, augmentation=False,
        run_confidence=heads, run_distogram=heads, run_bfactor=heads, use_scan=True,
        compute_dtype=dt, attention_backend=a.attention_backend,
        triangle_backend=a.triangle_backend, glu_backend=a.glu_backend,
    )

    dev = jax.devices()[0]
    t0 = time.perf_counter()
    out = boltz2_predict(params, feats, jax.random.PRNGKey(a.seed), **kw)
    coords = np.asarray(jax.block_until_ready(out["sample_atom_coords"]))
    t1 = time.perf_counter()

    peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1024**2
    plddt = np.asarray(out["plddt"]).reshape(-1) if "plddt" in out else np.array([0.0])
    assert np.all(np.isfinite(coords)), "non-finite coords"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_prediction(
        structure_npz=a.structure_npz, coords=coords,
        atom_pad_mask=np.asarray(f["atom_pad_mask"]).reshape(-1),
        out_path=a.out, plddts=plddt, fmt="cif",
    )
    tag = f"{a.compute_dtype}/{a.attention_backend}/{a.triangle_backend}/{a.glu_backend}"
    print(
        f"RESULT cfg={tag} steps={a.steps} "
        f"time={t1 - t0:.2f}s peak_vram={peak:.0f}MiB "
        f"plddt_mean={plddt.mean():.4f} WROTE {written}"
    )


if __name__ == "__main__":
    main()
