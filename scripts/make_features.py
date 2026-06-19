"""Generate Boltz features for an arbitrary protein sequence (msa: empty).

YAML -> preprocess (boltz_jax.data, no `import boltz`) -> featurize -> save a
torch .pt feature dict (batched, dim 0) usable by both the JAX sampler and the
PyTorch reference benchmark. Default: a deterministic ~length random valid AA
sequence (speed/memory benchmarking does not need a biological sequence).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from boltz_jax.data.module.inferencev2 import PredictionDataset
from boltz_jax.data.preprocess import check_inputs, process_inputs

ROOT = Path(__file__).resolve().parent.parent
MOL_DIR = ROOT.parent / "boltz/.cache/boltz/mols"
AA = "ACDEFGHIKLMNPQRSTVWY"

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=1025)
p.add_argument("--id", default="BENCH1025")
p.add_argument("--seq", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
a = p.parse_args()

seq = a.seq or "".join(np.random.default_rng(a.seed).choice(list(AA), a.length))
out = Path(a.out) if a.out else ROOT / "outputs/real_features" / f"{a.id}.pt"
prep = ROOT / "outputs/prep" / a.id
prep.mkdir(parents=True, exist_ok=True)
yaml = prep / f"{a.id}.yaml"
yaml.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n      msa: empty\n")  # noqa: E501

assert MOL_DIR.exists(), f"missing mols: {MOL_DIR}"
data = check_inputs(yaml)
manifest = process_inputs(data=data, out_dir=prep, ccd_path=prep / "unused.pkl",
                          mol_dir=MOL_DIR, use_msa_server=False, boltz2=True)
proc = prep / "processed"
ds = PredictionDataset(manifest=manifest, target_dir=proc / "structures",
                       msa_dir=proc / "msa", mol_dir=MOL_DIR,
                       constraints_dir=proc / "constraints",
                       template_dir=proc / "templates", extra_mols_dir=proc / "mols")
feats = ds[0]
out_d = {}
for k, v in feats.items():
    if k.startswith("_") or k == "record" or not torch.is_tensor(v):
        continue
    out_d[k] = v.unsqueeze(0).detach().cpu()
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(out_d, out)
n_tok = int(out_d["token_pad_mask"].sum())
n_atom = int(out_d["atom_pad_mask"].sum())
print(f"SAVED {out} | tokens={n_tok} atoms={n_atom} len(seq)={len(seq)}")
