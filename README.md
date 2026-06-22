# boltz2-jax

A JAX/XLA inference port of **[Boltz-2](https://github.com/jwohlwend/boltz)**,
the open-source biomolecular structure & affinity model. This project
reimplements the Boltz-2 inference graph (trunk, pairformer, MSA module,
diffusion sampler, and confidence/affinity heads) in pure JAX, loading the
original Boltz-2 PyTorch checkpoints unchanged.

It also ports the AlphaFold3-style efficiency path: fused attention/GLU kernels
via [tokamax](https://github.com/openxla/tokamax) (Triton), an optional custom
Pallas triangle-attention kernel, length-dependent chunking, and fp16/bf16
low-precision inference — selectable per backend without changing weights.

> Boltz-2 only. Boltz-1 checkpoints/features are not supported.

## Status

- Inference only (no training).
- Weight-compatible: the same Boltz-2 checkpoints load and run; no checkpoint
  key/shape or feature ABI change.
- Data pipeline (raw input → features) runs without `import boltz` (vendored
  under `src/boltz_jax/data/`). Supported inputs: proteins, ligands (SMILES and
  CCD codes), templates (CIF/PDB), and MSAs from `msa: empty`, precomputed
  a3m/csv, or the colabfold MSA server.
- Faster and lighter than the PyTorch Boltz-2 reference: at integrin9 (952 res,
  deep MSA, 200 steps) steady inference is **89 s / 11.2 GiB** vs `boltz predict`
  602 s / 21.8 GiB; fp32 matches torch to **1e-4 Å**. Wins come from XLA fusion,
  MSA subsampling, and feature/compile caches.

## Quickstart

Two commands ([uv](https://docs.astral.sh/uv/) required):

```bash
# 1. One-time setup: install deps + download Boltz-2 weights/mols + convert
bash scripts/setup.sh

# 2. Predict: raw YAML -> structure
uv run python scripts/predict.py --input job.yaml --fmt cif
```

`scripts/setup.sh` auto-detects your CUDA major version from the driver
(override with `CUDA=cuda12 bash scripts/setup.sh`), runs `uv sync` with the GPU
+ torch-bridge extras, downloads the weights + molecule DB, and converts the
checkpoints. It is idempotent. Because the env is synced once with the extras,
`uv run python …` afterwards needs **no extra flags** — the GPU JAX plugin and
the torch-side featurizer are already in the environment.

`predict.py` defaults match Boltz-2 (`--steps 200 --recycling 3`, step scale
1.5, fp32) and read the setup paths, so step 2 needs no path flags.

All artifacts stay inside the project and are git-ignored: downloads + native
weights under `.cache/`, and predictions, compile cache, and feature cache under
`outputs/`. Nothing is written outside the repo.

## Inference

`scripts/predict.py` turns a raw YAML job into a structure file. Full form (all
defaults shown; override only what you need):

```bash
uv run python scripts/predict.py \
  --input job.yaml \
  --weights outputs/native_weights/boltz2_conf \
  --mols .cache/boltz/mols \
  --out-dir outputs/predictions \
  --fmt cif
```

### Input examples

Protein + ligand (CCD code; SMILES also works via `smiles:`):

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
  - ligand:
      id: B
      ccd: ATP
```

With a structural template:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
templates:
  - cif: path/to/template.cif
```

MSAs: set `msa: empty` (single sequence), point to a precomputed `.a3m`/`.csv`,
or omit `msa` and generate one from the colabfold server with `--use-msa-server`:

```bash
uv run python scripts/predict.py \
  --input job.yaml --use-msa-server --fmt cif
# [--msa-server-url https://api.colabfold.com] [--msa-pairing-strategy greedy|complete]
```

### Optimization knobs

Defaults are the fastest verified-safe path on this GPU: fp32, XLA backends,
`use_scan` on, persistent compile cache on.

| Knob | Options | Notes |
|------|---------|-------|
| `--compute-dtype` | `float32` / `bfloat16` | bf16-mixed (trunk bf16, diffusion fp32 island) is ~2.12x; fp16 is range-unstable in the sampler |
| `--compile-cache` | dir (default on) | persistent XLA cache; reuses compiles across runs |
| `--feature-cache` | dir (default on) | memoize features by input digest; cache hit is bit-identical and skips featurization |
| `--bucket` | flag (default off) | pad token/atom dims to a ladder so different lengths share compile-cache entries (serving); shifts coords ~1e-4 Å |
| `matmul_precision` | `highest` / `default` | `default` = TF32 (GPU) |
| `attention_backend` | `xla` / `tokamax` | fused tokamax attention |
| `triangle_backend` | `xla` / `tokamax` / `pallas` | fused triangle kernels |
| `glu_backend` | `xla` / `tokamax` | transition & triangle-mult GLU |

The fused-kernel backends (`tokamax`/`pallas`) are opt-in. On Blackwell
**sm120** (triton-only) they are a net regression end-to-end — the 200-step
diffusion attention is an fp32 island, so the kernels fall back to a slow fp32
path. `xla` is the default. On cudnn-capable GPUs / TPU they may win; re-measure
per target.

## Python API

After `setup.sh`, call inference from Python (heavy deps load lazily, so
`import boltz_jax` stays cheap):

```python
import boltz_jax

out = boltz_jax.predict(
    seq=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"],
    ligand_ccd=["ATP"],                 # also dna=, rna=, ligand_smiles=, or input="job.yaml"
    weights="outputs/native_weights/boltz2_conf",
    mols=".cache/boltz/mols",
    write_fmt="cif",                    # omit to skip writing a structure file
)
out["coords"]   # (n_atom, 3) np.ndarray;  out["plddt"], out["raw"], out["out_path"]
```

For composing with other JAX models, use the low-level pure-JAX function and the
weight loader directly:

```python
from boltz_jax import boltz2_predict, load_params, build_job_yaml  # all lazy
```

`boltz2_predict(params, feats, key, ...)` is a jit-friendly function over a
parameter + feature pytree; `boltz_jax.featurize(...)` produces `feats`.

## Tests

```bash
uv run pytest -q          # import + end-to-end sample smoke (CPU-forced)
uv run ruff check .
```

## Attribution & license

Released under the [MIT License](LICENSE).

Derivative work of **Boltz** (© 2024 Jeremy Wohlwend, Gabriele Corso, Saro
Passaro et al.), used under the MIT License. The preprocessing/featurization
code under `src/boltz_jax/data/` is adapted from the Boltz repository; the
original copyright and license are retained in [`NOTICE`](NOTICE). Model weights
are distributed by the Boltz community under MIT.

If you use this work, please cite the Boltz-2 technical report
(https://doi.org/10.1101/2025.06.14.659707).
