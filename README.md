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

## Install

Uses [uv](https://docs.astral.sh/uv/). Pick the CUDA extra matching your driver
(this workstation is CUDA 13):

```bash
# GPU (CUDA 13) + torch bridge (checkpoint conversion + featurization) + dev
uv sync --extra cuda13 --extra torch-bridge --extra dev
# CUDA 12 alternative:
uv sync --extra cuda12 --extra torch-bridge --extra dev
```

> Note: `uv sync` must always include the `cuda13` (or `cuda12`) extra, or the
> GPU JAX plugin is pruned and JAX silently falls back to CPU. Verify with
> `uv run python -c "import jax; print(jax.default_backend())"` → `gpu`.

## Model weights & data

Weights and the molecule database come from the **same sources as upstream
Boltz** (MIT licensed, Boltz community on HuggingFace):

| Artifact | URL |
|----------|-----|
| Structure model | `https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt` |
| Affinity model | `https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt` |
| Molecule DB (CCD `mols`) | `https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar` |

```bash
mkdir -p .cache/boltz && cd .cache/boltz
curl -L -o boltz2_conf.ckpt https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt
curl -L -o boltz2_aff.ckpt  https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt
curl -L -o mols.tar         https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar
tar -xf mols.tar            # -> .cache/boltz/mols/   (canonical residue molecules)
cd -
```

(These are the same files upstream Boltz's `download_boltz2` fetches; if you
already have a Boltz cache, point at that directory instead.)

### Convert checkpoints to native JAX weights

The JAX runtime loads `safetensors` converted once from the PyTorch `.ckpt`:

```bash
uv run --extra torch-bridge python scripts/export_native_weights.py \
  --conf-ckpt .cache/boltz/boltz2_conf.ckpt \
  --aff-ckpt  .cache/boltz/boltz2_aff.ckpt \
  --out-dir   outputs/native_weights \
  --dtype fp32          # or bf16 / fp16 for half-precision storage
# -> outputs/native_weights/boltz2_conf.safetensors, boltz2_aff.safetensors
```

## Inference

`scripts/predict.py` is the entry point: raw YAML → structure file. Defaults
match Boltz-2 (`--steps 200 --recycling 3`, step scale 1.5, fp32). `mols/` (from
the download above) is required at featurization time.

```bash
uv run --extra torch-bridge python scripts/predict.py \
  --input job.yaml \
  --weights outputs/native_weights/boltz2_conf \
  --mols .cache/boltz/mols \
  --out-dir outputs/predictions \
  --fmt cif            # or pdb
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
uv run --extra torch-bridge python scripts/predict.py \
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
per target. See [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md).

## Tests

```bash
uv run pytest -q          # module + checkpoint parity (CPU-forced)
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
