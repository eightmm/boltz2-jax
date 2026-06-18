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
- Self-contained data pipeline: raw input → features runs without `import boltz`
  (the protein featurization path is vendored under `src/boltz_jax/data/`).

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

## Data pipeline (input → features)

The vendored featurizer turns a Boltz-style YAML/FASTA into the feature dict the
model consumes, using only `boltz_jax.data` (no `import boltz`). The protein
path currently supports `msa: empty` (single + multimer chains; ligands /
templates / MSA-search are not yet ported).

```yaml
# 1ubq.yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
```

```bash
# YAML -> processed structure NPZ -> feature dict
uv run --extra torch-bridge python scripts/preprocess_standalone.py
```

`mols/` (from the download above) supplies canonical residue geometry and is
required at featurization time.

## Inference

```bash
uv run python scripts/run_standalone_inference.py \
  --weights outputs/native_weights/boltz2_conf \
  --features outputs/real_features/1UBQ_A.npz \
  --steps 25
```

### Optimization knobs

Selectable at the sampler/predict entry points; defaults reproduce the bit-exact
fp32 XLA path.

| Knob | Options | Notes |
|------|---------|-------|
| `compute_dtype` | `float32` / `float16` / `bfloat16` | fp16 lowest drift on this model |
| `matmul_precision` | `highest` / `default` | `default` = TF32 (GPU) |
| `attention_backend` | `xla` / `flash` | `flash` = tokamax token attention |
| `triangle_backend` | `xla` / `tokamax` / `pallas` | tokamax Triton wins at long N on supported GPUs |
| `glu_backend` | `xla` / `tokamax` | transition & triangle-mult GLU |
| chunking | `chunk_size`, `triangle_attention_chunk`, … | AF3-style length-dependent tiling |

Hardware note: tokamax `cudnn`/`mosaic` attention kernels are unavailable on
Blackwell (sm120); the Triton path is the fast option there.

### Known limitations

- Per-module fp16 parity (triangle attention / GLU) is verified, but the **fully
  combined fp16 + tokamax end-to-end sampler currently produces non-finite
  output** — the fp16 additive mask constants saturate to inf and need an
  fp16-safe `-inf` plus fp32 softmax islands. Until fixed, run fp16 with the XLA
  backends, or run tokamax in bf16/fp32. The default fp32/XLA path is bit-exact.
- Data pipeline covers the protein path with `msa: empty`; ligands, templates,
  and MSA-server search are not yet ported.

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
