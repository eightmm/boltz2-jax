# boltz_jax

Experimental JAX inference engine for Boltz-2.

This project is separate from the canonical PyTorch/TensorRT path in
`../boltz`. The first goal is to test whether JAX/XLA can reduce peak VRAM for
static-shape Boltz-2 structure inference. Latency is secondary until numerical
parity and memory behavior are measured.

## Scope

- Port inference only.
- Start with static/precomputed inputs.
- Keep PyTorch Boltz checkpoints as the source of truth.
- Compare against the existing PyTorch baseline and crystal structures.
- Exclude MSA server time and file writer time from benchmark regions.

## Non-goals

- No training path.
- No checkpoint format change.
- No replacement of the existing `boltz_fast` path until measured.
- No confidence module until the structure path is validated.

## Initial Milestones

1. Inspect PyTorch checkpoint key/shape layout.
2. Define JAX pytree parameter layout.
3. Port small pure tensor/geometry utilities.
4. Port one Pairformer block and compare tensor outputs.
5. Port one diffusion score block and compare tensor outputs.
6. Build static-shape VRAM and latency benchmark.
7. Expand to full structure inference only if the block probes pass.

## Commands

```bash
cd /home/jaemin/non-project/optimizing/boltz_jax
uv sync --extra dev
```

Inspect a Boltz checkpoint:

```bash
uv run --extra torch-bridge boltz-jax-inspect-checkpoint \
  --checkpoint ../boltz/.cache/boltz/boltz2_conf.ckpt \
  --limit 40
```

For CUDA JAX, use the optional CUDA extra in a separate environment after
checking local driver/toolkit compatibility. CUDA 13 is the first candidate on
this workstation because the existing PyTorch environment is cu13-based:

```bash
uv sync --extra cuda13 --extra dev --extra torch-bridge
```

CUDA 12 is also available as an explicit extra for fallback testing:

```bash
uv sync --extra cuda12 --extra dev --extra torch-bridge
```

## Microbench

Run the PyTorch/JAX-equivalent micro Pairformer/Structure benchmark:

```bash
uv run python scripts/benchmark_micro_modules.py \
  --residues 64 128 \
  --token-s 128 \
  --token-z 64 \
  --heads 8 \
  --blocks 2 \
  --steps 10 \
  --warmup 2 \
  --iters 5 \
  --output outputs/microbench_cuda_64_128.json
```

This is not a full Boltz port. It is a parity-controlled speed/VRAM probe for
Pairformer-like O(N^3) pair updates and a structure-like iterative coordinate
loop.

## Checkpoint Blocks

Run checkpoint-compatible PyTorch/JAX block benchmarks:

```bash
uv run python scripts/benchmark_checkpoint_blocks.py \
  --residues 117 256 \
  --warmup 2 \
  --iters 5 \
  --output outputs/checkpoint_blocks_plus_structure_cuda_117_256.json
```

Covered checkpoint-backed blocks:

- Pairformer attention, triangle multiplication, and transitions.
- `structure_module.score_model.single_conditioner`.
- One `structure_module.score_model.token_transformer` layer.

For full score-model coverage, use the diffusion score graph benchmark below.

## Diffusion Score Graph

Run a checkpoint-compatible score-model benchmark with precomputed diffusion
conditioning:

```bash
uv run python scripts/benchmark_diffusion_score.py \
  --tokens 8 \
  --atoms 64 \
  --token-layers 24 \
  --warmup 1 \
  --iters 3 \
  --output outputs/diffusion_score_cuda_tokens8_atoms64_layers24.json
```

This compiles the score path from `SingleConditioning` through atom
encoder/decoder and the full token transformer stack. It still assumes
diffusion conditioning tensors are already prepared.
