# Boltz-2 JAX Porting Plan

## Goal

Measure whether a static-shape JAX/XLA inference engine can reduce peak VRAM for
Boltz-2 structure generation while keeping crystal-level structure quality.

## Reference Path

- Source implementation: `../boltz/src/boltz`
- Current optimized PyTorch path: `../boltz/src/boltz_fast`
- Checkpoint source: PyTorch Lightning checkpoint
- Acceptance gates:
  - tensor-level parity for isolated blocks where possible
  - final crystal CA RMSD for complete structure runs
  - no severe clashes in saved structures
  - peak VRAM lower than PyTorch baseline on large inputs

## Port Order

1. Checkpoint inspection and parameter mapping.
2. Geometry and indexing utilities.
3. Atom encoder/decoder transformer blocks.
4. Diffusion token transformer blocks.
5. Pairformer single block.
6. Static-shape full structure path.
7. Shape bucket cache and benchmark harness.

## Measurement

- Compile time and steady-state runtime must be reported separately.
- Use `jax.block_until_ready()` around timed calls.
- Record peak VRAM after warmup.
- Test real processed inputs only once full structure inference exists.

## 2026-06-16 Microbench Prototype

Implemented a first PyTorch/JAX-equivalent microbench instead of claiming a full
Pairformer/Structure port. It covers:

- token attention with pair bias
- triangle-like O(N^3) pair updates
- token/pair transition MLPs
- structure-like coordinate update loop conditioned on token and pair tensors

Smoke GPU result path:

- `outputs/microbench_cuda_smoke.json`
- `outputs/microbench_cuda_64_128.json`

Current limitation: this validates JAX/XLA mechanics and memory reporting only.
It is not yet checkpoint-compatible Boltz-2 inference.

## 2026-06-16 Checkpoint-Compatible Blocks

Implemented checkpoint-backed JAX parity for:

- Pairformer stack through `PairformerModule`.
- `SingleConditioning`.
- One `DiffusionTransformerLayer` from the token score transformer.

Benchmark artifacts:

- `outputs/checkpoint_blocks_plus_structure_cuda_117_256.json`
- `outputs/checkpoint_blocks_plus_structure_cuda_512.json`

Current limitation: this is still isolated block parity. Atom attention
encoder/decoder and complete `DiffusionModule` score inference are not ported.

## 2026-06-16 Diffusion Score Graph

Implemented JAX parity for the precomputed-conditioning score path:

- atom window indexing and `single_to_keys`
- `AtomTransformer`
- `AtomAttentionEncoder`
- `AtomAttentionDecoder`
- `DiffusionModule.forward` equivalent from single conditioning through atom
  decoder, with precomputed `diffusion_conditioning`

CUDA smoke artifact:

- `outputs/diffusion_score_cuda_tokens8_atoms64_layers24_heads16.json`

Current limitation: real feature preprocessing, MSA module, trunk orchestration,
and sampling loop are not yet in the JAX graph.

## 2026-06-16 Conditioned Score Graph

Implemented checkpoint-backed JAX parity for `DiffusionConditioning` and
connected it to the score model:

- `PairwiseConditioning`
- `AtomEncoder`
- atom encoder/decoder bias projections
- token transformer bias projections
- `conditioned_diffusion_score_forward`

CUDA smoke artifacts:

- `outputs/diffusion_score_cuda_tokens8_atoms64_layers24_heads16.json`
- `outputs/conditioned_diffusion_score_cuda_tokens8_atoms64_layers24.json`

Current limitation: this still starts from trunk tensors and processed feature
tensors. MSA, full trunk orchestration, and diffusion sampling are separate
remaining graph segments.
