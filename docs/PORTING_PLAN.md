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
