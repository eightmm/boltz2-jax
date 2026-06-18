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

## 2026-06-16 Full Inference Path + Aux Heads

Completed and parity-tested against the PyTorch reference:

- `weighted_rigid_align` (Algorithm 28) ported to `trunk.py::_weighted_rigid_align`;
  added to the sampler as `alignment_reverse_diff` (default ON, matches eval).
  Test: `tests/test_weighted_rigid_align_parity.py`.
- Per-step centered random augmentation (`_compute_random_augmentation`) added to
  `boltz2_sample_forward` (default ON). Stochastic, so validated structurally not
  bit-for-bit.
- End-to-end structure sampling now runs on real processed features
  (`outputs/real_features/1UBQ_A.pt`) via `boltz2_sample_forward`, which is the
  single trunk -> conditioning -> sampling entry point.
  Test: `tests/test_end_to_end_sample_smoke.py` (marked `slow`).
- `DistogramModule` -> `models/distogram.py` + `map_distogram_state_dict`.
  Test: `tests/test_distogram_checkpoint_parity.py`.
- `BFactorModule` -> `models/bfactor.py` + `map_bfactor_state_dict`.
  Test: `tests/test_bfactor_checkpoint_parity.py`.

Test-only dev dep added: `einx` (needed to import the PyTorch
`weighted_rigid_align` reference module).

Full suite after this work: 38 passed, 1 skipped.

## 2026-06-16 Steering + Confidence + Affinity (completed)

All previously-remaining modules are now ported and parity-tested. Full suite:
63 passed, 1 skipped.

- Steering potentials -> `models/potentials.py`: JAX port of the schedule
  helpers and all 9 potentials (PoseBusters, Connections, VDWOverlap,
  SymmetricChainCOM, StereoBond, ChiralAtom, PlanarBond, TemplateReference,
  Contact) + `get_potentials`. Wired into `boltz2_sample_forward` behind
  `steering_args` (default None = unchanged no-steering path): FK resampling +
  physical/contact guidance update loop mirroring `diffusionv2.sample`.
  Test: `tests/test_potentials_parity.py` (23 cases, bit-parity 1e-4; only
  `TemplateReferencePotential` relaxed to 1e-3 due to float32 SVD; Contact union
  softmax NaN-on-underflow edge is intentionally guarded in JAX).
- `ConfidenceModule` -> `models/confidence.py` + `bridge/confidence_mapping.py`.
  Real config from checkpoint: add_s_to_z_prod / add_s_input_to_s /
  add_z_input_to_z / bond_type_feature all True, token_level_confidence,
  use_separate_heads, pairformer 8 blocks / 16 heads. Outputs pLDDT/PAE/PDE
  logits + aggregated metrics + pTM/ipTM/per-chain-pair iptm. Reuses
  pairformer / rel-pos / contact-conditioning forwards.
  Test: `tests/test_confidence_checkpoint_parity.py` (all outputs, 2e-3).
- `AffinityModule` -> `models/affinity.py` + `models/pairformer_noseq.py`
  (new pair-only stack) + `bridge/affinity_mapping.py`. Uses separate
  `boltz2_aff.ckpt` (`affinity_module1`, pairformer 8 blocks, transformer
  12 blocks / 8 heads). Outputs affinity_pred_value + affinity_logits_binary.
  Test: `tests/test_affinity_checkpoint_parity.py` (2e-3).

Not ported (disabled in these checkpoints / out of inference path): confidence
`run_sequentially` + atom-level head branch; `affinity_module2`; confidence
non-polymer frame reassignment is mirrored but only lightly exercised.

## 2026-06-16 Standalone runtime, native weights, full-graph + precision bench

- Sampler validated vs PyTorch with identical injected noise (augmentation off,
  alignment on): aligned RMSD 0.031 A @ 5 steps, 2e-5 A @ 25 steps
  (`scripts/compare_sampling_rmsd.py`). Difference is pure fp32 framework drift.
- Full-graph 200-step sampler: added `use_scan=True` (whole trunk+conditioning+
  200-step loop compiles as one XLA graph via `lax.scan`). NOTE: the first
  benchmark of this was NOT apples-to-apples (torch ran recycling=0 vs JAX
  recycling=3); see the corrected fair benchmark below.
  `scripts/benchmark_sampling_fullgraph.py`.
- Standalone runtime (no torch / no boltz at inference): `bridge/native.py`
  (safetensors `save_params`/`load_params` + `load_features_npz`),
  `scripts/export_native_weights.py` (one-time torch converter),
  `scripts/run_standalone_inference.py` (asserts torch/boltz not imported).
  `torch` moved to optional `torch-bridge` extra in pyproject. Native vs
  torch-checkpoint params: bit-identical (diff 0.0).
- Precision sweep (`scripts/benchmark_precision.py --matmul-precision ...`),
  200 steps, 1UBQ_A, drift vs fp32/highest:
  - fp32/highest: 966 ms, 1.00x, 0 A
  - fp32/default (TF32): 707 ms, 1.37x, 0.011 A
  - bf16/default: 624 ms, 1.55x, 0.022 A
  - fp16/default: 670 ms, 1.44x, 0.012 A
  All numerically stable (no NaN). Speed comes from matmul precision, not dtype
  alone; `_weighted_rigid_align` stays an fp32 island. Peak VRAM rises ~1 GiB for
  bf16/fp16 (cast buffers). Recommendation: bf16/default for speed,
  fp32/default for lowest drift+VRAM. Single-structure caveat.

## 2026-06-17 Review fixes: fair benchmark, predict wrapper, native confidence

Addressed an external review of benchmark credibility and end-to-end completeness.

- bf16 native weight storage (`save_params(dtype=)` / `load_params(dtype=)`,
  `export_native_weights.py --dtype`): `boltz2_conf_bf16.safetensors` ~919 MiB
  (half of fp32). Loading bf16 directly removes the fp32->bf16 cast dual buffer:
  50-step full-graph peak VRAM 2768 -> 1478 MiB (-1290 MiB), drift 0.0067 A.
  Fixed a real bf16 lax.scan carry-dtype bug in `atom.py` (no-op for fp32).
- Layer stacks (`pairformer`, `msa`, token/atom transformers) run via `lax.scan`
  (`models/_scan_utils.py::stack_layer_params`). Compile+first cut ~2.1-2.8x
  (e.g. 471 s -> 169 s at 200 steps). Steady latency rises ~60% vs unrolled —
  scan is a COMPILE-time win that COSTS steady latency. Tests stay 63 passed.
  Caveat: scan is currently always-on inside the module forwards; for a
  high-throughput server that amortizes compile, an unrolled path would give
  lower steady latency (consider making it opt-in).
- Memory-efficient attention (`jax.nn.dot_product_attention`) was implemented,
  verified bit-identical, but REVERTED: XLA already fuses the explicit softmax,
  cudnn flash rejects fp32, so it gave no speed/VRAM win (slightly worse).
- FAIR full-graph benchmark (equal recycling=3, augmentation=False BOTH sides,
  alignment on, fp32, 200 steps, RTX PRO 6000):
  - 1UBQ_A (76 tok / 608 at): JAX 1563 ms vs PyTorch 2825 ms = 1.81x;
    peak 2952 vs 1933 MiB; JAX compile+first 126 s.
  - 1US0_A (314 tok / 2528 at, REAL): JAX 5248 ms vs PyTorch 8190 ms = 1.56x;
    peak 3984 vs 3361 MiB; compile+first 144 s.
  `outputs/sampling_fullgraph_benchmark.json`, `outputs/fair_benchmark_summary.json`.
  The earlier ~7x figure was the non-fair (torch recycling=0) comparison and is
  retracted.
- Augmentation: the compiled scan path is augmentation=False (deterministic
  serving mode). Per-step random rigid rotation is removed by the per-step
  `weighted_rigid_align`, so it does not change final structure quality; eager
  augmentation=True was measured at ~no latency penalty. Benchmarks compare
  aug-off vs aug-off.
- Compile vs eager / why compile is long: only the benchmark jits the whole
  pipeline; `run_standalone_inference.py` / `benchmark_metric_drift.py` run eager
  (no big compile). Long compile = monolithic single-graph jit at a large
  concrete shape (O(N^3) triangle ops) + GEMM autotuning + no persistent cache.
  PyTorch reference runs eager (zero compile), which is why it "starts" instantly.
  Serving mitigation: persistent `jax_compilation_cache_dir` + shape bucketing
  (amortize compile once per bucket). Validated padding-invariance enables this.
- Single full predict wrapper: `models/predict.py::boltz2_predict` runs
  trunk -> sampling -> distogram -> bfactor -> confidence (-> affinity if params)
  returning one dict (sample_atom_coords, pdistogram, pbfactor, plddt/pae/pde/
  ptm/iptm/complex_*). `tests/test_predict_wrapper.py` (slow) asserts per-head
  outputs equal direct calls. Confidence is now included in the native weight
  bundle (`boltz2_conf.safetensors` top keys: trunk, conditioned_diffusion,
  distogram, bfactor, confidence). `run_standalone_inference.py` uses the wrapper.
- Metric drift on real 1UBQ_A (identical inputs, 15/15 feats real, no synthesis):
  all heads agree to ~1e-6 (pTM 0.93430 both; complex_plddt 0.93336 both;
  distogram argmax 100%). `outputs/metric_drift.json`.
- Padding-invariance verified: 1UBQ real-atom outputs unchanged when padded to
  128/256 tokens (worst real-atom RMSD 1.66e-5 A, no masking leak) -> shape
  bucketing is safe. `outputs/padding_invariance.json`.
- Geometry/clash sanity (JAX 1UBQ, 200 steps): 0 clashes <1.2 A, min dist
  1.22 A, mean CA-CA 3.805 A (100% in 3.4-4.2 A). Crystal RMSD vs PDB not wired
  (no ground-truth correspondence) - still open. `outputs/geometry_check.json`.

Gates after this work: `uv run pytest -q` = 64 passed, 1 skipped;
`uv run ruff check .` = clean.

Still open / not done: crystal-RMSD-vs-PDB validation; augmentation-on inside the
compiled scan path; affinity end-to-end on a real ligand input; making scan
opt-in vs unrolled for steady-latency-bound serving; multi-chain padding-invariance.

## 2026-06-17 Weight-compatible speed/memory optimizations (merged)

Constraint: NO checkpoint key/shape, native-weight-format, or feature-dict ABI
changes — same weights load & run. All items below satisfy this; full suite
71 passed / 1 skipped, ruff clean.

Memory (peak VRAM): OuterProductMean / triangle-mult / triangle-attention chunking
(i-axis & row-axis, output-buffer update instead of list+concat); one-hot ->
internal gather/scatter for atom_to_token, token_to_rep_atom, z_to_p; pairwise
[N,N,3] diff removed; relative-position one-hot -> embedding lookup (same kernel).
Result: L768 jitted peak_bytes_in_use 18.5 GiB -> 7.8 GiB (~= torch 7.4); L384
JAX below torch. 1US0 peak 3990 -> ~3571 MiB. Chunking is bit-exact (reduction
axis never split); chunk_size threaded from sampler (default 128).

Speed: `matmul_precision` flag threaded into the sampler/trunk (default "highest"
= bit-identical; opt-in "default"/TF32 relaxes the triangle-attention
Precision.HIGHEST pin + global matmul precision). q/g, k/v, fc1/fc2 projection
fusion; distogram reorder ((z@W)+(z@W).T); predict single trunk pass; use_scan
split into trunk_use_scan/score_use_scan.

1US0_A (314 tok / 2528 atoms, 200 steps, recycling 3, aug off, jitted scan):
- baseline highest: 5114 ms, 3571 MiB (bit-identical)
- TF32 default: 3805 ms (1.34x), 3574 MiB, aligned-RMSD drift ~0.02 A
- vs torch eager 8132 ms -> TF32 JAX ~2.1x

Not done / open: crystal RMSD vs PDB confirmed at 1.35 A (1UBQ, 100 steps);
bf16 cudnn flash unavailable on this Blackwell box; projection fusion win under
jit is marginal (XLA already fuses) but harmless; chunk_size=256 gave no speed
win at 314 tok (keep 128 default).

## 2026-06-18 AF3/JAX memory follow-up

References checked:

- AF3 `modules.py`: `mapping.inference_subbatch` query chunking for pair
  attention, row-sharded pair transition, Tokamax GLU/attention, and chunked
  `OuterProductMean`.
- JAX docs:
  - Pallas `BlockSpec`/`pallas_call` tile model:
    https://docs.jax.dev/en/latest/pallas/quickstart.html
  - buffer donation at `jit` boundaries:
    https://docs.jax.dev/en/latest/buffer_donation.html
  - host/offload mechanisms:
    https://docs.jax.dev/en/latest/notebooks/host-offloading.html
  - `shard_map` vs automatic `jit` partitioning:
    https://docs.jax.dev/en/latest/notebooks/shard_map.html

Applied/confirmed:

- AF3-style query chunking is now wired through the score graph as
  `token_attention_chunk`, not only through the sampler. This targets the real
  2048-token blocker: `[B, heads, N, N]` token attention scores in
  `diffusion_transformer._attention_pair_bias_no_proj_z_forward`.
- Diffusion token pair bias can now be computed lazily per token-transformer
  layer (`lazy_token_trans_bias=True` on sampler/graph-score entry points).
  This preserves the existing `token_trans_proj_z` weight list and avoids the
  full `[B, N, N, L * heads]` concat buffer.
- `benchmark_flash_sampling_scale.py` and `benchmark_triangle_backend_e2e.py`
  now record/pass `token_attention_chunk` and dtype, so large-length runs do
  not accidentally benchmark the pre-fix OOM path.
- Current GPU evidence:
  - 1000 tokens / 8000 atoms / 20 steps / recycling 3 / fp32 / XLA /
    `token_attention_chunk=256`: success, steady `54.1 s`,
    JAX allocator peak `10.7 GiB`, process peak `33.4 GiB`.
    `outputs/stage4_lazybias_tokenchunk_1000_float32.json`.
  - Superseded: 2048 tokens / 16384 atoms / same settings initially OOMed
    trying to allocate `35.31 GiB`; bf16 also OOMed at `35.07 GiB`.
    `outputs/stage4_lazybias_tokenchunk_2048_float32.json`,
    `outputs/stage4_lazybias_tokenchunk_2048_bfloat16.json`.
- Follow-up isolated changes:
  - Lazy token bias now pre-normalizes the pair input once and applies only the
    layer-local affine+linear inside each token-transformer layer. Weight ABI is
    unchanged and lazy/full output is bit-exact on checkpoint tests.
  - 1000-token fp32 XLA after pre-normalization: steady `52.6 s` vs `54.1 s`
    before; allocator peak unchanged at `10.7 GiB`.
    `outputs/stage5_lazybias_normed_1000_float32.json`.
  - 1536-token fp32 XLA after pre-normalization: steady `171.3 s`, allocator
    peak `24.1 GiB`, essentially unchanged vs baseline.
    `outputs/stage5_lazybias_normed_1536_float32.json`.
  - `benchmark_flash_sampling_scale.py` can now override `trunk_use_scan` /
    `score_use_scan` for diagnosis. `score_use_scan=false` helped at 500 tokens
    (`6.1 -> 5.3 GiB`) but hurt at 1000 tokens (`10.7 -> 14.7 GiB`) with no
    speed win, so the large-N path should keep score scan enabled.
    `outputs/stage5_score_unroll_500_float32.json`,
    `outputs/stage5_score_unroll_1000_float32.json`.
  - MSA `PairformerNoSeqLayer.transition_z` was missing the row-chunking already
    used by the main Pairformer `transition_z`. Adding `row_chunk_size=chunk_size`
    is weight-compatible and reduction-exact. The same default chunk path was
    added to the standalone `pairformer_noseq.py` used by affinity/confidence
    pair-only stacks.
  - Stage6 GPU evidence after MSA transition chunk:
    - 1000 tokens: allocator peak unchanged at `10.7 GiB`, process peak
      `33.4 -> 25.2 GiB`, steady `54.5 s`.
      `outputs/stage6_msa_transition_chunk_1000_float32.json`.
    - 1536 tokens: allocator peak `24.1 -> 22.7 GiB`, process peak
      `70.4 -> 54.0 GiB`, steady `171.9 s`.
      `outputs/stage6_msa_transition_chunk_1536_float32.json`.
    - 2048 tokens: now fits fp32 full graph, steady `380.0 s`, allocator peak
      `36.0 GiB`, process peak `91.9 GiB`, finite output. This is still close
      to a 96 GiB card's limit; further peak reduction is still needed for
      headroom.
      `outputs/stage6_msa_transition_chunk_2048_float32.json`.
  - AF3 long-sequence gap identified: AF3 uses residue-dependent pair attention
    chunking (`pair_attention_chunk_size=((1536, 128), (None, 32))`) and
    separate pair-transition sharding. Our previous single `chunk_size=128`
    left triangle attention too coarse at 2048 tokens. Added
    `triangle_attention_chunk` as a separate weight-compatible control.
  - Stage7 GPU evidence with `triangle_attention_chunk=64`:
    - 1000 tokens: allocator peak `10.7 -> 9.2 GiB`, process peak
      `25.2 -> 17.0 GiB`, steady `54.5 -> 52.8 s`.
      `outputs/stage7_triangle_attention_chunk64_1000_float32.json`.
    - 2048 tokens: allocator peak `36.0 -> 27.8 GiB`, process peak
      `91.9 -> 66.3 GiB`, steady `380.0 -> 368.6 s`, finite output.
      `outputs/stage7_triangle_attention_chunk64_2048_float32.json`.
  - Stage8 changed the default policy to AF3-style auto chunking:
    `triangle_attention_chunk=None` keeps the existing `chunk_size` up to 1536
    tokens and switches triangle attention query chunks to 32 above that.
    Explicit caller values still override the policy. This is weight-compatible:
    it changes only evaluation tiling, not parameters or math.
    - 2048 tokens with manual `triangle_attention_chunk=32`: allocator peak
      `25.6 GiB`, process peak `66.3 GiB`, steady `366.3 s`.
      `outputs/stage8_triangle_attention_chunk32_2048_float32.json`.
    - 2048 tokens with default auto policy: allocator peak `25.6 GiB`,
      process peak `66.3 GiB`, steady `357.7 s`, finite output.
      `outputs/stage8_auto_triangle_attention_chunk_2048_float32.json`.
    This brings the memory policy closer to AF3's subbatching model while
    preserving checkpoint/native-weight ABI and exact reductions.
  - Stage9 adds an inner query-row chunk inside XLA triangle attention. The
    Stage8 outer chunk reduces the independent triangle batch axis, but each
    block still materializes an `N x N` attention score. Above 2048 tokens the
    default now also chunks query rows at 512, reducing the score workspace from
    `[outer_chunk, heads, N, N]` to `[outer_chunk, heads, q_chunk, N]` while
    keeping the full key axis for each softmax row. This is still
    weight-compatible and mathematically equivalent.
    - CPU exact-parity test added for inner query chunk.
    - 64-token sampler smoke with explicit outer/q chunks and 20 sampling
      steps succeeded with finite output.
      `outputs/stage9_triangle_qchunk_smoke_20step.json`.
    - 3072-token fp32 full-graph probe was interrupted during
      `compile_and_load`, so it is not a valid pass/fail benchmark. During that
      attempt the process reached about `94.7 GiB` GPU memory before returning,
      which means fp32 XLA at 3072 still has almost no headroom on a 96 GiB GPU.
      `outputs/stage9_triangle_qchunk_3072_float32_probe.json`.
  - Stage10 starts aligning the low-precision path with AF3's broad bf16 policy:
    `triangle_multiplication_forward` still performs the triangle contraction in
    fp32, but now casts the contraction output back to the input activation dtype
    before the output norm/projection. This prevents bf16 pair activations from
    being promoted back to fp32 by the residual. The fp32 default path is
    unchanged. CPU tests cover checkpoint parity and bf16 output dtype. A
    64-token bf16 sampler smoke with explicit triangle chunks and 20 sampling
    steps produced finite output.
    `outputs/stage10_triangle_bf16_dtype_smoke.json`.
    - 2048-token bf16 full-graph probe now succeeds where the earlier stage4
      bf16 run OOMed. It is finite, compile+first `527.0 s`, allocator peak
      `27.4 GiB`, process peak `66.3 GiB`.
      `outputs/stage10_triangle_bf16_dtype_2048_probe.json`.
    - This did not reduce process peak versus stage8 fp32 (`66.3 GiB`) and
      allocator peak is slightly higher than fp32 (`27.4 GiB` vs `25.6 GiB`),
      so the current long-N blocker is not just triangle-mult residual dtype.
      Remaining suspects are fp32 attention islands, transition hidden buffers,
      and XLA executable/workspace memory.
  - Stage11 adds an explicit `transition_hidden_chunk` option for Pairformer and
    MSA pair-only `transition_z` blocks. It reuses the existing
    `transition_forward(chunk_size=...)` hidden-dimension chunking and keeps the
    default `None` path unchanged. This is the Boltz-JAX counterpart to AF3's
    sharded pair-transition policy, but opt-in because hidden chunking changes
    the fp32 accumulation order slightly.
    - CPU tests cover combined row+hidden transition chunking plus checkpoint
      paths.
    - 64-token bf16 sampler smoke with `transition_hidden_chunk=16` produced
      finite output.
      `outputs/stage11_transition_hidden_chunk_smoke.json`.
    - 2048-token bf16 with `transition_hidden_chunk=128` was also finite, but
      did not materially reduce peak: allocator `27.391 -> 27.376 GiB`,
      process `66.282 -> 66.290 GiB`, compile+first `538.1 s`.
      `outputs/stage11_transition_hidden_chunk128_2048_bfloat16.json`.
    - Conclusion: transition hidden buffers are not the current 2048 peak
      driver. The next suspects are fp32 attention islands and XLA
      executable/workspace memory; long-length Pallas triangle attention should
      be measured before spending more effort on transition chunking.
  - Stage12 exposes `triangle_backend` in the full synthetic sampler benchmark
    so XLA and Pallas triangle attention can be compared under identical
    20-step sampling settings.
    - 64-token bf16 Pallas smoke succeeded with finite output:
      `outputs/stage12_pallas_triangle_smoke.json`.
    - 512-token bf16 full sampler, 24 token layers, recycling 3:
      XLA triangle `8.925 s` steady, Pallas triangle `7.631 s` steady.
      Allocator/process peaks were effectively unchanged at `4.9 GiB` /
      `8.85 GiB`.
      `outputs/stage12_triangle_backend_xla_512_bfloat16.json`,
      `outputs/stage12_triangle_backend_pallas_512_bfloat16.json`.
    - 1024-token bf16 full sampler, same settings:
      XLA triangle `55.86 s` steady, Pallas triangle `42.84 s` steady.
      Allocator/process peaks were effectively unchanged at `12.1 GiB` /
      `25.2 GiB`.
      `outputs/stage12_triangle_backend_xla_1024_bfloat16.json`,
      `outputs/stage12_triangle_backend_pallas_1024_bfloat16.json`.
    - Conclusion: Pallas triangle attention is useful for speed at these
      lengths, but current long-length peak is not explained solely by dense
      triangle-attention score materialization. Remaining memory suspects are
      pair activation dtype islands, other pair-bias/attention buffers, and XLA
      executable/workspace memory.
  - Stage13 fixes several remaining bf16 activation dtype leaks while keeping
    the fp32 default path and weight ABI unchanged:
    - trunk bond/contact conditioning now follows parameter dtype instead of
      forcing fp32 pair activations;
    - MSA extra features, pair-weighted averaging masks, and triangle
      multiplication masks now follow activation dtype;
    - triangle attention no longer promotes bf16 q activations through
      `q / sqrt(float32)`.
    - Regression tests cover bf16 dtype preservation for contact conditioning,
      pair-weighted averaging, triangle multiplication, and triangle attention.
    - 512-token bf16 XLA full sampler improved versus Stage12 XLA:
      steady `8.925 -> 6.163 s`, allocator peak `4.893 -> 4.153 GiB`,
      process peak unchanged at `~8.85 GiB`.
      `outputs/stage13_dtype_leak_fix_xla_512_bfloat16.json`.
    - 1024-token bf16 XLA full sampler improved:
      steady `55.86 -> 41.95 s`, allocator peak `12.126 -> 10.330 GiB`,
      process peak unchanged at `25.236 GiB`.
      `outputs/stage13_dtype_leak_fix_xla_1024_bfloat16.json`.
    - 1024-token bf16 with Pallas triangle attention improved further:
      steady `31.86 s`, allocator peak `9.882 GiB`, process peak `17.042 GiB`.
      This is the first run where the AF3-like low-precision policy and fused
      triangle attention reduce both working set and process-level peak at
      1024 tokens.
      `outputs/stage13_dtype_leak_fix_pallas_1024_bfloat16.json`.
    - 2048-token bf16 XLA full sampler now has usable headroom:
      steady `303.54 s`, allocator peak `18.120 GiB`, process peak `33.510 GiB`.
      This is a large drop from the earlier bf16 runs around allocator
      `27.4 GiB` and process `66.3 GiB`.
      `outputs/stage13_dtype_leak_fix_xla_2048_bfloat16.json`.
    - 2048-token bf16 with Pallas triangle attention keeps the same peak class
      but improves steady time to `223.62 s` (`1.36x` versus Stage13 XLA):
      allocator peak `18.121 GiB`, process peak `33.504 GiB`.
      At 2048, dtype preservation is the memory win; Pallas is primarily a
      speed win.
      `outputs/stage13_dtype_leak_fix_pallas_2048_bfloat16.json`.
  - Stage14 starts 3000-token scaling:
    - 3072-token bf16 Pallas with the 2048 chunk policy (`chunk_size=128`,
      `token_attention_chunk=128`, auto triangle chunks) still OOMed on a
      single `33.83 GiB` allocation.
      `outputs/stage14_pallas_3072_bfloat16_fit_probe.json`.
    - Tight AF3-style chunks fit: `chunk_size=64`,
      `triangle_attention_chunk=16`, `triangle_attention_q_chunk=256`,
      `token_attention_chunk=64`. Compile+first was very slow (`1855.7 s`),
      but the run completed with allocator peak `31.233 GiB` and process peak
      `66.526 GiB`.
      `outputs/stage14_pallas_3072_bfloat16_tight_chunks_probe.json`.
    - Conclusion: 3000-token synthetic full sampling now fits on the 96 GiB
      GPU, but only with smaller chunks. The next bottleneck is compile/runtime,
      not just peak memory. The long-length policy should become
      length-dependent: 2048 can use `128/128`, while 3072 needs
      `64/16/256/64`-class chunking unless another large buffer is removed.
    - The length-dependent policy is now encoded in
      `resolve_long_sequence_chunks`: above 2048 tokens it caps the general
      pair/OPM/transition row chunk at `64` and defaults triangle/token
      attention chunks to the 3072-fit probe values
      (`triangle_attention_chunk=16`, `triangle_attention_q_chunk=256`,
      `token_attention_chunk=64`). Explicit smaller/manual overrides are still
      honored. This is weight-compatible; it changes only evaluation tiling.
    - `benchmark_flash_sampling_scale.py` can now enable JAX persistent
      compilation cache with `--compilation-cache-dir`. This does not reduce
      activation peak, but it is required for practical repeated long-shape
      sweeps because the 3072 fit probe spent most of its `1855.7 s`
      compile+first time in compilation/autotuning. Example:
      `--compilation-cache-dir outputs/jax_cache --cache-min-compile-time-secs 0`.
- Why AF3 reaches much longer lengths on the same GPU while this port did not:
  the gap was mainly memory policy, not JAX itself. AF3 defaults to broad bf16,
  residue-dependent pair attention chunks (`128` up to 1536, then `32`), and
  sharded pair transitions. The earlier Boltz-JAX path kept more fp32 islands
  and left several dense pair/attention/transition buffers too coarse inside one
  compiled graph. The current OOM-to-fit changes at 2048 and 3072 tokens
  support that diagnosis, but 4096/6000-token scaling is still unverified.
- Pallas triangle attention remains opt-in via `triangle_backend="pallas"`.
  Existing real-data parity evidence: 1UBQ_A 200-step XLA vs Pallas
  aligned RMSD `4.07e-05 A`, max abs `8.77e-05 A`
  (`outputs/triangle_pallas_realdata_parity.json`).
- Pallas E2E synthetic evidence at 512 tokens: Pallas triangle backend
  `7.64 s` vs XLA `9.19 s` steady, same peak class (`~6.2 GiB` JAX,
  `~10.9 GiB` process). The older 2048 rows in
  `outputs/triangle_backend_e2e.json` are invalid for the current code because
  they ran without `token_attention_chunk` and hit the known 48 GiB projection
  OOM.

Decision:

- Keep chunking/code-level memory reductions as the primary path. They preserve
  checkpoint/native-weight ABI and are deterministic.
- Use Pallas only behind explicit flags until long-length RMSD/speed is measured
  with `token_attention_chunk`.
- Buffer donation is benchmark-only for now. JAX docs note donation is only
  effective when the donated positional input is not reused and matches an
  output buffer; the sampler reuses params/features across iterations, so
  production donation needs a dedicated single-shot entry point.
- Host offload/remat is not a first-line inference fix: it mainly trades HBM for
  host traffic, and our current blocker is fused attention/projection
  activation, not static params.
