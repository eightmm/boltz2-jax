# boltz2-jax optimization roadmap

Inference-only, weight-compatible (model math unchanged) — only the evaluation
graph may change. Measured baseline (RTX PRO 6000 Blackwell sm120, 1024-residue,
50 steps, recycling 3): JAX fp32 = **1.66x** faster than torch Boltz at **−3%**
peak; **bf16-mixed = 2.12x** + **−17%** peak (10.2 GiB). bf16-mixed = trunk in
bf16, diffusion structure module kept fp32 (Boltz profile).

## Done
- Chunking: triangle / OuterProductMean / transition / token-attention.
- Fused kernels: tokamax (Triton) + custom Pallas flash (opt-in).
- bf16-mixed precision profile (trunk bf16, diffusion fp32 island).
- Projection fusion (q/g, k/v, fc1/fc2); lazy token pair-bias.
- `lax.scan` over trunk + 200-step sampling loop; trunk computed once.
- TF32 matmul-precision flag; fp16-safe additive masks.
- Persistent compilation cache wired into `scripts/predict.py` (`--compile-cache`).
- Recycling loop via `lax.scan` (under `use_scan`): trunk traced once instead of
  unrolling `recycling_steps+1` copies into the HLO. CPU compile of the rc=3
  trunk dropped **221.96s → 1.46s** (~150x); coords scan-vs-eager RMSD 7e-5 Å
  (max 1e-3), trunk latents differ ~5e-5 relative (fp32 reassociation, does NOT
  grow with steps). Eager path retained for parity debugging.

## Measured findings (this round)

- **Buffer donation: no effect.** `donate_argnums=(params,feats)` on the sampler
  jit at 1024-residue gave Δpeak ≈ 0 (11891 → 11892 MiB). Params are live through
  the whole graph and the output (coords) is tiny, so there is nothing to alias;
  the memory-dominant pair scan carry is already aliased by `lax.scan`. Donation
  is effectively already done — drop it from the list.
- **HLO peak buffer = the pair tensor.** The largest XLA buffer-assignment module
  at 1024 is exactly 536,870,912 B = 512 MiB = `[1,N,N,128]` fp32 pair tensor
  (`[1,1024,1024,1,64]` per-head views). Plus resident params (~2.9 GiB). bf16
  halves the pair → matches the measured −17% peak.
- **Stray copy in the triangle path.** The peak module holds `copy(args_0)` of a
  pair-sized `[1,1024,1024,1,64]` (268 MiB) — an avoidable duplication from the
  `swapaxes` layout round-trip in XLA triangle attention. The tokamax/pallas
  triangle backends avoid it (the kernel handles layout). Targeted lever:
  eliminate the swapaxes copy (fold into einsum subscripts) or route long-N
  triangle through the kernel backend.
- **Compilation is mandatory.** JAX eager (no jit) at 1024 is 1.59x SLOWER than
  torch eager with far higher memory (process 33.4 vs 17.2 GiB). The entire JAX
  win (jitted 1.66x faster, peak ≤ torch) comes from XLA fusion — eager
  materializes every op (the 268 MiB copies included). Production must jit; this
  is why shape bucketing + persistent compile cache (#3) is required for serving.

## Next (ranked — updated by the measured findings above; all math-unchanged)

1. **Eliminate the triangle `swapaxes` copy** (HLO showed a 268 MiB pair-sized
   `copy` in the XLA triangle path). Fold the head-axis layout into the einsum
   subscripts so XLA picks the contraction layout without a physical transpose,
   or route long-N triangle attention through the tokamax/pallas kernel backend
   (which avoids the copy). Direct peak win, math-unchanged. Guard with the
   triangle chunk-parity test.
2. **Free the full pair tensor before diffusion.** If the sampler only needs a
   pair-derived bias, precompute the smallest exact conditioning tensor and drop
   the [N,N,C] pair before the 200-step loop. Pair is an O(N²·C) hard floor
   (N=3000 bf16 ≈ 2.15 GiB) — can only cut transients/lifetime, not existence.
3. ~~Buffer donation~~ — MEASURED Δpeak ≈ 0 (see findings); `lax.scan` already
   aliases the dominant carries. Dropped.
4. **Shape bucketing + persistent compile cache + persistent tokamax/Triton
   autotune cache.** Pad sequence length to an N-ladder (e.g. 1024/1536/2048/3072)
   so the "hundreds of distinct shapes" collapse to a handful — kills the tokamax
   autotuning stall and XLA recompiles. Pads FLOPs; net win at low request counts.
   Must mask padded positions (math unchanged). Cache keyed by driver/XLA version.
4. **Free the full pair tensor before diffusion.** If the sampler only needs a
   pair-derived bias, precompute the smallest exact conditioning tensor and drop
   the [N,N,C] pair before the 200-step loop. Pair is an O(N²·C) hard floor
   (N=3000 bf16 ≈ 2.15 GiB) — can only cut transients/lifetime, not existence.
5. **Recycling via `lax.scan` with donated bf16 carries**; cache static
   embeddings / masks / relative-position encodings across recycles.
6. **Layout audit.** Inspect HLO for stray `transpose`/`convert`/`copy` on the
   pair tensor from layout mismatch — one of these silently doubles peak.

## remat for inference — verdict
`jax.remat` (activation checkpointing) is a TRAINING tool: it trades a saved
backward tape for recompute. Inference has no backward tape, so training-style
remat does not apply. The *recompute-to-lower-peak* idea still works, but achieve
it explicitly via chunk→donated-buffer streaming, **not** by relying on
`jax.remat` (which on a pure forward graph may be neutral or act as a harmful CSE
barrier). Advisors split on wording but converge on: don't bet on `jax.remat`;
control transient materialization through the scan/chunk structure.

## Out of contract (changes math/output — only if the constraint is relaxed)
Low-rank / sparse pair, quantization below bf16, fewer or early-stopped
recycles, approximate attention. Custom kernels must be validated vs fp32 on
small N (precision/masking drift risk).

## Serving (post code-freeze)
Compilation cache + autotune cache + shape bucketing make repeated/large-batch
serving practical. Multi-GPU `shard_map` (N-axis sharding of the pair tensor;
hooks exist in `trunk.py`) is the path beyond single-GPU length limits.
