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

## Next (ranked, from 3-model advisory synthesis — all math-unchanged)

1. **HLO buffer-assignment dump first.** `--xla_dump_to` → read the
   buffer-assignment summary to find the true peak buffer set. Optimize targeted,
   not blind.
2. **Buffer donation.** `jax.jit(donate_argnums=...)` + donated `lax.scan`
   carries for pair/single/coords so XLA aliases storage and never holds the old
   and new pair tensor simultaneously. Biggest math-safe memory win, no recompute
   cost. Caveat: donated buffers must not be reused in Python; the heads run after
   sampling, so scope donation to the sampler stage only.
3. **Shape bucketing + persistent compile cache + persistent tokamax/Triton
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
