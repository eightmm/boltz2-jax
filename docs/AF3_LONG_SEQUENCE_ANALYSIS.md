# AF3 Long-Sequence Memory Analysis

This note compares the local AlphaFold 3 codebase against the current
Boltz-JAX port, focusing on why AF3 can handle much longer token lengths on the
same GPU class.

## AF3 Mechanisms

### Global Policy

Source: `alphafold3/src/alphafold3/model/model_config.py`.

- `bfloat16 = "all"` is the default global precision policy.
- Pair attention uses length-dependent chunks:
  `pair_attention_chunk_size = ((1536, 128), (None, 32))`.
- Pair transition uses length-dependent sharding:
  `pair_transition_shard_spec = ((2048, None), (None, 1024))`.
- Pair attention defaults to Tokamax/Triton:
  `flash_attention_implementation = "triton"`.

Interpretation: AF3 is not relying on one large dense XLA graph. Long sequence
support is built from a default bf16 activation policy plus module-level
subbatching, sharding, and flash attention.

### Precision Boundaries

Sources:

- `model/model.py:create_target_feat_embedding`
- `network/evoformer.py:Evoformer.__call__`
- `network/diffusion_head.py:DiffusionHead.__call__`
- `components/utils.py:bfloat16_context`

AF3 runs most module internals inside `utils.bfloat16_context()`, whose custom
getter casts bf16 parameters when the requested original dtype is bf16. The
Evoformer target features are explicitly cast to bf16. Recycle outputs are then
cast back to fp32 at the model boundary:

- `embeddings["pair"] = embeddings["pair"].astype(jnp.float32)`
- `embeddings["single"] = embeddings["single"].astype(jnp.float32)`

Interpretation: AF3 keeps large intermediates mostly bf16 but uses fp32 at
selected boundaries and numerically sensitive spots.

### Pair Attention

Source: `network/modules.py:GridSelfAttention`.

AF3 pair attention:

- Projects pair activations to Q/K/V.
- Builds pair bias per head.
- Uses `tokamax.dot_product_attention`.
- Runs through `mapping.inference_subbatch`.
- Chooses subbatch size from `pair_attention_chunk_size`.

For `N > 1536`, the pair attention subbatch is 32. This limits the live
attention score workspace. Tokamax/Triton also avoids materializing the same
dense score buffers that a plain XLA implementation tends to build.

Boltz-JAX status:

- XLA triangle attention now has an outer chunk of 32 above 1536 and an inner
  query-row chunk of 512 above 2048.
- Pallas triangle attention exists but is still opt-in.
- Token attention chunking exists for diffusion.
- Tokamax/flash was tested and was slower at short/mid lengths with Boltz's
  bias-heavy shapes.

Remaining gap: AF3's long path is the default and backed by Tokamax/Triton for
pair attention. Boltz-JAX still falls back to dense XLA for several pair paths.

### Pair Transition

Sources:

- `network/modules.py:PairFormerIteration`
- `network/modules.py:EvoformerIteration`
- `components/mapping.py:sharded_apply`

AF3 wraps pair transition blocks in `mapping.sharded_apply` when
`shard_transition_blocks` is true. For `N > 2048`, shard size becomes 1024.
`sharded_apply` slices the mapped axis, applies the transition block shard by
shard, and writes each shard into an output buffer via dynamic update.

Boltz-JAX status:

- Pair `transition_z` has row chunking with `row_chunk_size=chunk_size`.
- This is more aggressive than AF3's row shard at current defaults (`128` vs
  AF3's `1024` above 2048).
- Hidden-dimension chunking exists in `transition_forward` and is now exposed to
  Pairformer / MSA pair-only `transition_z` via the optional
  `transition_hidden_chunk` argument. The default remains `None` to preserve the
  original fp32 accumulation order.

Remaining gap: row sharding is present and hidden chunking is available, but the
long-length effect still needs a 2048/2560 benchmark with
`transition_hidden_chunk` enabled.

### Outer Product Mean

Source: `network/modules.py:OuterProductMean`.

AF3 computes OPM in chunks over the residue axis using
`mapping.inference_subbatch`. It avoids the full `[N, N, c, d]` intermediate at
once.

Boltz-JAX status:

- OPM is already chunked over token `i`.
- Output is cast back to the input dtype.

Remaining gap: this part is now broadly aligned.

### Triangle Multiplication

Source: `network/modules.py:TriangleMultiplication`.

AF3 uses Tokamax GLU for projection/gating, transposes to channel-first, and
runs the triangle contraction under the global bf16 context unless an op forces
otherwise.

Boltz-JAX status:

- Triangle multiplication chunks the output `i` axis.
- It casts projected activations to fp32 before contraction for stability, then
  casts the contraction output back to the input activation dtype before the
  output norm/projection.

Status: the prior bf16 activation leak is fixed while keeping the fp32 default
path unchanged.

### Layer Stacks And Compile Shape

Sources:

- `network/evoformer.py`
- `network/diffusion_transformer.py`

AF3 uses `hk.experimental.layer_stack` for Evoformer, Pairformer, diffusion
transformer super-blocks, and atom cross-attention transformer blocks. Diffusion
transformer also groups layers into super-blocks and precomputes pair logits per
super-block rather than for all 24 blocks at once.

Boltz-JAX status:

- `lax.scan` exists for trunk and score stacks.
- Layer stack/scan behavior is less uniform across modules.
- The 3072-token fp32 full graph probe was interrupted during
  `compile_and_load` after GPU memory reached about 94.7 GiB, showing compile
  and executable memory are also practical bottlenecks.

Remaining gap: better super-blocking and persistent compilation cache are needed
for repeated long-shape serving. This does not replace activation memory fixes.

### Diffusion Sampling

Source: `network/diffusion_head.py:sample`.

AF3 sampling:

- Uses 200 steps by default.
- Runs the denoising loop with `hk.scan(..., unroll=4)`.
- Vmapps over samples.
- Uses dense atom shape `(num_samples, num_tokens, max_atoms_per_token, 3)`.

Boltz-JAX status:

- Sampling loop can use `lax.scan`.
- Multiplicity/sample axis is supported.
- Lazy token transition bias avoids a large `[N, N, layers * heads]` bias.

Remaining gap: Boltz-JAX still carries Boltz-specific atom windows and token
bias paths; sample count >1 must be tested separately at long lengths.

### Atom Path

Source: `network/atom_cross_attention.py`.

AF3 atom attention is not full atom-to-atom attention. It uses `atom_layout`
`GatherInfo` objects to convert token atoms into query/key subsets, then runs
local subset cross-attention via `CrossAttTransformer`. Pair conditioning from
trunk pair activations is gathered into local query-key atom pairs.

Boltz-JAX status:

- Boltz atom path uses windowed atom attention and token/atom gather-scatter.
- Lazy token transition bias has reduced one major token-pair memory issue.

Remaining gap: long-length atom-path memory should be profiled separately from
trunk memory. AF3's atom layout is explicitly local-subset oriented.

## Main Differences Versus Boltz-JAX

1. AF3's bf16 policy is default and broad; Boltz-JAX low precision is opt-in and
   still has fp32 leaks.
2. AF3 pair attention uses Tokamax/Triton plus length-dependent subbatching by
   default; Boltz-JAX still relies mostly on XLA chunking and optional Pallas.
3. AF3 transition sharding is consistently built into pair stacks; Boltz-JAX row
   chunking exists but hidden-dimension and dtype behavior still need work.
4. AF3 uses Haiku `layer_stack` and scan/super-block patterns consistently;
   Boltz-JAX has mixed unroll/scan paths.
5. AF3 diffusion transformer projects pair logits per super-block; Boltz-JAX
   lazy token bias reduces full bias materialization but the long-shape score
   path still needs stress testing.
6. AF3 atom attention is local-subset by construction; Boltz-JAX has windowing,
   but its Boltz-specific pair-conditioned atom path needs separate long-N
   profiling.
7. AF3 accepts selected fp32 islands but avoids keeping entire pair activations
   fp32 through the whole trunk. Boltz-JAX currently risks fp32 promotion after
   triangle multiplication.

## Implementation Priority For Boltz-JAX

1. Continue fixing bf16 activation leaks without changing weight ABI.
   - `triangle_multiplication_forward` now preserves bf16 output dtype.
   - trunk bond/contact conditioning, MSA pair-weighted averaging, and triangle
     attention now preserve bf16 activation dtype in the opt-in low-precision
     path. At 1024 tokens this reduced XLA allocator peak (`12.13 -> 10.33 GiB`)
     and steady time (`55.86 -> 41.95 s`).
   - Preserve fp32 default behavior for every dtype fix.
   - Add dtype-specific tests for each module touched.

2. Re-run bf16 long-length probes.
   - 2048 bf16 with current chunks now fits with headroom: allocator
     `18.12 GiB`, process `33.51 GiB`.
   - 3072 bf16 with 2048-style chunks still OOMs on a `33.83 GiB` allocation.
   - 3072 bf16 with tighter chunks (`chunk_size=64`,
     `triangle_attention_chunk=16`, `triangle_attention_q_chunk=256`,
     `token_attention_chunk=64`) fits: allocator `31.23 GiB`, process
     `66.53 GiB`, but compile+first is `1855.7 s`.
   - Next: move from manual probes to an automatic length-dependent chunk
     policy, then re-test 3072 steady time and 4096 fit.

3. Benchmark hidden-dimension transition chunk policy for long N.
   - Existing `transition_forward(chunk_size=...)` supports hidden chunking.
   - Pairformer now exposes `transition_hidden_chunk`; measure 2048/2560 peak
     before making any auto policy.

4. Make Pallas/Tokamax decisions shape-specific.
   - Tokamax was slower for short/mid Boltz biased attention shapes.
   - Pallas triangle attention now works inside the full synthetic sampler
     benchmark. At 512 and 1024 tokens with 20 sampling steps it improved steady
     time (`8.93 -> 7.63 s`, `55.86 -> 42.84 s`) but did not reduce measured
     allocator/process peak. This suggests the current peak gap versus AF3 is
     not only dense triangle-attention score materialization.
   - After the Stage13 bf16 dtype-leak fixes, Pallas at 1024 tokens improves
     both speed and peak: `31.86 s`, allocator `9.88 GiB`, process `17.04 GiB`
     versus the prior XLA baseline `55.86 s`, allocator `12.13 GiB`, process
     `25.24 GiB`.
   - At 2048 tokens, the dtype-leak fixes are the main memory win: bf16 XLA is
     now allocator `18.12 GiB`, process `33.51 GiB`, down from the earlier
     bf16 process peak around `66.3 GiB`. Pallas keeps the same peak class and
     improves speed (`303.54 -> 223.62 s`).
   - At 3072 tokens, Pallas plus tighter chunks fits but is very slow to
     compile/run. Longer Pallas runs still need RMSD/peak validation before
     enabling it by default.

5. Add compile-cache support for benchmark/serving scripts.
   - `benchmark_flash_sampling_scale.py` now supports
     `--compilation-cache-dir` plus persistent-cache thresholds. This targets
     repeated fixed-shape sweeps; it does not reduce activation peak.
   - Serving entry points still need a production cache/shape-bucket wrapper.
