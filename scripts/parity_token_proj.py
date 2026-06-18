"""Parity: token_trans_bias per-layer loop vs the old stacked-einsum path.

Stage 1 rewrote ``_projection_list_forward`` to loop over the L layers (boltz
style) instead of materializing a [..., L, C] ``normed`` buffer. This checks the
new loop matches the old stacked einsum up to fp32 reduce-order (< ~5e-5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

from boltz_jax.models.diffusion.diffusion_conditioning import (  # noqa: E402
    _projection_list_forward,
)

rng = np.random.default_rng(0)
L, C, D = 24, 128, 16
params = [
    {
        "norm": {
            "scale": jnp.asarray(rng.standard_normal(C).astype("f4")),
            "bias": jnp.asarray(rng.standard_normal(C).astype("f4")),
        },
        "linear": {"kernel": jnp.asarray(rng.standard_normal((C, D)).astype("f4"))},
    }
    for _ in range(L)
]


def _old_stacked(params, x, eps):
    """The pre-Stage-1 stacked-einsum reference (materializes [..., L, C])."""
    scale = jnp.stack([layer["norm"]["scale"] for layer in params], axis=0)
    bias = jnp.stack([layer["norm"]["bias"] for layer in params], axis=0)
    kernel = jnp.stack([layer["linear"]["kernel"] for layer in params], axis=0)
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    x_n = ((xf - mean) * jax.lax.rsqrt(var + eps)).astype(x.dtype)
    normed = x_n[..., None, :] * scale + bias
    out = jnp.einsum("...lc,lcd->...ld", normed, kernel)
    return jnp.reshape(out, (*out.shape[:-2], out.shape[-2] * out.shape[-1]))


fnew = jax.jit(lambda x: _projection_list_forward(params, x, 1e-5))
fold = jax.jit(lambda x: _old_stacked(params, x, 1e-5))
for N in (256, 800, 1280):
    x = jnp.asarray(rng.standard_normal((1, N, N, C)).astype("f4"))
    a = fnew(x).block_until_ready()
    b = fold(x).block_until_ready()
    diff = float(jnp.max(jnp.abs(a - b)))
    print(f"N={N} max_abs_diff={diff:.3e} bitexact={bool(jnp.array_equal(a, b))}")
