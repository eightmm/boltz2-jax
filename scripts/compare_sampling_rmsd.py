"""Deterministic PyTorch-vs-JAX diffusion sampling parity comparison.

Runs the JAX structure sampler (``boltz2_sample_forward``) against a PyTorch
reference that mirrors the SAME no-augmentation / alignment_reverse_diff Euler
loop, using IDENTICAL externally generated noise. The only remaining source of
difference is framework numerical drift in trunk + conditioning + score.

Reports raw and Kabsch-aligned RMSD over real (unmasked) atoms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_boltz2_graph_state_dict
from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward

# Reuse torch graph wiring + feature loader from the benchmark script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_boltz2_graph import (  # noqa: E402
    BOLTZ_SRC,
    _load_features_pt,
    _load_torch_graph,
    _tree_to_jax,
    _tree_to_torch,
)

# Sampler default constants (must match boltz2_sample_forward).
SIGMA_MIN = 0.0004
SIGMA_MAX = 160.0
SIGMA_DATA = 16.0
RHO = 7.0
GAMMA_0 = 0.8
GAMMA_MIN = 1.0
NOISE_SCALE = 1.003
STEP_SCALE = 1.5


def _sample_schedule_torch(num_steps: int, device: str) -> torch.Tensor:
    """Port of trunk._sample_schedule (returns sigmas padded with a final 0)."""
    inv_rho = 1.0 / RHO
    steps = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigmas = (
        SIGMA_MAX**inv_rho
        + steps / (num_steps - 1) * (SIGMA_MIN**inv_rho - SIGMA_MAX**inv_rho)
    ) ** RHO
    sigmas = sigmas * SIGMA_DATA
    return torch.nn.functional.pad(sigmas, (0, 1))


def _torch_sample(
    torch_model: torch.nn.Module,
    feats: dict,
    num_steps: int,
    init_noise: np.ndarray,
    step_noises: np.ndarray,
    device: str,
    recycling_steps: int = 0,
    trunk_autocast_dtype: "torch.dtype | None" = None,
) -> np.ndarray:
    """Mirror boltz2_sample_forward no-steering / no-augmentation branch.

    ``recycling_steps`` MUST match the JAX run: the trunk runs
    ``recycling_steps + 1`` passes, recycling (s, z) each iteration, exactly as
    ``boltz_graph_forward`` does on the JAX side.

    ``trunk_autocast_dtype`` (e.g. ``torch.bfloat16``) runs the trunk under
    autocast and casts (s, z) back to fp32 before diffusion — Boltz's
    ``bf16-mixed`` profile (bf16 trunk, fp32 diffusion island). ``None`` = fp32.
    """
    import contextlib  # noqa: PLC0415

    from boltz.model.loss.diffusionv2 import weighted_rigid_align  # noqa: PLC0415

    m = torch_model
    autocast = (
        torch.autocast("cuda", dtype=trunk_autocast_dtype)
        if trunk_autocast_dtype is not None
        else contextlib.nullcontext()
    )
    # --- Build trunk + diffusion conditioning ONCE (does not depend on r_noisy).
    with autocast:
        s_inputs = m.input_embedder(feats)
        s_init = m.s_init(s_inputs)
        z_init = m.z_init_1(s_inputs)[:, :, None] + m.z_init_2(s_inputs)[:, None, :]
        rel_pos = m.rel_pos(feats)
        z_init = z_init + rel_pos
        z_init = z_init + m.token_bonds(feats["token_bonds"].float())
        z_init = z_init + m.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + m.contact_conditioning(feats)
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        # recycling: run (recycling_steps + 1) passes, recycling (s, z) each time.
        for _ in range(recycling_steps + 1):
            s = s_init + m.s_recycle(m.s_norm(s))
            z = z_init + m.z_recycle(m.z_norm(z))
            z = z + m.msa_module(z, s_inputs, feats, use_kernels=False)
            s, z = m.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask, use_kernels=False
            )
    # Boltz casts trunk outputs back to fp32 (diffusion is an fp32 island).
    s_inputs, s, z, rel_pos = (
        s_inputs.float(), s.float(), z.float(), rel_pos.float()
    )
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        m.diffusion_conditioning(s, z, rel_pos, feats)
    )
    cond = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }

    atom_mask = feats["atom_pad_mask"].float()  # (1, n_atoms)
    init_t = torch.as_tensor(init_noise, device=device, dtype=torch.float32)
    step_t = torch.as_tensor(step_noises, device=device, dtype=torch.float32)

    sigmas = _sample_schedule_torch(num_steps, device)
    gammas = torch.where(
        sigmas > GAMMA_MIN,
        torch.tensor(GAMMA_0, device=device),
        torch.tensor(0.0, device=device),
    )

    def precond_score(r_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        padded_sigma = sigma.reshape(1, 1, 1)
        scaled_input = r_noisy / torch.sqrt(padded_sigma**2 + SIGMA_DATA**2)
        times = torch.full(
            (r_noisy.shape[0],),
            float(torch.log(sigma / SIGMA_DATA) * 0.25),
            device=device,
            dtype=r_noisy.dtype,
        )
        r_update = m.score_model(
            s_inputs=s_inputs,
            s_trunk=s,
            r_noisy=scaled_input,
            times=times,
            feats=feats,
            diffusion_conditioning=cond,
            multiplicity=1,
        )
        c_skip = SIGMA_DATA**2 / (padded_sigma**2 + SIGMA_DATA**2)
        c_out = (
            padded_sigma * SIGMA_DATA
            / torch.sqrt(SIGMA_DATA**2 + padded_sigma**2)
        )
        return c_skip * r_noisy + c_out * r_update

    atom_coords = sigmas[0] * init_t

    for step_idx in range(num_steps):
        sigma_tm = sigmas[step_idx]
        sigma_t = sigmas[step_idx + 1]
        gamma = gammas[step_idx + 1]

        t_hat = sigma_tm * (1.0 + gamma)
        noise_var = torch.clamp(NOISE_SCALE**2 * (t_hat**2 - sigma_tm**2), min=0.0)
        noise = torch.sqrt(noise_var) * step_t[step_idx]
        atom_coords_noisy = atom_coords + noise
        atom_coords_denoised = precond_score(atom_coords_noisy, t_hat)

        atom_coords_noisy = weighted_rigid_align(
            atom_coords_noisy.float(),
            atom_coords_denoised.float(),
            atom_mask,
            atom_mask,
        ).to(atom_coords_denoised.dtype)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords = (
            atom_coords_noisy
            + STEP_SCALE * (sigma_t - t_hat) * denoised_over_sigma
        )

    return atom_coords.detach().cpu().numpy()


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Aligned RMSD: rigidly superpose ``a`` onto ``b`` (numpy SVD)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    cov = ac.T @ bc
    u, _, vh = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vh))
    dmat = np.diag([1.0, 1.0, d])
    rot = u @ dmat @ vh
    a_aligned = ac @ rot
    return float(np.sqrt(np.mean(np.sum((a_aligned - bc) ** 2, axis=1))))


def _raw_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../boltz/.cache/boltz/boltz2_conf.ckpt"),
    )
    parser.add_argument(
        "--features-pt", type=Path, default=Path("outputs/real_features/1UBQ_A.pt")
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 25])
    parser.add_argument("--recycling", type=int, default=3)
    parser.add_argument("--msa-layers", type=int, default=4)
    parser.add_argument("--pairformer-layers", type=int, default=64)
    parser.add_argument("--token-layers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sampling_rmsd_comparison.json")
    )
    args = parser.parse_args()

    assert BOLTZ_SRC.exists(), f"boltz src not found: {BOLTZ_SRC}"

    state_cpu = load_checkpoint_state_dict(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = _load_torch_graph(
        state_cpu, args.msa_layers, args.pairformer_layers, args.token_layers, device
    )
    jax_params = map_boltz2_graph_state_dict(
        state_cpu,
        num_msa_layers=args.msa_layers,
        num_pairformer_layers=args.pairformer_layers,
        num_token_layers=args.token_layers,
        token_transformer_heads=16,
    )

    feats_np, record_id = _load_features_pt(args.features_pt)
    n_atoms = int(feats_np["atom_pad_mask"].shape[1])
    atom_mask = np.asarray(feats_np["atom_pad_mask"]).reshape(-1).astype(bool)
    real_idx = np.where(atom_mask)[0]

    torch_feats = _tree_to_torch(feats_np, device)
    jax_feats = _tree_to_jax(feats_np)

    results = {}
    for num_steps in args.steps:
        rng = np.random.default_rng(args.seed)
        init_noise = rng.standard_normal((1, n_atoms, 3)).astype(np.float32)
        step_noises = rng.standard_normal((num_steps, 1, n_atoms, 3)).astype(np.float32)

        with torch.no_grad():
            torch_coords = _torch_sample(
                torch_model, torch_feats, num_steps, init_noise, step_noises,
                device, recycling_steps=args.recycling,
            )

        jax_out = boltz2_sample_forward(
            jax_params,
            jax_feats,
            jax.random.PRNGKey(0),
            num_sampling_steps=num_steps,
            recycling_steps=args.recycling,
            token_layers=args.token_layers,
            augmentation=False,
            alignment_reverse_diff=True,
            init_noise=jnp.asarray(init_noise),
            step_noises=jnp.asarray(step_noises),
        )
        jax_coords = np.asarray(jax_out["sample_atom_coords"])

        t = torch_coords[0][real_idx]
        j = jax_coords[0][real_idx]
        raw = _raw_rmsd(t, j)
        aligned = _kabsch_rmsd(t, j)
        max_abs = float(np.max(np.abs(t - j)))
        results[str(num_steps)] = {
            "raw_rmsd_angstrom": raw,
            "kabsch_aligned_rmsd_angstrom": aligned,
            "max_abs_coord_diff": max_abs,
        }
        print(
            f"steps={num_steps:>3}  raw_RMSD={raw:.6f} A  "
            f"aligned_RMSD={aligned:.6f} A  max|d|={max_abs:.6f}"
        )

    payload = {
        "record_id": record_id,
        "features_pt": str(args.features_pt),
        "n_atoms": n_atoms,
        "n_real_atoms": int(real_idx.size),
        "torch_device": device,
        "recycling_steps": args.recycling,
        "augmentation": False,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
