import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from boltz_jax.bridge.affinity_mapping import map_affinity_module_state_dict
from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.models.heads.affinity import affinity_module_forward

CHECKPOINT = Path(__file__).resolve().parents[2] / "boltz/.cache/boltz/boltz2_aff.ckpt"
BOLTZ_SRC = Path(__file__).resolve().parents[2] / "boltz/src"
PREFIX = "affinity_module1"

TOKEN_S = 384
TOKEN_Z = 128
# Real config from checkpoint hyper_parameters['affinity_model_args1'].
PAIRFORMER_ARGS = {
    "num_blocks": 8,
    "dropout": 0.25,
    "activation_checkpointing": False,
    "use_trifast": True,
}
TRANSFORMER_ARGS = {
    "num_blocks": 12,
    "num_heads": 8,
    "token_s": TOKEN_S,
    "activation_checkpointing": False,
}
NUM_DIST_BINS = 64
MAX_DIST = 22
GROUPS = {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 affinity checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def _build_torch_module(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.affinity import AffinityModule

    module = AffinityModule(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        pairformer_args=PAIRFORMER_ARGS,
        transformer_args=TRANSFORMER_ARGS,
        num_dist_bins=NUM_DIST_BINS,
        max_dist=MAX_DIST,
        use_cross_transformer=False,
        groups=GROUPS,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _inputs() -> dict[str, object]:
    rng = np.random.default_rng(0)
    n = 6  # tokens
    m = 10  # atoms
    s_inputs = rng.standard_normal((1, n, TOKEN_S)).astype(np.float32) * 0.3
    z = rng.standard_normal((1, n, n, TOKEN_Z)).astype(np.float32) * 0.2
    x_pred = rng.standard_normal((1, m, 3)).astype(np.float32) * 5.0

    # one rep atom per token (one-hot rows)
    t2r = np.zeros((1, n, m), dtype=np.float32)
    atom_idx = rng.choice(m, size=n, replace=True)
    for i, a in enumerate(atom_idx):
        t2r[0, i, a] = 1.0

    token_pad_mask = np.ones((1, n), dtype=np.float32)
    mol_type = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)  # 3 rec, 3 lig
    affinity_token_mask = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)
    return {
        "s_inputs": s_inputs,
        "z": z,
        "x_pred": x_pred,
        "token_to_rep_atom": t2r,
        "token_pad_mask": token_pad_mask,
        "mol_type": mol_type,
        "affinity_token_mask": affinity_token_mask,
    }


def test_affinity_module_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _build_torch_module(checkpoint_state)
    params = map_affinity_module_state_dict(checkpoint_state, PREFIX)
    data = _inputs()

    feats_torch = {
        "token_to_rep_atom": torch.from_numpy(data["token_to_rep_atom"]),
        "token_pad_mask": torch.from_numpy(data["token_pad_mask"]),
        "mol_type": torch.from_numpy(data["mol_type"]),
        "affinity_token_mask": torch.from_numpy(data["affinity_token_mask"]),
    }
    with torch.no_grad():
        out_torch = torch_module(
            s_inputs=torch.from_numpy(data["s_inputs"]),
            z=torch.from_numpy(data["z"]),
            x_pred=torch.from_numpy(data["x_pred"]),
            feats=feats_torch,
            multiplicity=1,
            use_kernels=False,
        )

    feats_jax = {
        "token_to_rep_atom": jnp.asarray(data["token_to_rep_atom"]),
        "token_pad_mask": jnp.asarray(data["token_pad_mask"]),
        "mol_type": jnp.asarray(data["mol_type"]),
        "affinity_token_mask": jnp.asarray(data["affinity_token_mask"]),
    }
    out_jax = affinity_module_forward(
        params,
        jnp.asarray(data["s_inputs"]),
        jnp.asarray(data["z"]),
        jnp.asarray(data["x_pred"]),
        feats_jax,
        multiplicity=1,
    )

    for key in ("affinity_pred_value", "affinity_logits_binary"):
        np.testing.assert_allclose(
            np.asarray(out_jax[key]),
            out_torch[key].detach().numpy(),
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"mismatch for {key}",
        )
