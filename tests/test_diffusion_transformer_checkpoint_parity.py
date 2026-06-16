import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_diffusion_transformer_layer_state_dict
from boltz_jax.models.diffusion_transformer import diffusion_transformer_layer_forward

CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[2] / "boltz/src"
PREFIX = "structure_module.score_model.token_transformer.layers.0"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_diffusion_transformer_layer_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_layer(checkpoint_state)
    params = map_diffusion_transformer_layer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=8,
    )
    a, s, bias, mask = _layer_inputs()

    with torch.no_grad():
        expected = torch_module(
            a,
            s,
            bias=bias,
            mask=mask,
            multiplicity=1,
        )
    actual = diffusion_transformer_layer_forward(
        params,
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        jnp.asarray(bias.numpy()),
        jnp.asarray(mask.numpy()),
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_layer(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.transformersv2 import DiffusionTransformerLayer

    module = DiffusionTransformerLayer(
        heads=8,
        dim=768,
        dim_single_cond=768,
        post_layer_norm=False,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _layer_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residues = 4
    a_values = torch.linspace(-0.25, 0.25, steps=residues * 768)
    s_values = torch.linspace(0.3, -0.3, steps=residues * 768)
    bias_values = torch.linspace(-0.1, 0.1, steps=residues * residues * 8)
    a = a_values.reshape(1, residues, 768)
    s = s_values.reshape(1, residues, 768)
    bias = bias_values.reshape(1, residues, residues, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    return a, s, bias, mask
