import sys
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import map_atom_transformer_state_dict
from boltz_jax.models.atom import (
    atom_transformer_forward,
    get_indexing_matrix,
    single_to_keys,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[2] / "boltz/src"
PREFIX = "structure_module.score_model.atom_attention_encoder.atom_encoder"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_atom_window_indexing_matches_boltz_torch() -> None:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import get_indexing_matrix as torch_index

    actual = get_indexing_matrix(k=2, w=32, h_keys=128)
    expected = torch_index(K=2, W=32, H=128, device="cpu")

    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_single_to_keys_matches_boltz_torch() -> None:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    single = torch.arange(1 * 64 * 5, dtype=torch.float32).reshape(1, 64, 5)
    expected = torch_to_keys(
        single,
        indexing_matrix=torch_index(K=2, W=32, H=128, device="cpu"),
        W=32,
        H=128,
    )

    indexing = get_indexing_matrix(k=2, w=32, h_keys=128)
    actual = single_to_keys(
        jnp.asarray(single.numpy()), indexing, w=32, h_keys=128
    )

    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_checkpoint_atom_transformer_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_atom_transformer(checkpoint_state)
    params = map_atom_transformer_state_dict(
        checkpoint_state,
        PREFIX,
        num_heads=4,
    )
    q, c, bias, mask = _atom_transformer_inputs()

    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    torch_keys = torch_index(K=2, W=32, H=128, device="cpu")
    torch_to_keys_fn = partial(torch_to_keys, indexing_matrix=torch_keys, W=32, H=128)
    jax_keys = get_indexing_matrix(k=2, w=32, h_keys=128)

    with torch.no_grad():
        expected = torch_module(
            q=q,
            c=c,
            bias=bias,
            to_keys=torch_to_keys_fn,
            mask=mask,
            multiplicity=1,
        )
    actual = atom_transformer_forward(
        params,
        q=jnp.asarray(q.numpy()),
        c=jnp.asarray(c.numpy()),
        bias=jnp.asarray(bias.numpy()),
        to_keys=lambda x: single_to_keys(x, jax_keys, w=32, h_keys=128),
        mask=jnp.asarray(mask.numpy()),
        attn_window_queries=32,
        attn_window_keys=128,
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_atom_transformer(
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.transformersv2 import AtomTransformer

    module = AtomTransformer(
        attn_window_queries=32,
        attn_window_keys=128,
        depth=3,
        heads=4,
        dim=128,
        dim_single_cond=128,
        post_layer_norm=False,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _atom_transformer_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    atoms = 64
    q = torch.linspace(-0.2, 0.2, steps=atoms * 128).reshape(1, atoms, 128)
    c = torch.linspace(0.25, -0.25, steps=atoms * 128).reshape(1, atoms, 128)
    bias = torch.linspace(-0.1, 0.1, steps=2 * 32 * 128 * 12)
    bias = bias.reshape(1, 2, 32, 128, 12)
    mask = torch.ones(1, atoms, dtype=torch.float32)
    mask[:, -3:] = 0.0
    return q, c, bias, mask
