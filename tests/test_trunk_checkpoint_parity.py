import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
from boltz_jax.bridge.torch_mapping import (
    map_boltz2_graph_state_dict,
    map_boltz2_trunk_state_dict,
)
from boltz_jax.models.trunk import boltz2_graph_score_forward, boltz2_trunk_forward

CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[2] / "boltz/src"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_boltz2_trunk_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    num_msa_layers = 1
    num_pairformer_layers = 1
    torch_module = _load_torch_trunk(
        checkpoint_state,
        num_msa_layers,
        num_pairformer_layers,
    )
    params = map_boltz2_trunk_state_dict(
        checkpoint_state,
        num_msa_layers=num_msa_layers,
        num_pairformer_layers=num_pairformer_layers,
    )
    feats = _trunk_feats()

    with torch.no_grad():
        expected = torch_module(feats, recycling_steps=0)
    actual = boltz2_trunk_forward(params, _jax_feats(feats), recycling_steps=0)

    np.testing.assert_allclose(
        np.asarray(actual["s_inputs"]),
        expected["s_inputs"].detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["relative_position_encoding"]),
        expected["relative_position_encoding"].detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["s"]),
        expected["s"].detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["z"]),
        expected["z"].detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )


def test_checkpoint_boltz2_graph_score_jits(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    params = map_boltz2_graph_state_dict(
        checkpoint_state,
        num_msa_layers=1,
        num_pairformer_layers=1,
        num_token_layers=1,
        token_transformer_heads=16,
    )
    feats = _jax_feats(_trunk_feats())
    r_noisy = jnp.linspace(0.15, -0.15, num=64 * 3, dtype=jnp.float32).reshape(
        1, 64, 3
    )
    times = jnp.asarray([0.17], dtype=jnp.float32)
    compiled = jax.jit(
        boltz2_graph_score_forward,
        static_argnames=("recycling_steps", "token_layers", "multiplicity"),
    )
    actual = compiled(
        params,
        feats,
        r_noisy,
        times,
        recycling_steps=0,
        token_layers=1,
        multiplicity=1,
    )
    assert actual.shape == (1, 64, 3)
    assert bool(jnp.all(jnp.isfinite(actual)))


def _load_torch_trunk(
    state: dict[str, torch.Tensor],
    num_msa_layers: int,
    num_pairformer_layers: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.layers.pairformer import PairformerModule
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    from boltz.model.modules.trunkv2 import (
        ContactConditioning,
        InputEmbedder,
        MSAModule,
    )

    class TrunkSubset(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_embedder = InputEmbedder(
                atom_s=128,
                atom_z=16,
                token_s=384,
                token_z=128,
                atoms_per_window_queries=32,
                atoms_per_window_keys=128,
                atom_feature_dim=388,
                atom_encoder_depth=3,
                atom_encoder_heads=4,
                add_method_conditioning=True,
                add_modified_flag=True,
                add_cyclic_flag=True,
                add_mol_type_feat=True,
            )
            self.s_init = torch.nn.Linear(384, 384, bias=False)
            self.z_init_1 = torch.nn.Linear(384, 128, bias=False)
            self.z_init_2 = torch.nn.Linear(384, 128, bias=False)
            self.rel_pos = RelativePositionEncoder(128)
            self.token_bonds = torch.nn.Linear(1, 128, bias=False)
            self.token_bonds_type = torch.nn.Embedding(7, 128)
            self.contact_conditioning = ContactConditioning(128, 4.0, 20.0)
            self.s_norm = torch.nn.LayerNorm(384)
            self.z_norm = torch.nn.LayerNorm(128)
            self.s_recycle = torch.nn.Linear(384, 384, bias=False)
            self.z_recycle = torch.nn.Linear(128, 128, bias=False)
            self.msa_module = MSAModule(
                msa_s=64,
                token_z=128,
                token_s=384,
                msa_blocks=num_msa_layers,
                msa_dropout=0.15,
                z_dropout=0.25,
                pairwise_head_width=32,
                pairwise_num_heads=4,
                use_paired_feature=True,
            )
            self.pairformer_module = PairformerModule(
                token_s=384,
                token_z=128,
                num_blocks=num_pairformer_layers,
                num_heads=16,
                pairwise_head_width=32,
                pairwise_num_heads=4,
                v2=True,
            )

        def forward(self, feats, recycling_steps=0):
            s_inputs = self.input_embedder(feats)
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + self.contact_conditioning(feats)
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]
            for _ in range(recycling_steps + 1):
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))
                z = z + self.msa_module(z, s_inputs, feats, use_kernels=False)
                s, z = self.pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=False,
                )
            return {
                "s_inputs": s_inputs,
                "s": s,
                "z": z,
                "relative_position_encoding": relative_position_encoding,
            }

    module = TrunkSubset().eval()
    module.load_state_dict(_trunk_state(state, num_msa_layers, num_pairformer_layers))
    return module


def _trunk_state(
    state: dict[str, torch.Tensor],
    num_msa_layers: int,
    num_pairformer_layers: int,
) -> dict[str, torch.Tensor]:
    prefixes = (
        "input_embedder.",
        "s_init.",
        "z_init_1.",
        "z_init_2.",
        "rel_pos.",
        "token_bonds.",
        "token_bonds_type.",
        "contact_conditioning.",
        "s_norm.",
        "z_norm.",
        "s_recycle.",
        "z_recycle.",
        "msa_module.",
        "pairformer_module.",
    )
    module_state = {}
    for key, value in state.items():
        if not key.startswith(prefixes):
            continue
        if key.startswith("msa_module.layers."):
            index = int(key.split(".")[2])
            if index >= num_msa_layers:
                continue
        if key.startswith("pairformer_module.layers."):
            index = int(key.split(".")[2])
            if index >= num_pairformer_layers:
                continue
        module_state[key] = value
    return module_state


def _trunk_feats() -> dict[str, torch.Tensor]:
    atoms = 64
    tokens = 8
    msa_rows = 3
    atom_to_token = torch.zeros(1, atoms, tokens)
    atom_to_token[0, torch.arange(atoms), torch.arange(atoms) % tokens] = 1.0
    ref_element = torch.zeros(1, atoms, 128)
    ref_element[0, torch.arange(atoms), torch.arange(atoms) % 128] = 1.0
    chars = torch.zeros(1, atoms, 4, 64)
    for index in range(4):
        chars[0, torch.arange(atoms), index, (torch.arange(atoms) + index) % 64] = 1.0
    res_type = torch.zeros(1, tokens, 33)
    res_type[0, torch.arange(tokens), torch.arange(tokens) % 33] = 1.0
    profile = torch.zeros(1, tokens, 33)
    profile[0, torch.arange(tokens), (torch.arange(tokens) + 3) % 33] = 1.0
    contact_conditioning = torch.zeros(1, tokens, tokens, 5)
    for i in range(tokens):
        for j in range(tokens):
            contact_conditioning[0, i, j, 2 + ((i + j) % 3)] = 1.0
    contact_conditioning[:, 0, :, 0] = 1.0
    contact_conditioning[:, -1, :, 1] = 1.0

    feats = {
        "ref_pos": torch.linspace(-0.3, 0.3, steps=atoms * 3).reshape(
            1, atoms, 3
        ),
        "atom_pad_mask": torch.ones(1, atoms, dtype=torch.float32),
        "ref_space_uid": (torch.arange(atoms) // 8).reshape(1, atoms),
        "ref_charge": torch.linspace(-0.5, 0.5, steps=atoms).reshape(1, atoms),
        "ref_element": ref_element,
        "ref_atom_name_chars": chars,
        "atom_to_token": atom_to_token,
        "res_type": res_type,
        "profile": profile,
        "deletion_mean": torch.linspace(0.0, 1.0, steps=tokens).reshape(1, tokens),
        "method_feature": (torch.arange(tokens) % 12).reshape(1, tokens),
        "modified": (torch.arange(tokens) % 2).reshape(1, tokens),
        "cyclic_period": torch.zeros(1, tokens),
        "mol_type": (torch.arange(tokens) % 4).reshape(1, tokens),
        "asym_id": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        "residue_index": torch.arange(tokens).reshape(1, tokens),
        "entity_id": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        "token_index": torch.arange(tokens).reshape(1, tokens),
        "sym_id": torch.zeros(1, tokens, dtype=torch.long),
        "token_bonds": torch.eye(tokens).reshape(1, tokens, tokens, 1),
        "type_bonds": (
            torch.arange(tokens * tokens).reshape(1, tokens, tokens) % 7
        ),
        "contact_conditioning": contact_conditioning,
        "contact_threshold": torch.linspace(4.0, 20.0, steps=tokens * tokens).reshape(
            1, tokens, tokens
        ),
        "msa": (torch.arange(msa_rows * tokens).reshape(1, msa_rows, tokens) % 33),
        "has_deletion": torch.zeros(1, msa_rows, tokens),
        "deletion_value": torch.linspace(0.0, 1.0, steps=msa_rows * tokens).reshape(
            1, msa_rows, tokens
        ),
        "msa_paired": torch.ones(1, msa_rows, tokens),
        "msa_mask": torch.ones(1, msa_rows, tokens),
        "token_pad_mask": torch.ones(1, tokens, dtype=torch.float32),
    }
    feats["atom_pad_mask"][:, -3:] = 0.0
    feats["msa_mask"][:, -1, -1] = 0.0
    feats["token_pad_mask"][:, -1:] = 0.0
    return feats


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
