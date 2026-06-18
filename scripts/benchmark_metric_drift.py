"""JAX-vs-PyTorch drift of the auxiliary metric heads on REAL data.

End-to-end: run the JAX trunk + diffusion sampler on real 1UBQ_A features to
get SHARED real inputs (s_inputs, s, z, x_pred), then feed IDENTICAL numpy
inputs to both the JAX metric heads and the PyTorch reference heads. The drift
therefore isolates the JAX-port error of each head (not sampling stochasticity).

CPU ONLY. Run with: JAX_PLATFORMS=cpu uv run python scripts/benchmark_metric_drift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOLTZ_SRC = REPO.parents[0] / "boltz/src"
CHECKPOINT = REPO.parents[0] / "boltz/.cache/boltz/boltz2_conf.ckpt"
FEATS_NPZ = REPO / "outputs/real_features/1UBQ_A.npz"
OUT_JSON = REPO / "outputs/metric_drift.json"

# Real boltz2_conf checkpoint architecture.
NUM_MSA_LAYERS = 4
NUM_PAIRFORMER_LAYERS = 64
NUM_TOKEN_LAYERS = 24
NUM_CONF_PF_LAYERS = 8
TOKEN_S, TOKEN_Z = 384, 128
DISTO_BINS = 64

RECYCLING_STEPS = 3
NUM_SAMPLING_STEPS = 50
SEED = 0

# Real confidence config (from checkpoint hyper_parameters; see parity test).
CONFIDENCE_MODEL_ARGS = {
    "use_gaussian": False,
    "num_dist_bins": 64,
    "max_dist": 22,
    "use_miniformer": False,
    "no_trunk_feats": False,
    "add_s_to_z_prod": True,
    "add_s_input_to_s": True,
    "use_s_diffusion": False,
    "add_z_input_to_z": True,
    "pairformer_args": {
        "num_blocks": NUM_CONF_PF_LAYERS,
        "num_heads": 16,
        "dropout": 0.0,
    },
    "confidence_args": {
        "num_plddt_bins": 50,
        "num_pde_bins": 64,
        "num_pae_bins": 64,
        "relative_confidence": "none",
        "use_separate_heads": True,
    },
}


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-8)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "max_rel": float((diff / denom).max()) if diff.size else 0.0,
    }


def main() -> None:
    import jax
    import jax.numpy as jnp
    import torch

    torch.set_grad_enabled(False)

    from boltz_jax.bridge.confidence_mapping import map_confidence_module_state_dict
    from boltz_jax.bridge.native import load_features_npz
    from boltz_jax.bridge.torch_checkpoint import load_checkpoint_state_dict
    from boltz_jax.bridge.torch_mapping import (
        map_bfactor_state_dict,
        map_boltz2_graph_state_dict,
        map_distogram_state_dict,
    )
    from boltz_jax.models.heads.bfactor import bfactor_forward
    from boltz_jax.models.heads.confidence import confidence_module_forward
    from boltz_jax.models.heads.distogram import distogram_forward
    from boltz_jax.models.trunk_blocks.trunk import boltz2_sample_forward, boltz2_trunk_forward

    assert CHECKPOINT.exists(), CHECKPOINT
    assert FEATS_NPZ.exists(), FEATS_NPZ

    print("[1/6] loading checkpoint + features ...")
    state = load_checkpoint_state_dict(CHECKPOINT)
    feats = load_features_npz(FEATS_NPZ)  # jnp dict

    graph_params = map_boltz2_graph_state_dict(
        state,
        num_msa_layers=NUM_MSA_LAYERS,
        num_pairformer_layers=NUM_PAIRFORMER_LAYERS,
        num_token_layers=NUM_TOKEN_LAYERS,
        token_transformer_heads=16,
    )

    # ---- which feats keys the confidence head needs; real vs synthesized ----
    required = [
        "token_to_rep_atom", "token_pad_mask", "atom_pad_mask", "atom_to_token",
        "mol_type", "asym_id", "entity_id", "sym_id", "residue_index",
        "token_index", "token_bonds", "type_bonds", "contact_conditioning",
        "contact_threshold", "frames_idx",
    ]
    real_keys = [k for k in required if k in feats]
    synthesized: dict[str, str] = {}
    missing = [k for k in required if k not in feats]
    assert not missing, f"missing genuinely-required feats: {missing}"
    print(f"      confidence feats: {len(real_keys)}/{len(required)} REAL, "
          f"{len(synthesized)} synthesized")

    # ---- [2] JAX trunk: real s_inputs, s, z (recycling_steps=3) ----
    print("[2/6] JAX trunk (recycling_steps=3) ...")
    trunk = boltz2_trunk_forward(
        graph_params["trunk"], feats, recycling_steps=RECYCLING_STEPS
    )
    s_inputs = np.asarray(trunk["s_inputs"])
    s = np.asarray(trunk["s"])
    z = np.asarray(trunk["z"])

    # ---- [3] JAX sampler: real x_pred, fixed injected noise (seed 0) ----
    print(f"[3/6] JAX sample ({NUM_SAMPLING_STEPS} steps, seed {SEED}, scan, "
          "no augmentation) ...")
    atom_shape = (feats["atom_pad_mask"].shape[0], feats["atom_pad_mask"].shape[1], 3)
    nkey = jax.random.PRNGKey(SEED)
    nkey, ik = jax.random.split(nkey)
    init_noise = jax.random.normal(ik, atom_shape, dtype=jnp.float32)
    step_noises = jax.random.normal(
        nkey, (NUM_SAMPLING_STEPS, *atom_shape), dtype=jnp.float32
    )
    sample = boltz2_sample_forward(
        graph_params,
        feats,
        jax.random.PRNGKey(SEED),
        recycling_steps=RECYCLING_STEPS,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        token_layers=NUM_TOKEN_LAYERS,
        multiplicity=1,
        augmentation=False,
        use_scan=True,
        init_noise=init_noise,
        step_noises=step_noises,
    )
    x_pred = np.asarray(sample["sample_atom_coords"])
    print(f"      x_pred {x_pred.shape}  s {s.shape}  z {z.shape}")

    # ---- shared numpy inputs ----
    feats_np = {k: np.asarray(v) for k, v in feats.items()}

    results: dict = {}

    # ======================= DISTOGRAM =======================
    print("[4/6] distogram head ...")
    disto_jax = np.asarray(
        distogram_forward(map_distogram_state_dict(state, "distogram_module"),
                          jnp.asarray(z), num_bins=DISTO_BINS)
    )
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import (
        BFactorModule,
        DistogramModule,
    )

    dmod = DistogramModule(TOKEN_Z, DISTO_BINS).eval()
    dmod.load_state_dict({
        k.removeprefix("distogram_module."): v
        for k, v in state.items() if k.startswith("distogram_module.")
    })
    disto_torch = dmod(torch.from_numpy(z)).numpy()
    d = _stats(disto_jax, disto_torch)
    argmax_agree = float(
        (disto_jax.argmax(-1) == disto_torch.argmax(-1)).mean()
    )
    d["argmax_bin_agreement_pct"] = 100.0 * argmax_agree
    results["distogram_logits"] = d

    # distogram logits used by confidence (pred_distogram_logits) = torch output
    pred_distogram_logits = disto_torch.astype(np.float32)

    # ======================= BFACTOR =======================
    print("[5/6] bfactor head ...")
    bf_jax = np.asarray(
        bfactor_forward(map_bfactor_state_dict(state, "bfactor_module"),
                        jnp.asarray(s))
    )
    token_s_bf = int(state["bfactor_module.bfactor.weight"].shape[1])
    num_bins_bf = int(state["bfactor_module.bfactor.weight"].shape[0])
    bmod = BFactorModule(token_s_bf, num_bins_bf).eval()
    bmod.load_state_dict({
        k.removeprefix("bfactor_module."): v
        for k, v in state.items() if k.startswith("bfactor_module.")
    })
    bf_torch = bmod(torch.from_numpy(s)).numpy()
    results["bfactor_logits"] = _stats(bf_jax, bf_torch)

    # ======================= CONFIDENCE =======================
    print("[6/6] confidence head ...")
    from boltz.model.modules.confidencev2 import ConfidenceModule

    cmod = ConfidenceModule(
        token_s=TOKEN_S, token_z=TOKEN_Z, token_level_confidence=True,
        bond_type_feature=True, fix_sym_check=False, cyclic_pos_enc=False,
        conditioning_cutoff_min=4.0, conditioning_cutoff_max=20.0,
        **CONFIDENCE_MODEL_ARGS,
    ).eval()
    cstate = {}
    for key, value in state.items():
        if not key.startswith("confidence_module."):
            continue
        local = key.removeprefix("confidence_module.")
        if local.startswith("pairformer_stack.layers."):
            if int(local.split(".")[2]) >= NUM_CONF_PF_LAYERS:
                continue
        cstate[local] = value
    miss, _ = cmod.load_state_dict(cstate, strict=False)
    assert not [m for m in miss if not m.endswith("boundaries")], miss

    cparams = map_confidence_module_state_dict(
        state, "confidence_module", num_pairformer_layers=NUM_CONF_PF_LAYERS
    )

    # Build torch feats from the raw npz to preserve original int64 dtypes
    # (jnp downcasts int64->int32 by default, which breaks torch one_hot).
    raw_npz = np.load(FEATS_NPZ)
    feats_torch = {k: torch.from_numpy(raw_npz[k]) for k in raw_npz.files}
    conf_torch = cmod(
        s_inputs=torch.from_numpy(s_inputs), s=torch.from_numpy(s),
        z=torch.from_numpy(z), x_pred=torch.from_numpy(x_pred),
        feats=feats_torch,
        pred_distogram_logits=torch.from_numpy(pred_distogram_logits),
        multiplicity=1,
    )
    conf_jax = confidence_module_forward(
        cparams, s_inputs=jnp.asarray(s_inputs), s=jnp.asarray(s),
        z=jnp.asarray(z), x_pred=jnp.asarray(x_pred),
        feats={k: jnp.asarray(v) for k, v in feats_np.items()},
        pred_distogram_logits=jnp.asarray(pred_distogram_logits), multiplicity=1,
    )

    array_keys = [
        "plddt_logits", "pde_logits", "pae_logits", "resolved_logits",
        "plddt", "pde", "pae",
    ]
    scalar_keys = [
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
        "ptm", "iptm", "ligand_iptm", "protein_iptm",
    ]
    conf_res: dict = {}
    scalars_side: dict = {}
    for k in array_keys + scalar_keys:
        if k not in conf_torch:
            continue
        tj = np.asarray(conf_jax[k])
        tt = conf_torch[k].detach().numpy()
        conf_res[k] = _stats(tj, tt)
        if k in scalar_keys:
            scalars_side[k] = {"jax": float(tj.reshape(-1)[0]),
                               "torch": float(tt.reshape(-1)[0])}
    # pair_chains_iptm nested dict
    pci_j, pci_t = conf_jax["pair_chains_iptm"], conf_torch["pair_chains_iptm"]
    pci_diff = 0.0
    for c1 in pci_t:
        for c2 in pci_t[c1]:
            pci_diff = max(pci_diff, float(np.abs(
                np.asarray(pci_j[c1][c2]) - pci_t[c1][c2].detach().numpy()
            ).max()))
    conf_res["pair_chains_iptm"] = {"max_abs": pci_diff}
    results["confidence"] = conf_res

    report = {
        "config": {
            "checkpoint": str(CHECKPOINT),
            "feats": str(FEATS_NPZ),
            "recycling_steps": RECYCLING_STEPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "seed": SEED,
            "num_msa_layers": NUM_MSA_LAYERS,
            "num_pairformer_layers": NUM_PAIRFORMER_LAYERS,
            "num_token_layers": NUM_TOKEN_LAYERS,
            "num_conf_pairformer_layers": NUM_CONF_PF_LAYERS,
            "confidence_model_args": CONFIDENCE_MODEL_ARGS,
            "pred_distogram_logits_source": "torch DistogramModule output",
            "platform": "cpu",
        },
        "feats_provenance": {"real": real_keys, "synthesized": synthesized},
        "drift": results,
        "scalar_metrics_side_by_side": scalars_side,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    # ---- print table ----
    def row(name, st):
        print(f"  {name:24s} max={st['max_abs']:.3e}  mean={st['mean_abs']:.3e}  "
              f"maxrel={st['max_rel']:.3e}")

    print("\n========== METRIC DRIFT (JAX vs PyTorch, identical inputs) ==========")
    print("DISTOGRAM:")
    row("distogram_logits", results["distogram_logits"])
    _agree = results["distogram_logits"]["argmax_bin_agreement_pct"]
    print(f"  argmax bin agreement: {_agree:.2f}%")
    print("BFACTOR:")
    row("bfactor_logits", results["bfactor_logits"])
    print("CONFIDENCE arrays:")
    for k in array_keys:
        if k in conf_res:
            row(k, conf_res[k])
    print("CONFIDENCE scalars (JAX | torch):")
    for k, v in scalars_side.items():
        print(f"  {k:24s} jax={v['jax']:.6f}   torch={v['torch']:.6f}   "
              f"|d|={abs(v['jax']-v['torch']):.3e}")
    print(f"  pair_chains_iptm max_abs={conf_res['pair_chains_iptm']['max_abs']:.3e}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
