"""Parity test: boltz2_predict equals calling the head functions directly.

Loads the torch-free native conf bundle (which now includes the confidence
head) and the real 1UBQ_A features, then asserts the wrapper's per-head outputs
exactly equal the individual head functions called on the same trunk/sample.
This proves the wrapper does not alter numerics. Marked slow (loads the full
~1.9GiB checkpoint bundle).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boltz_jax.bridge.native import load_features_npz, load_params
from boltz_jax.models.heads.bfactor import bfactor_forward
from boltz_jax.models.heads.confidence import confidence_module_forward
from boltz_jax.models.heads.distogram import distogram_forward
from boltz_jax.models.predict import boltz2_predict
from boltz_jax.models.trunk_blocks.trunk import (
    boltz2_sample_forward,
    boltz2_trunk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "outputs/native_weights/boltz2_conf"
FEATURES = ROOT / "outputs/real_features/1UBQ_A.npz"

RECYCLING = 0
STEPS = 4
SEED = 0


@pytest.fixture(scope="module")
def params():
    if not (WEIGHTS.with_suffix(".safetensors").exists() or
            WEIGHTS.with_suffix(".npz").exists()):
        pytest.skip(f"native weights not found: {WEIGHTS}")
    return load_params(WEIGHTS)


@pytest.fixture(scope="module")
def feats():
    if not FEATURES.exists():
        pytest.skip(f"features not found: {FEATURES}")
    return load_features_npz(FEATURES)


@pytest.mark.slow
def test_predict_matches_direct_head_calls(params, feats) -> None:
    key = jax.random.PRNGKey(SEED)

    out = boltz2_predict(
        params,
        feats,
        key,
        recycling_steps=RECYCLING,
        num_sampling_steps=STEPS,
        augmentation=False,
        run_confidence=True,
        run_distogram=True,
        run_bfactor=True,
    )

    # Reproduce the structure sampler with the identical key/args.
    sample = boltz2_sample_forward(
        params,
        feats,
        key,
        recycling_steps=RECYCLING,
        num_sampling_steps=STEPS,
        augmentation=False,
    )["sample_atom_coords"]
    np.testing.assert_array_equal(
        np.asarray(out["sample_atom_coords"]), np.asarray(sample)
    )

    trunk = boltz2_trunk_forward(params["trunk"], feats, recycling_steps=RECYCLING)
    pdistogram = distogram_forward(params, trunk["z"])
    np.testing.assert_array_equal(
        np.asarray(out["pdistogram"]), np.asarray(pdistogram)
    )

    pbfactor = bfactor_forward(params, trunk["s"])
    np.testing.assert_array_equal(
        np.asarray(out["pbfactor"]), np.asarray(pbfactor)
    )

    conf = confidence_module_forward(
        params["confidence"],
        s_inputs=trunk["s_inputs"],
        s=trunk["s"],
        z=trunk["z"],
        x_pred=sample,
        feats=feats,
        pred_distogram_logits=pdistogram[:, :, :, 0],
        multiplicity=1,
    )
    for k in ("plddt", "pae", "pde", "ptm", "iptm", "complex_plddt"):
        np.testing.assert_array_equal(
            np.asarray(out[k]), np.asarray(conf[k]), err_msg=f"mismatch {k}"
        )

    coords = np.asarray(out["sample_atom_coords"])
    assert np.isfinite(coords).all()
    assert jnp.all(jnp.isfinite(out["ptm"]))
