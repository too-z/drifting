"""Parity: torch ConvNextFeatures vs Flax get_activations fixture.

    pytest tests/parity/test_convnext.py -v

The resize (jax.image.resize antialias vs F.interpolate antialias) is the one
knowingly-inexact op here, so tolerances are looser than the conv-stack tests
(plan: rel-L2 <= 1e-2 per key).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FIX = REPO / "tests" / "parity" / "fixtures" / "convnext.npz"

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="convnext fixture missing")


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX))


@pytest.fixture(scope="module")
def model():
    from pt.models.convnext import load_convnext_model

    return load_convnext_model("base")


def test_activations_parity(fx, model):
    x = torch.from_numpy(fx["x"]).permute(0, 3, 1, 2)  # BHWC -> BCHW
    with torch.no_grad():
        feats = model.get_activations(x)

    ref_keys = sorted(k[len("cn_"):] for k in fx if k.startswith("cn_"))
    assert sorted(feats) == ref_keys, (
        f"torch-only: {sorted(set(feats) - set(ref_keys))} "
        f"jax-only: {sorted(set(ref_keys) - set(feats))}"
    )
    worst = {}
    for k in ref_keys:
        got = feats[k].float().numpy()
        ref = fx[f"cn_{k}"]
        assert got.shape == ref.shape, f"{k}: {got.shape} vs {ref.shape}"
        worst[k] = rel_l2(got, ref)
    bad = {k: v for k, v in worst.items() if v > 1e-2}
    assert not bad, f"keys over rel-L2 1e-2: {bad} (all: {worst})"
