"""Parity + gradient checks for pt/drift_loss.py vs the JAX fixture.

    pytest tests/parity/test_drift_loss.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pt.drift_loss import drift_loss  # noqa: E402

FIX = REPO / "tests" / "parity" / "fixtures" / "drift_loss.npz"

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="drift_loss fixture missing")


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX))


def test_drift_loss_parity(fx):
    loss, info = drift_loss(
        gen=torch.from_numpy(fx["gen"]),
        fixed_pos=torch.from_numpy(fx["pos"]),
        fixed_neg=torch.from_numpy(fx["neg"]),
        weight_gen=torch.from_numpy(fx["w_gen"]),
        weight_pos=torch.from_numpy(fx["w_pos"]),
        weight_neg=torch.from_numpy(fx["w_neg"]),
        R_list=(0.2, 0.05, 0.02),
        distributed_stats=False,
    )
    np.testing.assert_allclose(loss.numpy(), fx["loss"], rtol=1e-5, atol=1e-6)
    for k in ("scale", "loss_0.2", "loss_0.05", "loss_0.02"):
        np.testing.assert_allclose(
            float(info[k]), float(fx[f"info_{k}"]), rtol=1e-5, atol=1e-7, err_msg=k
        )


def test_drift_loss_parity_no_neg(fx):
    loss, info = drift_loss(
        gen=torch.from_numpy(fx["gen"]),
        fixed_pos=torch.from_numpy(fx["pos"]),
        R_list=(0.2, 0.05, 0.02),
        distributed_stats=False,
    )
    np.testing.assert_allclose(loss.numpy(), fx["loss_noneg"], rtol=1e-5, atol=1e-6)
    for k in ("scale", "loss_0.2", "loss_0.05", "loss_0.02"):
        np.testing.assert_allclose(
            float(info[k]), float(fx[f"info_noneg_{k}"]), rtol=1e-5, atol=1e-7, err_msg=k
        )


def test_gradient_flows_only_through_gen(fx):
    """The goal branch is stop-gradiented: only `gen` gets grads, and the
    gradient of mean((gen/s - goal)^2) wrt gen equals 2*(gen/s - goal)/(s*numel_per_sample)."""
    gen = torch.from_numpy(fx["gen"]).clone().requires_grad_(True)
    pos = torch.from_numpy(fx["pos"]).clone().requires_grad_(True)

    loss, _ = drift_loss(gen=gen, fixed_pos=pos, R_list=(0.2, 0.05, 0.02),
                         distributed_stats=False)
    loss.mean().backward()

    assert gen.grad is not None and torch.isfinite(gen.grad).all()
    assert gen.grad.abs().sum() > 0
    # fixed_pos only feeds the no-grad goal — zero (or absent) gradient.
    assert pos.grad is None or pos.grad.abs().max() == 0
