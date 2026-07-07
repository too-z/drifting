"""Parity: LR schedule vs optax fixture; AdamW + clip + EMA trajectory vs
optax fixture.

    pytest tests/parity/test_optim_schedule.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pt.utils.model_builder import create_learning_rate_fn  # noqa: E402

FIXDIR = REPO / "tests" / "parity" / "fixtures"

LR_CASES = {
    "const_2e4_w5000_t100000": dict(learning_rate=2e-4, warmup_steps=5000,
                                    total_steps=100000, lr_schedule="const"),
    "const_4e4_w10000_t200000": dict(learning_rate=4e-4, warmup_steps=10000,
                                     total_steps=200000, lr_schedule="const"),
    "cos_4e4_w10000_t200000": dict(learning_rate=4e-4, warmup_steps=10000,
                                   total_steps=200000, lr_schedule="cosine"),
    "cos_4e3_w4000_t200000": dict(learning_rate=4e-3, warmup_steps=4000,
                                  total_steps=200000, lr_schedule="cos"),
}


@pytest.mark.skipif(not (FIXDIR / "lr_schedule.npz").exists(), reason="lr fixture missing")
@pytest.mark.parametrize("case", sorted(LR_CASES))
def test_lr_schedule_matches_optax(case):
    fx = dict(np.load(FIXDIR / "lr_schedule.npz"))
    fn = create_learning_rate_fn(**LR_CASES[case])
    got = np.array([fn(int(s)) for s in fx["steps"]], dtype=np.float64)
    # optax evaluates schedules in fp32 (with ~1e-10 absolute cancellation
    # noise in the warmup); the torch port is exact float64. Differences are
    # bounded by fp32 rounding of the *endpoint* magnitudes, hence atol.
    np.testing.assert_allclose(got, fx[case], rtol=1e-5, atol=1e-8, err_msg=case)


@pytest.mark.skipif(not (FIXDIR / "optim.npz").exists(), reason="optim fixture missing")
def test_adamw_clip_ema_trajectory():
    fx = dict(np.load(FIXDIR / "optim.npz"))
    w = torch.nn.Parameter(torch.from_numpy(fx["init_w"].copy()))
    b = torch.nn.Parameter(torch.from_numpy(fx["init_b"].copy()))
    opt = torch.optim.AdamW([w, b], lr=fx["lr_values"][0], betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)
    ema_w = w.detach().clone()
    decay = float(fx["ema_decay"])

    n_steps = fx["grads_w"].shape[0]
    for t in range(n_steps):
        opt.zero_grad(set_to_none=True)
        w.grad = torch.from_numpy(fx["grads_w"][t].copy())
        b.grad = torch.from_numpy(fx["grads_b"][t].copy())
        g_norm = torch.nn.utils.clip_grad_norm_([w, b], 2.0)
        np.testing.assert_allclose(float(g_norm), fx["gnorms"][t], rtol=1e-5,
                                   err_msg=f"pre-clip g_norm step {t}")
        for group in opt.param_groups:
            group["lr"] = float(fx["lr_values"][t])
        opt.step()
        with torch.no_grad():
            ema_w.mul_(decay).add_(w, alpha=1 - decay)

        np.testing.assert_allclose(w.detach().numpy(), fx["traj_w"][t],
                                   rtol=1e-5, atol=1e-7, err_msg=f"w step {t}")
        np.testing.assert_allclose(b.detach().numpy(), fx["traj_b"][t],
                                   rtol=1e-5, atol=1e-7, err_msg=f"b step {t}")
        np.testing.assert_allclose(ema_w.numpy(), fx["traj_ema_w"][t],
                                   rtol=1e-5, atol=1e-7, err_msg=f"ema step {t}")
