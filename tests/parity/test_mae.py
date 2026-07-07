"""Numerical parity: torch MAEResNet vs JAX fixtures (mae_<name>.npz).

    pytest tests/parity/test_mae.py -v

Requires the converted artifact at $HF_ROOT/models/mae/torch/<name>
(python -m pt.convert.convert_mae --name <name>).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pt.models.mae_model import MAEResNet, mae_from_metadata, patch_input  # noqa: E402

ROOT = Path(os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache")))
FIXDIR = REPO / "tests" / "parity" / "fixtures"

NAMES = sorted(p.stem[len("mae_"):] for p in FIXDIR.glob("mae_*.npz")) or ["mae_latent_256"]

ACT_KWARGS = dict(patch_mean_size=[2, 4], patch_std_size=[2, 4],
                  use_std=True, use_mean=True, every_k_block=2)


def art_dir(name):
    return ROOT / "models" / "mae" / "torch" / name


def require(name):
    if not (art_dir(name) / "model.safetensors").exists():
        pytest.skip(f"converted artifact missing: {art_dir(name)}")
    fix = FIXDIR / f"mae_{name}.npz"
    if not fix.exists():
        pytest.skip(f"fixture missing: {fix}")
    return dict(np.load(fix))


def build_model(name, tag):
    from safetensors.torch import load_file

    metadata = json.loads((art_dir(name) / "metadata.json").read_text())
    metadata["model_config"] = dict(metadata["model_config"])
    metadata["model_config"]["use_bf16"] = tag == "bf16"
    model = mae_from_metadata(metadata)
    state = load_file(str(art_dir(name) / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def assert_close(tag, got, ref_tag, ref_fp32, ctx="", fp32_tol=1e-5):
    """fp32: tight relative tolerance. bf16: scale-aware (see test_generator)."""
    if tag == "fp32":
        r = rel_l2(got, ref_fp32)
        assert r <= fp32_tol, f"{ctx}: rel_l2={r:.3e}"
    else:
        d_torch = rel_l2(got, ref_fp32)
        d_jax = rel_l2(ref_tag, ref_fp32)
        r = rel_l2(got, ref_tag)
        limit = max(3e-2, 0.5 * d_jax)
        assert d_torch <= d_jax * 1.3 + 1e-4, (
            f"{ctx}: torch-bf16 {d_torch:.3e} vs jax-bf16 {d_jax:.3e} from fp32 truth"
        )
        assert r <= limit, f"{ctx}: mutual rel_l2={r:.3e} (limit {limit:.3e})"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_activations_parity(name, tag):
    fx = require(name)
    model = build_model(name, tag)
    x = torch.from_numpy(fx["x"]).permute(0, 3, 1, 2)  # BHWC -> BCHW

    with torch.no_grad():
        acts = model.get_activations(x, **ACT_KWARGS)

    keys = sorted(k[len(f"act_{tag}_"):] for k in fx if k.startswith(f"act_{tag}_"))
    assert sorted(acts) == keys, (
        f"activation key mismatch:\n torch-only: {sorted(set(acts) - set(keys))}"
        f"\n jax-only: {sorted(set(keys) - set(acts))}"
    )
    for k in keys:
        got = acts[k].float().numpy()
        ref = fx[f"act_{tag}_{k}"]
        assert got.shape == ref.shape, f"{k}: {got.shape} vs {ref.shape}"
        assert_close(tag, got, ref, fx[f"act_fp32_{k}"], ctx=k)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_encoder_decoder_parity(name, tag):
    """Direct encoder+decoder+fc pass with the injected mask (loss path)."""
    fx = require(name)
    model = build_model(name, tag)
    cd = model.compute_dtype

    x = torch.from_numpy(fx["x"]).permute(0, 3, 1, 2).to(cd)
    mask = torch.from_numpy(fx["mask"]).permute(0, 3, 1, 2).to(cd)  # (B,1,H',W')
    xp = patch_input(x, model.input_patch_size)
    x_in = xp * (1.0 - mask)

    with torch.no_grad():
        feats = model.encoder(x_in)
        pooled = feats["layer4"].mean(dim=(2, 3))
        logits = model.fc(pooled)
        recon = model.decoder(feats)

    assert_close(tag, logits.float().numpy(), fx[f"logits_{tag}"], fx["logits_fp32"],
                 ctx="logits")
    got_recon = recon.permute(0, 2, 3, 1).float().numpy()  # -> BHWC to match fixture
    assert_close(tag, got_recon, fx[f"recon_{tag}"], fx["recon_fp32"], ctx="recon")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_forward_loss_parity(name, tag):
    """forward() with injected mask reproduces the JAX loss composition."""
    fx = require(name)
    model = build_model(name, tag)

    x = torch.from_numpy(fx["x"]).permute(0, 3, 1, 2)
    mask = torch.from_numpy(fx["mask"]).permute(0, 3, 1, 2)
    labels = torch.from_numpy(fx["labels"]).long()

    with torch.no_grad():
        loss, metrics = model(x, labels, lambda_cls=float(fx["lambda_cls"]), mask=mask)

    for key, got in [("loss", loss), ("cls_loss", metrics["cls_loss"]),
                     ("recon_loss", metrics["recon_loss"])]:
        assert_close(tag, got.float().numpy(), fx[f"{key}_{tag}"], fx[f"{key}_fp32"],
                     ctx=key, fp32_tol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_gradients_flow(name):
    """Sanity: forward() is differentiable end to end (training path)."""
    fx = require(name)
    model = build_model(name, "fp32")
    model.train()
    x = torch.from_numpy(fx["x"]).permute(0, 3, 1, 2)
    labels = torch.from_numpy(fx["labels"]).long()
    g = torch.Generator().manual_seed(0)
    loss, _ = model(x, labels, lambda_cls=0.1, generator=g)
    loss.mean().backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for _ in model.parameters())
    assert n_grads == n_params, f"only {n_grads}/{n_params} params got gradients"
