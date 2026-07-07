"""Numerical parity: torch DitGen vs JAX fixtures (tests/parity/fixtures/gen_<name>.npz).

Runs in the torch env (no jax needed):
    pytest tests/parity/test_generator.py -v

Parametrized over every generator fixture present; each needs its converted
artifact at $HF_ROOT/models/gen/torch/<name>
(python -m pt.convert.convert_generator --name <name>).
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

from pt.models.generator import LightningDiTBlock, build_generator_from_config  # noqa: E402

ROOT = Path(os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache")))
FIXDIR = REPO / "tests" / "parity" / "fixtures"

NAMES = sorted(
    p.stem[len("gen_"):] for p in FIXDIR.glob("gen_*.npz")
) or ["ablation"]


def art_dir(name):
    return ROOT / "models" / "gen" / "torch" / name


def require(name):
    if not (art_dir(name) / "model.safetensors").exists():
        pytest.skip(f"converted artifact missing: {art_dir(name)}")
    fix = FIXDIR / f"gen_{name}.npz"
    if not fix.exists():
        pytest.skip(f"fixture missing: {fix}")
    return dict(np.load(fix))


def load_artifact(name):
    from safetensors.torch import load_file

    metadata = json.loads((art_dir(name) / "metadata.json").read_text())
    state = load_file(str(art_dir(name) / "model.safetensors"))
    return state, metadata


def build_model(name, tag):
    state, metadata = load_artifact(name)
    cfg = dict(metadata["model_config"])
    if tag == "fp32":
        cfg.update(use_bf16=False, attn_fp32=True)
    model = build_generator_from_config(cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(metadata["model_config"])


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def assert_fp32_close(got, ref, rel_tol=1e-6):
    """fp32 parity: identical math up to op-reordering noise, scaled to |ref|."""
    r = rel_l2(got, ref)
    assert r <= rel_tol, (
        f"rel_l2={r:.3e} (max abs {np.abs(got - ref).max():.3e}, "
        f"ref scale {np.abs(ref).max():.3f})"
    )


def assert_bf16_close(got, ref_bf16, ref_fp32=None):
    """bf16 parity: both frameworks are independent bf16 roundings of the same
    fp32 function. Two criteria:
      1. equidistance — torch-bf16 is no farther from the fp32 truth than
         jax-bf16 is (x1.2 slack);
      2. mutual distance — the two bf16 outputs are close relative to their
         common intrinsic deviation from fp32 (floor 3e-2 for the cases where
         that deviation is tiny). With attn_fp32=False the intrinsic bf16
         deviation is large (rel ~0.3 for latent_B_sota) and the mutual
         distance scales with it.
    """
    r = rel_l2(got, ref_bf16)
    c = cosine(got, ref_bf16)
    if ref_fp32 is not None:
        d_torch = rel_l2(got, ref_fp32)
        d_jax = rel_l2(ref_bf16, ref_fp32)
        assert d_torch <= d_jax * 1.2 + 1e-6, (
            f"torch-bf16 drifts from fp32 truth ({d_torch:.3e}) "
            f"more than jax-bf16 does ({d_jax:.3e})"
        )
        limit = max(3e-2, 0.25 * d_jax)
        assert r <= limit and c >= 0.995, (
            f"rel_l2={r:.3e} (limit {limit:.3e}) cos={c:.6f} d_jax={d_jax:.3e}"
        )
    else:
        assert r <= 3e-2 and c >= 0.999, f"rel_l2={r:.3e} cos={c:.6f}"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_cond_parity(name, tag):
    fx = require(name)
    model, _ = build_model(name, tag)
    with torch.no_grad():
        cond = model.c_cfg_noise_to_cond(
            torch.from_numpy(fx["c"]).long(),
            float(fx["cfg_scale"]),
            torch.from_numpy(fx["noise_labels"]).long(),
        )
    got = cond.float().numpy()
    if tag == "fp32":
        assert_fp32_close(got, fx["cond_fp32"])
    else:
        assert_bf16_close(got, fx["cond_bf16"], fx["cond_fp32"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_block_parity(name, tag):
    fx = require(name)
    state, metadata = load_artifact(name)
    cfg = dict(metadata["model_config"])
    use_bf16 = cfg["use_bf16"] if tag == "bf16" else False
    cd = torch.bfloat16 if use_bf16 else torch.float32
    blk = LightningDiTBlock(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        use_qknorm=cfg["use_qknorm"],
        use_swiglu=cfg["use_swiglu"],
        use_rmsnorm=cfg["use_rmsnorm"],
        cond_dim=cfg["cond_dim"],
        use_rope=cfg["use_rope"],
        attn_fp32=cfg["attn_fp32"] if tag == "bf16" else True,
        compute_dtype=cd,
    )
    prefix = "model.blocks.0."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    blk.load_state_dict(sub, strict=True)
    blk.eval()

    with torch.no_grad():
        y = blk(
            torch.from_numpy(fx["block_x"]).to(cd),
            torch.from_numpy(fx["block_c"]).to(cd),
        )
    got = y.float().numpy()
    if tag == "fp32":
        assert_fp32_close(got, fx["block_out_fp32"])
    else:
        assert_bf16_close(got, fx["block_out_bf16"], fx["block_out_fp32"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_full_forward_parity(name, tag):
    fx = require(name)
    model, cfg = build_model(name, tag)
    cd = torch.bfloat16 if cfg.get("use_bf16") and tag == "bf16" else torch.float32
    x = torch.from_numpy(fx["x"])  # BHWC, fp32 pre-cast
    with torch.no_grad():
        cond = model.c_cfg_noise_to_cond(
            torch.from_numpy(fx["c"]).long(),
            float(fx["cfg_scale"]),
            torch.from_numpy(fx["noise_labels"]).long(),
        )
        samples = model.generate_image(x.to(cd), cond)
    got = samples.float().numpy()
    if tag == "fp32":
        assert_fp32_close(got, fx["samples_fp32"], rel_tol=1e-5)
    else:
        assert_bf16_close(got, fx["samples_bf16"], fx["samples_fp32"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tag", ["fp32", "bf16"])
def test_forward_with_injected_noise(name, tag):
    """End-to-end forward() plumbing: injected noise must reproduce the
    composed cond+generate_image path, with BCHW output."""
    fx = require(name)
    model, _ = build_model(name, tag)
    with torch.no_grad():
        out = model(
            torch.from_numpy(fx["c"]).long(),
            cfg_scale=float(fx["cfg_scale"]),
            noise=torch.from_numpy(fx["x"]),
            noise_labels=torch.from_numpy(fx["noise_labels"]).long(),
        )
    got = out["samples"].permute(0, 2, 3, 1).float().numpy()  # BCHW -> BHWC
    if tag == "fp32":
        assert_fp32_close(got, fx["samples_fp32"], rel_tol=1e-5)
    else:
        assert_bf16_close(got, fx["samples_bf16"], fx["samples_fp32"])
