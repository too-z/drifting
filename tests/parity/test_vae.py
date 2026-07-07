"""Parity: torch AutoencoderKL (stabilityai/sd-vae-ft-mse) vs the Flax VAE.

    pytest tests/parity/test_vae.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FIX = REPO / "tests" / "parity" / "fixtures" / "vae.npz"

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="vae fixture missing")


@pytest.fixture(scope="module")
def vae():
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX))


def to_bchw_like(ref, arr_bchw):
    """Fixture arrays from flax may be BHWC; compare in whichever layout matches."""
    if ref.shape == tuple(arr_bchw.shape):
        return arr_bchw
    return arr_bchw.transpose(0, 2, 3, 1)


def test_encode_moments(fx, vae):
    with torch.no_grad():
        dist = vae.encode(torch.from_numpy(fx["img"])).latent_dist
    mean = to_bchw_like(fx["enc_mean"], dist.mean.numpy())
    std = to_bchw_like(fx["enc_std"], dist.std.numpy())
    assert np.abs(mean - fx["enc_mean"]).max() <= 2e-4, np.abs(mean - fx["enc_mean"]).max()
    assert np.abs(std - fx["enc_std"]).max() <= 2e-4


def test_decode(fx, vae):
    lat = torch.from_numpy(fx["lat"]).permute(0, 3, 1, 2)  # BHWC -> BCHW
    with torch.no_grad():
        dec = vae.decode(lat / 0.18215).sample
    got = to_bchw_like(fx["decoded"], dec.numpy())
    assert np.abs(got - fx["decoded"]).max() <= 2e-4, np.abs(got - fx["decoded"]).max()


def test_wrapper_roundtrip(fx):
    """pt/dataset/vae.py wrapper: shapes, scale, determinism of the mean path."""
    from pt.dataset.vae import vae_enc_decode

    encode_fn, decode_fn = vae_enc_decode(device="cpu")
    img = torch.from_numpy(fx["img"])
    g = torch.Generator().manual_seed(0)
    lat = encode_fn(img, generator=g)
    assert lat.shape == (2, 4, 8, 8)
    pix = decode_fn(lat)
    assert pix.shape == img.shape
    # same seed -> same sampled latent
    g2 = torch.Generator().manual_seed(0)
    lat2 = encode_fn(img, generator=g2)
    assert torch.equal(lat, lat2)
