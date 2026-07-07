"""Torch port of dataset/vae.py: SD-VAE (ft-MSE) encode/decode.

Same weights as the Flax "pcuenq/sd-vae-ft-mse-flax" (that repo is the Flax
conversion of stabilityai/sd-vae-ft-mse). Layout delta vs the JAX version:
torch latents are BCHW (the Flax pipeline used BHWC latents); the whole torch
pipeline is BCHW so no transposes are needed anywhere.
"""

from functools import partial

import torch

SCALE = 0.18215

# Module-level cache so repeated calls don't reload the VAE.
_vae_cache = {}


def vae_enc_decode(device=None, dtype=torch.float32):
    """
    Returns:
        encode_fn(images, generator=None): (B, 3, H, W) in [-1, 1] ->
            sampled latents (B, 4, H/8, W/8), scaled by 0.18215.
        decode_fn(latents): (B, 4, h, w) scaled latents -> (B, 3, H, W) pixels.

    Both run under no_grad on `device` (default: cuda if available).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    cache_key = ("vae_enc_decode", str(device), str(dtype))
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def _encode_fn(images, generator=None, vae=vae):
        dist = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
        return dist.sample(generator=generator) * SCALE

    @torch.no_grad()
    def _decode_fn(latents, vae=vae):
        return vae.decode(latents.to(device=device, dtype=dtype) / SCALE).sample

    result = (_encode_fn, _decode_fn)
    _vae_cache[cache_key] = result
    return result
