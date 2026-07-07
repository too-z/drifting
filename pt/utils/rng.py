"""Deterministic seed derivation — torch analog of jax.random.fold_in plus the
named streams of utils/misc.py:prepare_rng.

Determinism contract (mirrors the JAX release): no RNG state is checkpointed;
every stochastic step re-derives its generator from (seed, stream, step, rank),
so resume reproduces the same draws.
"""

import hashlib

import torch


def fold(*parts) -> int:
    """Hash arbitrary ints/strings into a stable 63-bit seed.

    Unlike builtins.hash, stable across processes and runs (blake2b, no
    PYTHONHASHSEED salting).
    """
    payload = "\x1f".join(str(p) for p in parts).encode()
    h = hashlib.blake2b(payload, digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def make_generator(device, *parts) -> torch.Generator:
    """torch.Generator seeded by fold(*parts), e.g.
    make_generator(dev, seed, "noise", step, rank)."""
    g = torch.Generator(device=device)
    g.manual_seed(fold(*parts))
    return g
