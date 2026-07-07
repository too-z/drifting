"""ImageNet dataset pipeline — torch port of dataset/dataset.py.

Layout: everything is BCHW (the JAX pipeline used BHWC at model runtime).
The latent cache stores per-image HWC latents (interchangeable with the JAX
builder); the cache preprocess permutes them to BCHW.

`preprocess_fn(batch, generator=None)` -> {"images": BCHW, "labels": int64}
`postprocess_fn(images)` -> BCHW pixels in [0, 1]
"""

from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from pt.dataset.latent import LatentDataset
from pt.dataset.vae import vae_enc_decode
from pt.utils import dist_util
from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop image with ADM preprocessing style."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _build_transforms(resolution: int, use_aug: bool, split: str):
    if use_aug and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _build_imagenet_dataset(*, resolution: int, use_aug: bool, use_cache: bool, split: str):
    if use_cache:
        return LatentDataset(root=os.path.join(IMAGENET_CACHE_PATH, split))

    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=os.path.join(IMAGENET_PATH, split), transform=transform)


def worker_init_fn(worker_id: int, rank: int) -> None:
    """Initialize deterministic RNG for each data-loader worker."""
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    local: bool | None = None,
):
    """Create ImageNet split loader and preprocess/postprocess functions.

    Returns `(loader, preprocess_fn, postprocess_fn)` — see module docstring
    for the batch contracts. batch_size is per-rank.
    """
    del local
    ds = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
    )

    rank = dist_util.rank()
    sampler = DistributedSampler(
        ds, num_replicas=dist_util.world_size(), rank=rank, shuffle=True
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    if use_latent or use_cache:
        encode_fn, decode_fn = vae_enc_decode()
        if use_cache:
            def preprocess_fn(batch, generator=None):
                del generator
                cached_latent, label = batch
                # cache stores HWC latents -> BCHW
                images = _as_tensor(cached_latent).permute(0, 3, 1, 2)
                return {"images": images, "labels": _as_tensor(label).long()}
        else:
            def preprocess_fn(batch, generator=None):
                image, label = batch
                images = encode_fn(_as_tensor(image), generator=generator)
                return {"images": images, "labels": _as_tensor(label).long()}

        def postprocess_fn(images):
            # BCHW latent -> decode -> BCHW pixel in [0,1]
            return torch.clamp((decode_fn(images) + 1) / 2, 0, 1)

        return loader, preprocess_fn, postprocess_fn

    def preprocess_fn(batch, generator=None):
        del generator
        image, label = batch
        return {"images": _as_tensor(image), "labels": _as_tensor(label).long()}

    def postprocess_fn(images):
        # BCHW pixel in [-1,1] -> [0,1]
        return torch.clamp((images + 1) / 2, 0, 1)

    return loader, preprocess_fn, postprocess_fn


def get_postprocess_fn(*, use_aug: bool = False, use_latent: bool = False, use_cache: bool = False, has_clip: bool = True):
    """Return postprocess function for generated samples by dataset mode flags."""
    if use_latent or use_cache:
        _, decode_fn = vae_enc_decode()

        def postprocess(images):
            out = (decode_fn(images) + 1) / 2
            return torch.clamp(out, 0, 1) if has_clip else out

        return postprocess

    if use_aug or (not use_latent and not use_cache):

        def postprocess(images):
            out = (images + 1) / 2
            return torch.clamp(out, 0, 1) if has_clip else out

        return postprocess

    raise ValueError("Unsupported dataset flags.")


def infinite_sampler(it, start_step: int = 0):
    """Yield `(image, label)` numpy batches forever, resuming at `start_step`."""
    step_per_epoch = len(it)
    epoch_idx = start_step // step_per_epoch
    it.sampler.set_epoch(epoch_idx)
    skip_batches = start_step % step_per_epoch
    while True:
        for i, batch in enumerate(it):
            if skip_batches > 0 and i < skip_batches:
                continue
            image, label = batch
            yield (image.numpy(), label.numpy())
        skip_batches = 0
        epoch_idx += 1
        it.sampler.set_epoch(epoch_idx)


def epoch0_sampler(it):
    """Yield one deterministic epoch (`sampler.set_epoch(0)`)."""
    it.sampler.set_epoch(0)
    for batch in it:
        image, label = batch
        yield (image.numpy(), label.numpy())
