"""Latent cache dataset and cache builder — torch port of dataset/latent.py.

Cache format is byte-identical to the JAX builder's: one .pt per image holding
{"moments": float32 (H, W, C) latent * 0.18215, "moments_flip": same for the
h-flipped image}, mirroring the ImageNet directory layout. Caches built by
either backend are interchangeable.

Deltas vs the JAX builder (documented, behavior-preserving):
  * each rank writes only its own slice of the global batch (the JAX version
    allgathered and every host redundantly wrote all files — same file set);
  * both moments and moments_flip are sampled with the same per-batch seed,
    matching the JAX reuse of one rng key for both encodes.

Multi-GPU: torchrun --nproc_per_node=N -m pt.dataset.latent ...
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


@dataclass(frozen=True)
class _CacheWriteItem:
    output_path: str
    moments: np.ndarray
    moments_flip: np.ndarray


def _write_cache_file(item: _CacheWriteItem) -> None:
    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    torch.save(
        {
            "moments": item.moments,
            "moments_flip": item.moments_flip,
        },
        tmp_path,
    )
    os.replace(tmp_path, output_path)


class LatentDataset(datasets.DatasetFolder):
    """ImageFolder-style dataset for cached latent `.pt` files."""

    def __init__(self, root: str):
        super().__init__(root=root, loader=str, extensions=(".pt",))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        moments = data["moments"] if torch.rand(1) < 0.5 else data["moments_flip"]
        return np.asarray(moments), target


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop used before encoding to latent."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


class OriginalImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns class/file relative path for cache writing."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        rel_path = os.path.join(*path.split(os.path.sep)[-2:])
        return sample, target, rel_path


def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    save_workers: int = 0,
) -> None:
    """Encode ImageNet train/val images and write latent cache files."""
    from pt.dataset.vae import vae_enc_decode
    from pt.utils import dist_util

    dist_util.init_distributed()
    dev = dist_util.device()
    ws = dist_util.world_size()
    rk = dist_util.rank()

    if rk == 0:
        Path(target_path, "train").mkdir(parents=True, exist_ok=True)
        Path(target_path, "val").mkdir(parents=True, exist_ok=True)
    dist_util.barrier("latent cache target dirs ready")

    encode_fn, _ = vae_enc_decode(device=dev)

    save_pool = None
    save_futures: list[Future] = []
    if save_workers > 0:
        save_pool = ProcessPoolExecutor(max_workers=save_workers, mp_context=mp.get_context("spawn"))

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    global_batch_size = local_batch_size * ws
    process_slice_start = rk * local_batch_size
    process_slice_end = process_slice_start + local_batch_size

    for split in ("train", "val"):
        dataset = OriginalImageFolder(os.path.join(data_path, split), transform=transform)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": global_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = torch.utils.data.DataLoader(**loader_kwargs)

        for step, (samples, _, rel_paths) in tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"cache:{split}:rank{rk}",
            disable=rk != 0,
        ):
            n_valid_global = samples.shape[0]
            rel_paths = list(rel_paths)
            if n_valid_global != global_batch_size:
                pad = global_batch_size - n_valid_global
                samples = torch.cat(
                    [samples, torch.zeros((pad,) + samples.shape[1:], dtype=samples.dtype)], dim=0
                )
                rel_paths.extend([""] * pad)

            local_samples = samples[process_slice_start:process_slice_end].to(dev)
            # One seed per (step, rank); reused for the flipped encode, matching
            # the JAX builder's single rng key per batch.
            from pt.utils.rng import make_generator

            moments = encode_fn(
                local_samples, generator=make_generator(dev, "latent-cache", step, rk)
            )
            moments_flip = encode_fn(
                torch.flip(local_samples, dims=[3]),
                generator=make_generator(dev, "latent-cache", step, rk),
            )
            # BCHW -> per-image HWC float32, the cache's on-disk layout.
            moments = moments.permute(0, 2, 3, 1).float().cpu().numpy()
            moments_flip = moments_flip.permute(0, 2, 3, 1).float().cpu().numpy()

            write_items = []
            local_rel_paths = rel_paths[process_slice_start:process_slice_end]
            for i, rel_path in enumerate(local_rel_paths):
                if not rel_path:
                    continue
                output_path = str(Path(target_path, split, rel_path).with_suffix(".pt"))
                write_items.append(
                    _CacheWriteItem(
                        output_path=output_path,
                        moments=np.asarray(moments[i]),
                        moments_flip=np.asarray(moments_flip[i]),
                    )
                )
            if save_pool is None:
                for item in write_items:
                    _write_cache_file(item)
            else:
                save_futures.extend(save_pool.submit(_write_cache_file, item) for item in write_items)

        dist_util.barrier(f"latent cache split {split} encoded")

    if save_pool is not None:
        for future in tqdm(save_futures, desc="cache:flush", disable=rk != 0):
            future.result()
        save_pool.shutdown()

    dist_util.barrier("latent cache files flushed")


def build_cache_from_args(args: argparse.Namespace) -> None:
    create_cached_dataset(
        local_batch_size=int(args.local_batch_size),
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        save_workers=int(args.save_workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ImageNet latent cache files for release generator configs.")
    parser.add_argument("--data-path", default=IMAGENET_PATH, help="ImageNet root containing train/ and val/.")
    parser.add_argument("--target-path", default=IMAGENET_CACHE_PATH, help="Output cache root for latent .pt files.")
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=128,
        help="Per-rank cache batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory for the cache build.")
    parser.add_argument(
        "--save-workers",
        type=int,
        default=0,
        help="Optional process count for asynchronous latent file writes on each rank.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    build_cache_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
