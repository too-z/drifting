"""FID/IS/precision-recall evaluation. Torch port of utils/fid_util.py.

Single-process plain PyTorch: the JAX version's pmap/multihost plumbing
(process_allgather, ddp sharding, sync_global_devices, padding batches to a
static ``devices * batch`` shape) is dropped; everything else — the resize ->
Inception -> statistics pipeline, mask handling, metric math, and the
``evaluate_fid`` call signature — mirrors the JAX file 1:1.

Semantic deltas vs utils/fid_util.py:
- single-process: no multihost allgather; ``_compute_stats`` iterates plain
  chunks of ``batch_size`` instead of padding to ``local_device_count * 200``
  (per-sample results are identical, eval-mode BatchNorm).
- ``rng_eval`` is an integer base seed (default 0) instead of a PRNGKey; batch
  ``i`` receives ``rng=rng_eval + i`` (see the ``gen_func`` contract below).
- ``logger`` may be ``None`` to skip logging.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from utils.env import IMAGENET_FID_NPZ, IMAGENET_PR_NPZ

from pt.utils.torch_fid import resize
from pt.utils.torch_fid.fid import compute_frechet_distance
from pt.utils.torch_fid.precision_recall import compute_precision_recall


INCEPTION_NET = None
_DATASET_STATS = {
    "imagenet256": IMAGENET_FID_NPZ,
}
_PR_REF_PATH = IMAGENET_PR_NPZ


def _canonical_dataset_name(name: str) -> str:
    n = name.lower()
    if "imagenet256" in n:
        return "imagenet256"
    raise ValueError(f"Only ImageNet is supported now, got: {name}")


def _build_torch_inception():
    """Create the (cached) torch Inception network used for FID/IS features."""
    from pt.utils.torch_fid.inception import InceptionV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3().to(device).eval()
    return {"model": model, "device": device}


def _to_uint8(samples):
    """Convert float ``[0, 1]`` samples to ``uint8 [0, 255]``."""
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)
    return (samples * 255).clip(0, 255).astype(np.uint8)


def _to_numpy(x):
    """Convert a torch tensor (or array-like) to a numpy array on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _compute_stats(samples_uint8: np.ndarray, num_samples: int, *, compute_logits: bool, compute_features: bool, masks=None, batch_size: int = 200):
    """Run Inception over generated samples and compute dataset statistics.

    Args:
        samples_uint8: generated images as `NHWC` or `NCHW` uint8 arrays.
        num_samples: target number of valid samples after removing padding.
        compute_logits: whether to keep classifier logits for IS.
        compute_features: whether to keep raw pool features for PR.
        masks: optional validity mask with shape `(N,)`; padded samples should be `0`.
        batch_size: Inception forward batch size (JAX used 200 per device).
    """
    global INCEPTION_NET
    if INCEPTION_NET is None:
        INCEPTION_NET = _build_torch_inception()
    model = INCEPTION_NET["model"]
    device = INCEPTION_NET["device"]

    if samples_uint8.shape[-1] != 3:
        samples_uint8 = samples_uint8.transpose(0, 2, 3, 1)

    if masks is None:
        masks = np.ones((len(samples_uint8),), dtype=np.float32)

    feats_list = []
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(samples_uint8), batch_size):
            # Inception consumes the resize helper's BCHW output in [-1, 1];
            # the resize helper consumes BCHW float in [0, 255] (on CPU).
            x = torch.from_numpy(samples_uint8[i : i + batch_size].astype(np.float32).transpose(0, 3, 1, 2))
            x = resize.forward(x)
            pooled, logits = model(x.to(device))
            feats_list.append(pooled.cpu().numpy())
            if compute_logits and logits is not None:
                logits_list.append(logits.cpu().numpy())

    all_feats = np.concatenate(feats_list, axis=0)

    all_masks = np.asarray(masks).reshape(-1)
    valid_len = min(all_feats.shape[0], all_masks.shape[0])
    all_feats = all_feats[:valid_len]
    all_masks = all_masks[:valid_len]
    all_feats = all_feats[all_masks > 0.5][:num_samples]

    feats64 = all_feats.astype(np.float64)
    out = {
        "mu": np.mean(feats64, axis=0),
        "sigma": np.cov(feats64, rowvar=False),
    }

    if compute_features:
        out["features"] = all_feats

    if compute_logits and logits_list:
        all_logits = np.concatenate(logits_list, axis=0)
        all_logits = all_logits[:valid_len]
        all_logits = all_logits[all_masks > 0.5][:num_samples]
        out["logits"] = all_logits

    return out


def _softmax_np(x, axis=-1):
    """Numpy softmax matching ``jax.nn.softmax`` (computed in the input dtype)."""
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _compute_inception_score(logits, splits=10):
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]
    probs = _softmax_np(logits, axis=-1)
    probs = np.asarray(probs, dtype=np.float64)

    n = probs.shape[0]
    split_size = n // splits
    probs = probs[: split_size * splits]
    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def _load_ref_stats(dataset_name: str):
    canon = _canonical_dataset_name(dataset_name)
    path = _DATASET_STATS[canon]
    data = np.load(path)
    if "ref_mu" in data:
        return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return {"mu": data["mu"], "sigma": data["sigma"]}


def _epoch0_sampler(it):
    """Yield one deterministic epoch (`sampler.set_epoch(0)` when available).

    Torch port of dataset.dataset.epoch0_sampler; batches are yielded as-is
    (torch tensors or numpy arrays), not converted.
    """
    sampler = getattr(it, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    for batch in it:
        yield batch


def _pad_and_merge(batch, goal_bsz):
    """Pad every leaf of ``batch`` to ``goal_bsz`` along dim 0 with zeros.

    Single-process port of utils.hsdp_util.pad_and_merge (no device sharding).

    Returns:
        (padded_batch, mask) where ``mask`` is a float32 ``(goal_bsz,)`` array
        with 1 for real samples and 0 for padding.
    """
    leaves = list(batch) if isinstance(batch, (tuple, list)) else [batch]
    if not leaves:
        raise ValueError("Data is empty")
    current_len = leaves[0].shape[0]
    pad_len = goal_bsz - current_len
    assert pad_len >= 0, f"goal_bsz: {goal_bsz} is less than current_len: {current_len}"
    mask = np.concatenate([
        np.ones(current_len, dtype=np.float32),
        np.zeros(pad_len, dtype=np.float32),
    ])
    if pad_len == 0:
        return batch, mask

    def _pad(x):
        if isinstance(x, torch.Tensor):
            zeros = torch.zeros((pad_len, *x.shape[1:]), dtype=x.dtype, device=x.device)
            return torch.cat([x, zeros], dim=0)
        x = np.asarray(x)
        return np.concatenate([x, np.zeros((pad_len, *x.shape[1:]), dtype=x.dtype)], axis=0)

    padded = [_pad(x) for x in leaves]
    if isinstance(batch, (tuple, list)):
        return type(batch)(padded), mask
    return padded[0], mask


def evaluate_fid(
    dataset_name,
    gen_func,
    gen_params,
    eval_loader,
    logger,
    num_samples=5000,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_isc=True,
    eval_fid=True,
    rng_eval=None,
):
    """Generate samples, run Inception statistics, and log release metrics.

    Torch port of utils.fid_util.evaluate_fid with the same call signature.
    Single-process: the JAX version's multihost allgather / device sharding /
    sync_global_devices are dropped.

    Args:
        dataset_name: Dataset identifier used to select reference statistics.
            Only ImageNet-256 is supported in this release.
        gen_func: Generation callable. For every eval batch it is invoked as
            ``gen_func(batch, **gen_params, rng=<int>)`` where ``batch`` is the
            (zero-padded) ``(images, labels)`` pair from ``eval_loader`` and
            ``rng`` is a deterministic per-batch integer seed
            (``rng_eval + batch_index``). It must return generated samples for
            the whole padded batch in ``BCHW`` or ``BHWC`` format with values
            in ``[0, 1]`` (torch tensor or numpy array); padded entries are
            discarded via the validity mask.
        gen_params: Keyword arguments forwarded into ``gen_func`` for every eval
            batch. This typically contains the EMA model and a fixed CFG scale.
        eval_loader: Iterable of ``(images, labels)`` batches. The labels are
            used to drive conditional generation; the image tensors are ignored.
            If it exposes ``.sampler.set_epoch``, epoch 0 is selected for
            determinism.
        logger: Logger that receives scalar metrics via ``log_dict`` and a
            64-image preview grid via ``log_image``. May be ``None`` to skip
            logging.
        num_samples: Number of valid generated samples to score after padding is
            removed.
        log_folder: Top-level metric namespace written into the logger.
        log_prefix: Per-run metric prefix inside ``log_folder``.
        eval_prc_recall: Whether to compute precision/recall in addition to FID.
        eval_isc: Whether to compute Inception Score.
        eval_fid: Whether to compute FID.
        rng_eval: Integer base seed for deterministic evaluation sampling
            (default 0). Replaces the JAX PRNGKey; batch ``i`` gets seed
            ``rng_eval + i``.

    Returns:
        Dict[str, float] containing the computed metrics. Keys may include
        ``fid``, ``isc_mean``, ``isc_std``, ``precision``, ``recall``, and
        ``fid_time`` depending on which evaluations are enabled.
    """
    if rng_eval is None:
        rng_eval = 0
    rng_eval = int(rng_eval)

    start = time.time()

    eval_iter = _epoch0_sampler(eval_loader)
    all_samples = []
    all_masks = []
    cur = 0
    goal_bsz = None
    for i, batch in enumerate(eval_iter):
        if goal_bsz is None:
            leaves = list(batch) if isinstance(batch, (tuple, list)) else [batch]
            goal_bsz = leaves[0].shape[0]
        # Pad the final batch so gen_func always sees a static shape.
        batch, mask = _pad_and_merge(batch, goal_bsz)
        gen_samples = gen_func(batch, **gen_params, rng=rng_eval + i)
        gen_samples = _to_numpy(gen_samples)
        all_samples.append(_to_uint8(gen_samples))
        all_masks.append(np.asarray(mask, dtype=np.float32))
        cur += gen_samples.shape[0]
        if cur >= num_samples:
            break

    samples = np.concatenate(all_samples, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    stats = _compute_stats(samples, num_samples, compute_logits=eval_isc, compute_features=eval_prc_recall, masks=masks)
    ref = _load_ref_stats(dataset_name)

    metrics: Dict[str, float] = {}
    if eval_fid:
        metrics["fid"] = float(compute_frechet_distance(ref["mu"], stats["mu"], ref["sigma"], stats["sigma"]))
    if eval_isc and "logits" in stats:
        mean, std = _compute_inception_score(stats["logits"])
        metrics["isc_mean"] = mean
        metrics["isc_std"] = std
    if eval_prc_recall and "features" in stats:
        ref_images = np.load(_PR_REF_PATH)["arr_0"].astype(np.uint8)
        ref_stats = _compute_stats(ref_images, 10000, compute_logits=False, compute_features=True)
        precision, recall = compute_precision_recall(ref_stats["features"], stats["features"], k=3)
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)

    metrics["fid_time"] = float(time.time() - start)
    if logger is not None:
        logger.log_dict({f"{log_folder}/{log_prefix}_{k}": v for k, v in metrics.items()})
        logger.log_image(f"{log_folder}/{log_prefix}_viz", samples[:64])
    return metrics
