"""MAE training loop — torch port of train_mae.py.

Structure mirrors the JAX file function-by-function. Key mappings:
  * jax.jit train_step + value_and_grad  -> eager forward/backward
  * optax.global_norm + clip_by_global_norm(applied pre-optimizer)
      -> clip_grad_norm_ (returns the pre-clip norm, logged as g_norm)
  * flax TrainState(ema_params, ema_decay) -> (module, optimizer, ema dict, step)
  * named rng streams via fold_in(step) -> pt.utils.rng.make_generator
  * pad_and_merge global eval accounting -> per-rank pad + all-reduced
    weighted sums (rank-major trim reproduces the merged-mask semantics)
"""

from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pt.dataset.dataset import epoch0_sampler, infinite_sampler
from pt.models.mae_model import MAEResNet
from pt.utils import dist_util
from pt.utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from pt.utils.init_util import load_params_for_init
from pt.utils.logging import is_rank_zero, log_for_0
from pt.utils.misc import load_config
from pt.utils.model_builder import build_model_dict, set_lr
from pt.utils.rng import make_generator
from utils import env


def input_dict(batch):
    """Convert preprocessed batch dict to model forward kwargs."""
    return {"x": batch["images"], "labels": batch["labels"]}


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_step(
    module,
    fwd_module,
    optimizer,
    ema,
    step,
    batch,
    *,
    seed,
    forward_dict,
    learning_rate_fn,
    preprocess_fn,
    max_grad_norm=2.0,
    ema_decay,
    device,
):
    """One MAE optimization step. `module` is the raw model (for clipping/EMA);
    `fwd_module` is the DDP-wrapped one (or the same object single-process)."""
    rank = dist_util.rank()
    batch = preprocess_fn(batch, generator=make_generator(device, seed, "vae", step, rank))
    batch = _to_device(batch, device)
    forward_kwargs = input_dict(batch)

    g = make_generator(device, seed, "masking", step, rank)
    loss_all, metric = fwd_module(
        forward_kwargs["x"], forward_kwargs["labels"], generator=g, **forward_dict
    )
    loss = loss_all.mean()

    set_lr(optimizer, learning_rate_fn(step))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    g_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        for name, p in module.named_parameters():
            ema[name].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

    metric = {k: v.detach().float().mean() for k, v in metric.items()}
    metric["loss"] = loss.detach()
    metric["lr"] = learning_rate_fn(step)
    metric["g_norm"] = g_norm.detach()
    return dist_util.all_reduce_mean(metric)


@torch.no_grad()
def eval_loop(
    eval_module,
    eval_loader,
    preprocess_fn,
    *,
    eval_samples=5000,
    forward_dict=None,
    rng_seed=0,
    device,
):
    """Evaluate MAE. Reproduces the JAX pad_and_merge accounting: each rank
    pads short batches to the first-seen batch size, global sample count is
    goal_bsz * world_size per iteration, the final batch is trimmed rank-major
    (the merged array was rank-major), and the weighted mean is computed from
    all-reduced (sum, mask_sum)."""
    forward_dict = forward_dict or {}
    dist_util.barrier("eval loop started")
    eval_iter = epoch0_sampler(eval_loader)
    ws = dist_util.world_size()
    rank = dist_util.rank()

    sums = {}
    mask_total = 0.0
    n_total = 0.0
    goal_bsz = 0
    n_samples = 0
    for i, batch in enumerate(eval_iter):
        image, label = batch
        if i == 0:
            goal_bsz = image.shape[0]
        b = image.shape[0]
        mask = torch.zeros(goal_bsz)
        mask[:b] = 1.0
        if b < goal_bsz:
            pad = goal_bsz - b
            image = np.concatenate([image, np.zeros((pad,) + image.shape[1:], image.dtype)], 0)
            label = np.concatenate([label, np.zeros((pad,), label.dtype)], 0)

        pbatch = preprocess_fn((image, label), generator=make_generator(device, rng_seed, "eval-vae", i))
        pbatch = _to_device(pbatch, device)

        g = make_generator(device, rng_seed, "eval", i)
        loss, metric = eval_module(
            pbatch["images"], pbatch["labels"], generator=g, **dict(forward_dict)
        )
        metric = dict(metric)
        metric["loss"] = loss

        # Global trim, rank-major like the merged mask in JAX.
        global_bsz = goal_bsz * ws
        if n_samples + global_bsz > eval_samples:
            keep_global = eval_samples - n_samples
            keep_local = max(0, min(goal_bsz, keep_global - rank * goal_bsz))
            mask[keep_local:] = 0.0
        n_samples += global_bsz

        mask_dev = mask.to(device)
        for k, v in metric.items():
            v = v.detach().float()
            sums[k] = sums.get(k, 0.0) + float((v * mask_dev).sum())
        mask_total += float(mask.sum())
        n_total += goal_bsz

        if n_samples >= eval_samples:
            break

    reduced = dist_util.all_reduce_mean({**{k: v for k, v in sums.items()},
                                         "__mask": mask_total, "__n": n_total})
    # all_reduce_mean averages; multiply back by ws to get global sums.
    out = {}
    denom = reduced["__mask"] * ws + 1e-8 * reduced["__n"] * ws
    for k in sums:
        out[k] = reduced[k] * ws / denom
    dist_util.barrier("eval loop finished")
    return out


def train_mae(
    *,
    model,  # MAEResNet instance to train
    optimizer,  # torch.optim.AdamW
    logger,  # logger with log_dict / finish
    eval_loader,  # evaluation dataloader iterator source
    train_loader,  # training dataloader iterator source
    learning_rate_fn,  # callable(step) -> lr
    forward_dict,  # kwargs passed to MAE forward in train
    eval_forward_dict,  # kwargs passed to MAE forward in eval
    preprocess_fn,  # preprocessing function for dataloader batches
    postprocess_fn,  # kept for interface compatibility (unused)
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=2000,
    eval_samples=5000,
    ema_decay=0.999,
    seed=42,
    finetune_last_steps=0,
    warmup_finetune=1000,
    finetune_cls=0.5,
    max_grad_norm=2.0,
    keep_every=500000,
    keep_last=2,
    init_from="",
    workdir="runs",
    model_config=None,
):
    """MAE training loop (torch port of train_mae.py:train_mae)."""
    del postprocess_fn

    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)

    device = dist_util.device()
    module = model.to(device)
    ema = {k: p.detach().clone().float() for k, p in module.named_parameters()}

    step = restore_checkpoint(module, optimizer, ema, workdir=workdir)
    if step == 0 and init_from:
        log_for_0("Initializing MAE params from init_from=%s", init_from)
        sd = load_params_for_init("mae", init_from, hf_cache_dir=env.HF_ROOT)
        module.load_state_dict(sd, strict=True)
        ema = {k: p.detach().clone().float() for k, p in module.named_parameters()}

    if dist_util.world_size() > 1:
        fwd_module = torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[dist_util.local_rank()] if device.type == "cuda" else None
        )
    else:
        fwd_module = module

    # Shadow module for EMA evals (params only; the models have no buffers).
    ema_module = None

    def get_ema_module():
        nonlocal ema_module
        if ema_module is None:
            ema_module = copy.deepcopy(module)
            ema_module.eval().requires_grad_(False)
        ema_module.load_state_dict(ema, strict=True)
        return ema_module

    forward_zeros_dict = copy.deepcopy(dict(forward_dict))
    forward_zeros_dict["mask_ratio_min"] = 0.0
    forward_zeros_dict["mask_ratio_max"] = 0.0

    log_for_0("Starting MAE training loop...")
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    train_iter = infinite_sampler(train_loader, step)
    dist_util.barrier("train loop started")

    start_finetune_step = total_steps - finetune_last_steps
    start_time_all = time.time()
    for step in pbar:
        start_time = time.time()
        logger.set_step(step)

        batch = next(train_iter)
        finish_prepare = time.time()

        cur_dict = dict(copy.deepcopy(dict(forward_dict)))
        if step >= start_finetune_step:
            cur_dict["lambda_cls"] = finetune_cls * min(1.0, (step - start_finetune_step) / max(1, warmup_finetune))

        step_kwargs = dict(
            seed=seed,
            forward_dict=cur_dict,
            learning_rate_fn=learning_rate_fn,
            preprocess_fn=preprocess_fn,
            max_grad_norm=max_grad_norm,
            ema_decay=ema_decay,
            device=device,
        )
        if step == initial_step and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        module.train()
        metrics = train_step(module, fwd_module, optimizer, ema, step, batch, **step_kwargs)

        profile_metrics = {}
        if step == initial_step and torch.cuda.is_available():
            torch.cuda.synchronize()
            profile_metrics["profile/train_step_memory_GB"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )

        finish_train = time.time()
        metrics["kimg"] = (step - initial_step + 1) * batch[0].shape[0] * dist_util.world_size() / 1000.0
        metrics["time/total"] = finish_train - start_time
        metrics["time/prepare"] = finish_prepare - start_time
        metrics["time/train"] = finish_train - finish_prepare
        metrics["time/per_step"] = (finish_train - start_time_all) / (step - initial_step + 1)
        metrics.update(profile_metrics)
        logger.log_dict(metrics)

        step += 1

        if step % eval_per_step == 0:
            module.eval()
            eval_metrics = eval_loop(
                module, eval_loader, preprocess_fn,
                eval_samples=eval_samples, forward_dict=eval_forward_dict,
                rng_seed=seed + 1, device=device,
            )
            logger.log_dict_dir("eval", eval_metrics)
            eval_metrics_ema = eval_loop(
                get_ema_module(), eval_loader, preprocess_fn,
                eval_samples=eval_samples, forward_dict=eval_forward_dict,
                rng_seed=seed + 1, device=device,
            )
            logger.log_dict_dir(f"eval_ema_{ema_decay:g}", eval_metrics_ema)

            eval_metrics_nomask = eval_loop(
                module, eval_loader, preprocess_fn,
                eval_samples=eval_samples, forward_dict=forward_zeros_dict,
                rng_seed=seed + 1, device=device,
            )
            logger.log_dict_dir("eval_nomask", eval_metrics_nomask)
            eval_metrics_nomask_ema = eval_loop(
                get_ema_module(), eval_loader, preprocess_fn,
                eval_samples=eval_samples, forward_dict=forward_zeros_dict,
                rng_seed=seed + 1, device=device,
            )
            logger.log_dict_dir(f"eval_ema_{ema_decay:g}_nomask", eval_metrics_nomask_ema)

        if (step in [total_steps, start_finetune_step]) or (step % save_per_step == 0 and step < start_finetune_step):
            save_checkpoint(module, optimizer, ema, step, ema_decay=ema_decay,
                            keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                ema, step=step, ema_decay=ema_decay, workdir=workdir,
                kind="mae", model_config=model_config,
            )

        if step % 100 == 0:
            dist_util.barrier(f"train step {step} finished")

    dist_util.barrier("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist_util.barrier("train loop cleanup finished")


def main_mae(config, output_dir="runs"):
    """Build MAE model pipeline and launch MAE training."""
    dist_util.init_distributed()
    if "hsdp_dim" in config:
        log_for_0("hsdp_dim=%s is ignored under the torch backend (DDP).", config["hsdp_dim"])
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    model_dict = build_model_dict(config, MAEResNet, workdir=output_dir)
    train_mae(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        model_config=dict(config.model),
        workdir=output_dir,
        **config.train,
    )


def main(args):
    """CLI entrypoint for MAE training."""
    config = load_config(args.config)
    main_mae(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to MAE config.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    args = parser.parse_args()
    args.output_dir = args.workdir
    main(args)
