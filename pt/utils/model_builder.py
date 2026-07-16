"""Torch port of utils/model_builder.py.

Deltas: optax.adamw -> torch.optim.AdamW (single param group — the optax
version applied weight decay to ALL params including norms/biases, so no
no-decay groups here); the optax schedule becomes a closed-form python
function applied by direct param_group assignment in the train loops.
"""

import math
from pathlib import Path

import torch

from pt.dataset.dataset import create_imagenet_split
from pt.utils import dist_util
from pt.utils.logging import WandbLogger
from pt.utils.misc import EasyDict


def create_learning_rate_fn(
    learning_rate,
    warmup_steps,
    total_steps,
    lr_schedule="const",
):
    """Closed-form port of the optax schedule:
    linear_schedule(1e-6 -> lr, warmup) joined with constant or
    cosine_decay_schedule(alpha=1e-6) evaluated at (step - warmup)."""
    init_value = 1e-6
    alpha = 1e-6

    def lr_at(step):
        step = float(step)
        if step < warmup_steps:
            return init_value + (learning_rate - init_value) * (step / warmup_steps)
        t = step - warmup_steps
        if lr_schedule == "const":
            return float(learning_rate)
        if lr_schedule in ("cosine", "cos"):
            d = max(total_steps - warmup_steps, 1)
            frac = min(t / d, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
            return float(learning_rate) * ((1 - alpha) * cosine + alpha)
        raise NotImplementedError(lr_schedule)

    return lr_at


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_model_dict(config, model_class, *, workdir: str = "runs"):
    """Build model, datasets, optimizer, and logger from config."""
    print("Building model...")
    model = model_class(
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    print("Building dataset...")
    batch_size_per_rank = config.dataset.batch_size // dist_util.world_size()
    dataset_type = str(config.dataset.get("type", "imagenet")).lower()

    if dataset_type == "tabular":
        from pt.dataset.tabular import create_tabular_split
        ds_kwargs = dict(config.dataset.get("kwargs", {}))
        train_loader, preprocess_fn, postprocess_fn = create_tabular_split(
            batch_size = batch_size_per_rank,
            split="train",
            **ds_kwargs,
        )
        eval_loader, _, _ = create_tabular_split(
            batch_size = config.dataset.eval_batch_size // dist_util.world_size(),
            split="val",
            **ds_kwargs,
        )
        dataset_name = str(config.dataset.get("name", "tabular"))
    else:
        resolution = int(config.dataset.resolution)
        use_aug = bool(config.dataset.get("use_aug", False))
        use_latent = bool(config.dataset.get("use_latent", False))
        use_cache = bool(config.dataset.get("use_cache", False))
    
        train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
            resolution=resolution,
            use_aug=use_aug,
            use_latent=use_latent,
            use_cache=use_cache,
            batch_size=batch_size_per_rank,
            split="train",
            **config.dataset.kwargs,
        )
    
        eval_loader, _, _ = create_imagenet_split(
            resolution=resolution,
            use_aug=use_aug,
            use_latent=use_latent,
            use_cache=use_cache,
            batch_size=config.dataset.eval_batch_size // dist_util.world_size(),
            split="val",
            **config.dataset.kwargs,
        )
        dataset_name = f"imagenet{resolution}"

    learning_rate_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_fn(0),
        betas=(config.optimizer.adam_b1, config.optimizer.adam_b2),
        eps=1e-8,
        weight_decay=config.optimizer.get("weight_decay", 0.0),
    )

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.get("use_wandb", config.get("use_wandb", True)))
    if "use_wandb" in w_cfg:
        del w_cfg["use_wandb"]
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=str(output_root),
        **w_cfg,
    )

    return EasyDict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        eval_loader=eval_loader,
        train_loader=train_loader,
        dataset_name=dataset_name,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        train=config.train,
        learning_rate_fn=learning_rate_fn,
        feature=config.get("feature", {}),
    )
