"""Generator ("drift") training loop — torch port of train.py.

Mappings from the JAX file:
  * jitted train_step with global-batch shardings -> eager per-rank step under
    DDP; drift_loss(distributed_stats=True) all-reduces the two global-batch
    scalar statistics (scale, per-R force norm) so per-rank math equals the
    JAX global-batch computation.
  * frozen feature model: params-as-data + stop_gradient -> module outside
    DDP with requires_grad_(False); the positives/negatives branch runs under
    torch.no_grad(), the generated branch differentiates through it.
  * memory banks stay host-side NumPy, one pair per rank (JAX: per host).
    Global push rate = push_per_step * world_size — scale push_per_step when
    changing world size (README guidance).
  * merge_data disappears: each rank keeps its local slice.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from pt.dataset.dataset import get_postprocess_fn, infinite_sampler
from pt.drift_loss import drift_loss
from pt.memory_bank import ArrayMemoryBank
from pt.models.mae_model import build_activation_function
from pt.utils import dist_util
from pt.utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from pt.utils.fid_util import evaluate_fid
from pt.utils.init_util import load_params_for_init
from pt.utils.logging import is_rank_zero, log_for_0
from pt.utils.misc import load_config
from pt.utils.model_builder import build_model_dict, set_lr
from pt.utils.rng import fold, make_generator
from utils import env


def sample_cfg(B, *, cfg_min, cfg_max, neg_cfg_pw, no_cfg_frac, generator, device):
    """Per-sample CFG scales — verbatim port of train.py:71-82."""
    frac = torch.rand(B, device=device, generator=generator)
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = torch.exp(
            np.log(cfg_min) + frac * (np.log(cfg_max) - np.log(cfg_min))
        )
    else:
        cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1 / pw)

    frac2 = torch.rand(B, device=device, generator=generator)
    return torch.where(frac2 < no_cfg_frac, torch.ones_like(cfg), cfg)


def train_step(
    module,
    fwd_module,
    optimizer,
    ema,
    step,
    labels,
    samples,
    negative_samples,
    activation_fn,
    *,
    seed,
    learning_rate_fn,
    activation_kwargs,
    loss_kwargs,
    gen_per_label=16,
    cfg_min=1.0,
    cfg_max=4.0,
    neg_cfg_pw=1.0,
    no_cfg_frac=0.0,
    max_grad_norm=2.0,
    ema_decay,
    device,
):
    """One drift-training step on this rank's slice of the global batch.

    Args:
        labels: (B,) int64 on device.
        samples: (B, P, C, H, W) positive bank samples on device.
        negative_samples: (B, N, C, H, W) negative bank samples on device.
    """
    rank = dist_util.rank()
    B = labels.shape[0]

    g_cfg = make_generator(device, seed, "cfg", step, rank)
    cfg = sample_cfg(
        B, cfg_min=cfg_min, cfg_max=cfg_max, neg_cfg_pw=neg_cfg_pw,
        no_cfg_frac=no_cfg_frac, generator=g_cfg, device=device,
    )

    uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, negative_samples.shape[1])
    n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]

    neg_samples_input = rearrange(
        torch.cat([samples, negative_samples], dim=1), "b x ... -> (b x) ..."
    )
    with torch.no_grad():  # stop_gradient branch (frozen features of bank samples)
        sg_features = activation_fn(None, neg_samples_input, **activation_kwargs)
        sg_features = {
            k: rearrange(v, "(b x) ... -> b x ...", x=n_pos + n_uncond)
            for k, v in sg_features.items()
        }

    input_labels = repeat(labels, "b -> (b g)", g=gen_per_label)
    input_cfg = repeat(cfg, "b -> (b g)", g=gen_per_label)
    g_noise = make_generator(device, seed, "noise", step, rank)
    gen_samples = fwd_module(input_labels, cfg_scale=input_cfg, generator=g_noise)["samples"]
    gen_features = activation_fn(None, gen_samples, **activation_kwargs)
    gen_features = {
        k: rearrange(v, "(b g) ... -> b g ...", g=n_gen) for k, v in gen_features.items()
    }

    total_loss = 0.0
    total_info = {}
    for key in sg_features:
        feature_pos = rearrange(sg_features[key][:, :n_pos], "b x f d -> (b f) x d")
        feature_uncond = rearrange(sg_features[key][:, n_pos:], "b x f d -> (b f) x d")
        feature_gen = rearrange(gen_features[key], "b x f d -> (b f) x d")
        Bf = feature_gen.shape[0]
        loss, info = drift_loss(
            gen=feature_gen,
            fixed_pos=feature_pos,
            fixed_neg=feature_uncond,
            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
            weight_neg=repeat(uncond_w, "b -> (b f) k", f=Bf // uncond_w.shape[0], k=n_uncond),
            **loss_kwargs,
        )
        total_loss = total_loss + loss.mean()
        for k2, v2 in info.items():
            total_info[f"{k2}/{key}"] = v2

    set_lr(optimizer, learning_rate_fn(step))
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    g_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        for name, p in module.named_parameters():
            ema[name].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

    metric = {k: v.detach().float().mean() for k, v in total_info.items()}
    metric["loss"] = total_loss.detach()
    metric["g_norm"] = g_norm.detach()
    metric["lr"] = learning_rate_fn(step)
    return dist_util.all_reduce_mean(metric)


def make_gen_step(gen_module, postprocess_fn, device):
    """FID generation callable following the pt.utils.fid_util contract:
    gen_step(batch, rng=<int>, cfg_scale=<float>) -> [0,1] BCHW."""

    @torch.no_grad()
    def gen_step(batch, rng=0, cfg_scale=1.0):
        _, labels = batch
        labels = torch.as_tensor(np.asarray(labels)).long().to(device)
        g = make_generator(device, "eval-noise", int(rng))
        samples = gen_module(labels, cfg_scale=float(cfg_scale), generator=g)["samples"]
        return postprocess_fn(samples)

    return gen_step


def train_gen(
    model,  # DitGen model instance
    optimizer,  # torch.optim.AdamW
    logger,  # logger with log_dict / finish
    eval_loader,  # evaluation dataloader iterator source
    train_loader,  # training dataloader iterator source
    learning_rate_fn,  # callable(step) -> lr
    preprocess_fn,  # preprocessing function for dataloader batches
    postprocess_fn,  # generated sample postprocess function
    dataset_name="imagenet256",
    train_batch_size=0,
    total_steps=100000,
    save_per_step=10000,
    eval_per_step=5000,
    eval_samples=50000,
    activation_fn=None,
    feature_params=None,  # dict of frozen feature modules (torch backend)
    ema_decay=0.999,
    seed=42,
    pos_per_sample=32,
    neg_per_sample=16,
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ),
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2, 4],
        patch_std_size=[2, 4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000,
    keep_last=2,
    init_from="",
    push_per_step=0,
    push_at_resume=3000,
    workdir="runs",
    model_config=None,
):
    """Main training loop (torch port of train.py:train_gen)."""
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)
    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    device = dist_util.device()
    module = model.to(device)
    ema = {k: p.detach().clone().float() for k, p in module.named_parameters()}

    step = restore_checkpoint(module, optimizer, ema, workdir=workdir)
    if step == 0 and init_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        sd = load_params_for_init("gen", init_from, hf_cache_dir=env.HF_ROOT)
        module.load_state_dict(sd, strict=True)
        ema = {k: p.detach().clone().float() for k, p in module.named_parameters()}

    if dist_util.world_size() > 1:
        fwd_module = torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[dist_util.local_rank()] if device.type == "cuda" else None
        )
    else:
        fwd_module = module

    assert activation_fn is not None, "activation_fn must be provided"
    loss_kwargs = dict(loss_kwargs)
    loss_kwargs["R_list"] = tuple(loss_kwargs["R_list"])

    # EMA shadow module for FID eval (params only; no buffers in DitGen).
    ema_module = None

    def get_ema_module():
        nonlocal ema_module
        if ema_module is None:
            import copy

            ema_module = copy.deepcopy(module)
            ema_module.eval().requires_grad_(False)
        ema_module.load_state_dict(ema, strict=True)
        return ema_module

    log_for_0("Starting training loop...")
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    dist_util.barrier("train loop started")
    train_iter = infinite_sampler(train_loader, step)

    print(f"world_size={dist_util.world_size()} rank={dist_util.rank()} device={device}")

    step_kwargs = dict(
        seed=seed,
        learning_rate_fn=learning_rate_fn,
        activation_kwargs=dict(activation_kwargs),
        loss_kwargs=loss_kwargs,
        max_grad_norm=max_grad_norm,
        ema_decay=ema_decay,
        device=device,
        **dict(forward_dict),
    )

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        # fill memory banks; per rank (JAX: per host)
        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            print(f"pushing at resume: {goal}")
        while True:
            batch = next(train_iter)
            processed_batch = preprocess_fn(batch)
            images = processed_batch["images"].detach().cpu().numpy()
            labels_np = processed_batch["labels"].detach().cpu().numpy()
            memory_bank_positive.add(images, labels_np)
            memory_bank_negative.add(images, labels_np * 0)
            n_push += images.shape[0]
            if n_push >= goal:
                break

        bsz_per_rank = train_batch_size // dist_util.world_size()
        assert labels_np.shape[0] >= bsz_per_rank, (
            f"Labels shape {labels_np.shape[0]} < bsz_per_rank {bsz_per_rank}"
        )
        rng_sel = np.random.Generator(np.random.PCG64(fold(seed, "select", step, dist_util.rank())))
        select_indices = rng_sel.choice(labels_np.shape[0], bsz_per_rank, replace=False)
        labels_np = labels_np[select_indices]

        positive_samples = memory_bank_positive.sample(labels_np, n_samples=pos_per_sample)
        negative_samples = memory_bank_negative.sample(labels_np * 0, n_samples=neg_per_sample)

        labels = torch.from_numpy(labels_np).long().to(device)
        positive_samples = torch.from_numpy(positive_samples).to(device)
        negative_samples = torch.from_numpy(negative_samples).to(device)

        process_time = time.time() - start_time

        if step == initial_step and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        module.train()
        metrics = train_step(
            module, fwd_module, optimizer, ema, step,
            labels, positive_samples, negative_samples, activation_fn,
            **step_kwargs,
        )
        if step == initial_step and torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics["profile/train_step_memory_GB"] = torch.cuda.max_memory_allocated() / 1e9

        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["process_time"] = process_time
        global_bsz = positive_samples.shape[0] * dist_util.world_size()
        metrics["kimg"] = (step + 1) * global_bsz / 1000.0
        metrics["forward_kimg"] = (step + 1) * global_bsz / 1000.0 * forward_dict["gen_per_label"]

        logger.log_dict(metrics)
        step += 1

        if step % save_per_step == 0 or step == total_steps:
            dist_util.barrier("save checkpoint started")
            save_checkpoint(module, optimizer, ema, step, ema_decay=ema_decay,
                            keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                ema, step=step, ema_decay=ema_decay, workdir=workdir,
                kind="gen", model_config=model_config,
            )
            dist_util.barrier("save checkpoint finished")

        if (step % eval_per_step == 0) or (step == 1) or (step == total_steps):
            is_sanity = (step == 1)  # sanity check that the FID env works

            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            module.eval()
            gen_step_fn = make_gen_step(get_ema_module(), postprocess_fn, device)
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            for eval_cfg in eval_cfg_list:
                dist_util.barrier("eval started")
                result = evaluate_fid(
                    dataset_name=dataset_name,
                    gen_func=gen_step_fn,
                    gen_params={"cfg_scale": eval_cfg},
                    eval_loader=eval_loader,
                    logger=logger,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{ema_decay:g}",
                    rng_eval=seed + 1,
                )
                dist_util.barrier("eval finished")
                fid_val = result.get("fid", float("inf"))
                if fid_val < round_best_fid:
                    round_best_fid = fid_val
                    round_best_cfg = eval_cfg
            if not is_sanity:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, step)
                logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})

        if step % 100 == 0:
            dist_util.barrier(f"train step {step} finished")

    dist_util.barrier("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist_util.barrier("train loop cleanup finished")


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name

    from pt.models.generator import DitGen

    dist_util.init_distributed()
    if "hsdp_dim" in config:
        log_for_0("hsdp_dim=%s is ignored under the torch backend (DDP).", config["hsdp_dim"])

    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        has_clip=False,
    )
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name:
                mae_path = f"hf://{model_name}"
    if bool(feature_cfg.get("use_mae", True)) and not mae_path:
        raise ValueError("feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) is required when use_mae=true.")
    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        convnext_bf16=bool(feature_cfg.get("convnext_bf16", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
    )
    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        # jax artifacts carry num_classes inside model_config (it comes from
        # the dataset section); keep the torch metadata identical.
        model_config={**dict(config.model), "num_classes": config.dataset.num_classes},
        workdir=output_dir,
        **config.train,
    )
    dist_util.barrier("main_gen finished")
    del model_dict
    gc.collect()
    dist_util.barrier("main_gen finished")


def main(args):
    dist_util.init_distributed()
    config = load_config(args.config)
    main_gen(config, output_dir=args.workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="runs")
    args = parser.parse_args()
    args.output_dir = args.workdir
    main(args)
