"""FID-only inference entrypoint — torch port of inference.py.

Usage:
    python -m pt.inference --init-from "hf://latent_L_sota" --workdir runs/fid

CLI is identical to the JAX version; --hsdp-dim is accepted and ignored
(single-process torch evaluation; the model is small enough per GPU).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from pt.dataset.dataset import create_imagenet_split, get_postprocess_fn
from pt.utils import dist_util
from pt.utils.fid_util import evaluate_fid
from pt.utils.init_util import load_generator_model_and_params
from pt.utils.logging import WandbLogger
from pt.utils.rng import make_generator
from utils.env import HF_ROOT


def _hf_root():
    return os.environ.get("HF_ROOT", HF_ROOT)


def _is_latent(metadata: dict) -> bool:
    """Determine if the model operates in latent space from its metadata."""
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _load_model(init_from: str, device=None):
    """Build generator, load EMA params, return (gen_step, metadata).

    gen_step follows the pt.utils.fid_util.evaluate_fid contract:
    gen_step(batch, rng=<int>, cfg_scale=<float>) -> [0,1] BCHW samples.
    """
    if device is None:
        device = dist_util.device()
    model, metadata = load_generator_model_and_params(init_from, hf_cache_dir=_hf_root())
    model = model.to(device)
    latent = _is_latent(metadata)
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=latent)

    @torch.no_grad()
    def gen_step(batch, rng=0, cfg_scale=1.0):
        _, labels = batch
        labels = torch.as_tensor(np.asarray(labels)).long().to(device)
        g = make_generator(device, "eval-noise", int(rng))
        samples = model(labels, cfg_scale=float(cfg_scale), generator=g)["samples"]
        return postprocess_fn(samples)

    return gen_step, metadata


# ---------------------------------------------------------------------------
# eval_fid
# ---------------------------------------------------------------------------

def run_eval_fid(
    gen_step, metadata, init_from: str, workdir: str,
    *, num_samples: int, cfg_scale: float, eval_batch_size: int,
    use_wandb: bool, wandb_entity: str | None, wandb_project: str, wandb_name: str | None,
) -> dict:
    eval_loader, _, _ = create_imagenet_split(
        resolution=256, split="val",
        batch_size=eval_batch_size // dist_util.world_size(),
        num_workers=0,
    )

    work_path = Path(workdir).resolve()
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(work_path),
        log_every_k=1,
    )

    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_step,
        gen_params={"cfg_scale": cfg_scale},
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=(num_samples >= 50000),
        eval_isc=True,
        eval_fid=True,
        rng_eval=0,
    )
    logger.finish()
    return {"init_from": init_from, "cfg_scale": cfg_scale, "metadata": metadata, **metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference: FID evaluation.")
    parser.add_argument("--init-from", required=True,
                        help="hf://<name> or local checkpoint path.")
    parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--hsdp-dim", type=int, default=None,
                        help="Ignored under the torch backend (kept for CLI parity).")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="release-fid")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    dist_util.init_distributed()
    gen_step, metadata = _load_model(args.init_from)
    result = run_eval_fid(
        gen_step, metadata, args.init_from, args.workdir,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        eval_batch_size=args.eval_batch_size,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference_from_args(args)
    print(json.dumps(result, indent=2))
    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
