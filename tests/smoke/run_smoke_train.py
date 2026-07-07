"""End-to-end smoke test for the torch training loops on synthetic data.

Runs in the torch env (GPU or CPU):
    python tests/smoke/run_smoke_train.py [--tmp /tmp/drift_smoke]

Exercises: fake latent cache -> MAE training (train step, 4-variant eval,
checkpoint + EMA artifact, resume) -> generator training with the tiny MAE as
frozen feature model (banks, CFG sampling, drift loss, EMA, step-1 sanity FID
against fake reference stats).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pt.utils.misc import _dict_to_easydict  # noqa: E402
from utils import env  # noqa: E402


def make_fake_cache(root: Path, n_classes=4, n_train=40, n_val=12):
    rs = np.random.RandomState(11)
    for split, n in (("train", n_train), ("val", n_val)):
        for c in range(n_classes):
            d = root / split / f"class_{c:03d}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                torch.save(
                    {
                        "moments": rs.randn(32, 32, 4).astype(np.float32) * 0.5,
                        "moments_flip": rs.randn(32, 32, 4).astype(np.float32) * 0.5,
                    },
                    d / f"img_{i:04d}.pt",
                )


def make_fake_fid_stats(path: Path):
    mu = np.zeros(2048, dtype=np.float64)
    sigma = np.eye(2048, dtype=np.float64)
    np.savez(path, mu=mu, sigma=sigma)


MAE_CONFIG = {
    "logging": {"use_wandb": False, "log_every_k": 1},
    "dataset": {
        "resolution": 256, "use_aug": False, "use_latent": True, "use_cache": True,
        "num_classes": 1000, "batch_size": 16, "eval_batch_size": 16,
        "kwargs": {"num_workers": 0, "pin_memory": False},
    },
    "model": {
        "base_channels": 32, "patch_size": 2, "dropout_prob": 0.0,
        "layers": [1, 1, 1, 1], "in_channels": 4, "use_bf16": True,
        "input_patch_size": 1,
    },
    "optimizer": {
        "lr_schedule": {"learning_rate": 1e-3, "warmup_steps": 2,
                        "lr_schedule": "const", "total_steps": 100},
        "weight_decay": 0.01, "adam_b1": 0.9, "adam_b2": 0.95,
    },
    "train": {
        "seed": 42, "total_steps": 6, "save_per_step": 4, "eval_per_step": 3,
        "eval_samples": 32, "ema_decay": 0.999, "max_grad_norm": 2.0,
        "finetune_last_steps": 2, "warmup_finetune": 1, "finetune_cls": 0.1,
        "keep_every": 100, "keep_last": 2,
        "forward_dict": {"mask_ratio_min": 0.5, "mask_ratio_max": 0.5, "lambda_cls": 0.0},
        "eval_forward_dict": {"mask_ratio_min": 0.5, "mask_ratio_max": 0.5},
    },
}

GEN_CONFIG = {
    "logging": {"use_wandb": False, "log_every_k": 1},
    "dataset": {
        "resolution": 256, "use_aug": False, "use_latent": True, "use_cache": True,
        "num_classes": 1000, "batch_size": 16, "eval_batch_size": 32,
        "kwargs": {"num_workers": 0, "pin_memory": False},
    },
    "model": {
        "cond_dim": 64, "input_size": 32, "in_channels": 4, "patch_size": 2,
        "hidden_size": 64, "depth": 2, "num_heads": 2, "mlp_ratio": 4.0,
        "out_channels": 4, "use_qknorm": True, "use_swiglu": True,
        "use_rope": True, "use_rmsnorm": True, "n_cls_tokens": 4,
        "noise_classes": 8, "noise_coords": 4, "use_bf16": True, "attn_fp32": True,
    },
    "optimizer": {
        "lr_schedule": {"learning_rate": 2e-4, "warmup_steps": 2,
                        "lr_schedule": "const", "total_steps": 100},
        "weight_decay": 0.01, "adam_b1": 0.9, "adam_b2": 0.95,
    },
    "train": {
        "train_batch_size": 8, "seed": 42, "total_steps": 3, "save_per_step": 3,
        "eval_per_step": 100, "pos_per_sample": 8, "neg_per_sample": 4,
        "positive_bank_size": 16, "negative_bank_size": 64,
        "forward_dict": {"gen_per_label": 4, "cfg_min": 1.0, "cfg_max": 4.0,
                         "neg_cfg_pw": 3.0},
        "activation_kwargs": {"patch_mean_size": [2, 4], "patch_std_size": [2, 4],
                              "use_std": True, "use_mean": True, "every_k_block": 2},
        "loss_kwargs": {"R_list": [0.2, 0.05, 0.02]},
        "cfg_list": [1.0], "ema_decay": 0.999, "push_per_step": 16,
        "eval_samples": 500,
    },
    "feature": {"mae_path": "", "use_mae": True, "use_convnext": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="/tmp/drift_smoke")
    args = ap.parse_args()

    tmp = Path(args.tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    cache = tmp / "cache"
    make_fake_cache(cache)
    env.IMAGENET_CACHE_PATH = str(cache)

    fid_npz = tmp / "fake_fid_stats.npz"
    make_fake_fid_stats(fid_npz)
    import pt.utils.fid_util as fid_util

    for k in list(getattr(fid_util, "_REF_STATS", {}) or {}):
        pass  # no-op; patch below covers the lookup table
    # Patch whatever module-level path table exists.
    for attr in dir(fid_util):
        val = getattr(fid_util, attr)
        if isinstance(val, dict) and "imagenet256" in val:
            val["imagenet256"] = str(fid_npz)
    if hasattr(fid_util, "IMAGENET_FID_NPZ"):
        fid_util.IMAGENET_FID_NPZ = str(fid_npz)

    # ---- MAE smoke ----
    print("=== MAE training smoke ===")
    from pt.train_mae import main_mae

    mae_workdir = tmp / "mae_run"
    main_mae(_dict_to_easydict(json.loads(json.dumps(MAE_CONFIG))), output_dir=str(mae_workdir))

    metrics_file = mae_workdir / "log" / "metrics.jsonl"
    assert metrics_file.exists(), "MAE metrics.jsonl missing"
    lines = [json.loads(l) for l in metrics_file.read_text().splitlines()]
    train_lines = [l for l in lines if "loss" in l]
    assert train_lines, "no MAE train metrics logged"
    assert any("eval/loss" in l for l in lines), "no MAE eval metrics logged"
    assert (mae_workdir / "params_ema" / "model.safetensors").exists()
    assert list((mae_workdir / "checkpoints").glob("checkpoint_*.pt")), "no MAE checkpoint"
    print("MAE steps logged:", len(train_lines), "final loss:", train_lines[-1]["loss"])

    # ---- MAE resume smoke ----
    print("=== MAE resume smoke ===")
    cfg2 = json.loads(json.dumps(MAE_CONFIG))
    cfg2["train"]["total_steps"] = 8
    main_mae(_dict_to_easydict(cfg2), output_dir=str(mae_workdir))
    lines2 = [json.loads(l) for l in metrics_file.read_text().splitlines()]
    steps_logged = sorted({l["step"] for l in lines2 if "loss" in l})
    assert max(steps_logged) >= 7, f"resume did not continue: {steps_logged}"
    print("resume OK, steps:", steps_logged)

    # ---- Generator smoke (uses the tiny MAE as frozen feature model) ----
    print("=== Generator training smoke ===")
    from pt.train import main_gen

    gen_cfg = json.loads(json.dumps(GEN_CONFIG))
    gen_cfg["feature"]["mae_path"] = str(mae_workdir)
    gen_workdir = tmp / "gen_run"
    main_gen(_dict_to_easydict(gen_cfg), output_dir=str(gen_workdir))

    gmetrics = gen_workdir / "log" / "metrics.jsonl"
    assert gmetrics.exists(), "gen metrics.jsonl missing"
    glines = [json.loads(l) for l in gmetrics.read_text().splitlines()]
    gtrain = [l for l in glines if "loss" in l and "g_norm" in l]
    assert gtrain, "no generator train metrics"
    fid_lines = [l for l in glines if any("fid" in k for k in l)]
    assert fid_lines, "sanity FID did not log"
    assert (gen_workdir / "params_ema" / "model.safetensors").exists()
    print("gen steps:", len(gtrain), "final loss:", gtrain[-1]["loss"])
    print("sanity FID keys:", [k for l in fid_lines for k in l if "fid" in k])

    # Round-trip: the saved gen EMA artifact must rebuild + load.
    from pt.utils.init_util import load_generator_model_and_params

    model, meta = load_generator_model_and_params(str(gen_workdir), hf_cache_dir=str(tmp))
    assert meta["backend"] == "torch" and meta["kind"] == "gen"
    print("EMA artifact round-trip OK:", sum(p.numel() for p in model.parameters()), "params")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
