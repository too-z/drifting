"""Capture JAX reference fixtures for the torch parity tests.

The ONLY parity file that imports jax/flax. Run from the repo root in the JAX
env (CPU is fine):

    python tests/parity/capture_jax_fixtures.py --which gen drift lr
    python tests/parity/capture_jax_fixtures.py --which all

Writes .npz files to tests/parity/fixtures/. Every fixture stores its inputs
alongside outputs; bf16 outputs are saved upcast to fp32.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FIXDIR = REPO / "tests" / "parity" / "fixtures"
DEFAULT_ROOT = os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache"))


def capture_generator(root, name="ablation"):
    import jax.numpy as jnp
    from flax import serialization

    from models.generator import DitGen, LightningDiTBlock, build_generator_from_config

    art = Path(root) / "models" / "gen" / "jax" / name
    if not (art / "ema_params.msgpack").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Goodeat/drifting",
            allow_patterns=[f"models/gen/jax/{name}/*"],
            local_dir=root,
        )
    metadata = json.loads((art / "metadata.json").read_text())
    params = serialization.msgpack_restore((art / "ema_params.msgpack").read_bytes())
    cfg = dict(metadata["model_config"])

    rs = np.random.RandomState(0)
    B = 2
    x = rs.randn(B, cfg["input_size"], cfg["input_size"], cfg["in_channels"]).astype(np.float32)
    c = np.array([1, 977], dtype=np.int32)
    noise_labels = rs.randint(
        0, max(1, cfg.get("noise_classes", 0)),
        size=(B, max(1, cfg.get("noise_coords", 1))),
    ).astype(np.int32)
    cfg_scale = 1.5

    out = {
        "x": x,
        "c": c,
        "noise_labels": noise_labels,
        "cfg_scale": np.float32(cfg_scale),
    }

    # Full model at two precisions: a pure-fp32 reference (tight tolerances)
    # and the shipped config (bf16 realism).
    variants = [
        ("fp32", {"use_bf16": False, "attn_fp32": True}),
        ("bf16", {}),
    ]
    for tag, overrides in variants:
        mcfg = {**cfg, **overrides}
        model = build_generator_from_config(mcfg)
        xx = jnp.asarray(x)
        if mcfg["use_bf16"]:
            xx = xx.astype(jnp.bfloat16)
        cond = model.apply(
            {"params": params},
            jnp.asarray(c),
            cfg_scale,
            jnp.asarray(noise_labels),
            method=DitGen.c_cfg_noise_to_cond,
        )
        samples = model.apply(
            {"params": params}, xx, cond, method=DitGen.generate_image
        )
        out[f"cond_{tag}"] = np.asarray(cond, dtype=np.float32)
        out[f"samples_{tag}"] = np.asarray(samples, dtype=np.float32)

    # Single block (blocks_0) on synthetic input.
    N = cfg.get("n_cls_tokens", 0) + (cfg["input_size"] // cfg["patch_size"]) ** 2
    xblk = (rs.randn(B, N, cfg["hidden_size"]) * 0.5).astype(np.float32)
    cblk = rs.randn(B, cfg["cond_dim"]).astype(np.float32)
    out["block_x"] = xblk
    out["block_c"] = cblk
    for tag, overrides in variants:
        mcfg = {**cfg, **overrides}
        dt = jnp.bfloat16 if mcfg["use_bf16"] else jnp.float32
        blk = LightningDiTBlock(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            use_qknorm=cfg["use_qknorm"],
            use_swiglu=cfg["use_swiglu"],
            use_rmsnorm=cfg["use_rmsnorm"],
            cond_dim=cfg["cond_dim"],
            use_rope=cfg["use_rope"],
            attn_fp32=mcfg["attn_fp32"],
            dtype=dt,
            param_dtype=jnp.float32,
        )
        y = blk.apply(
            {"params": params["LightningDiT_0"]["blocks_0"]},
            jnp.asarray(xblk, dt),
            jnp.asarray(cblk, dt),
        )
        out[f"block_out_{tag}"] = np.asarray(y, dtype=np.float32)

    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / f"gen_{name}.npz", **out)
    print(f"wrote {FIXDIR / f'gen_{name}.npz'} ({len(out)} arrays)")


def capture_drift_loss():
    import jax.numpy as jnp

    from drift_loss import drift_loss

    rs = np.random.RandomState(1)
    gen = rs.randn(6, 8, 16).astype(np.float32)
    pos = rs.randn(6, 12, 16).astype(np.float32)
    neg = rs.randn(6, 4, 16).astype(np.float32)
    w_gen = np.ones((6, 8), np.float32)
    w_pos = np.ones((6, 12), np.float32)
    w_neg = (rs.rand(6, 4) * 2).astype(np.float32)

    loss, info = drift_loss(
        gen=jnp.asarray(gen),
        fixed_pos=jnp.asarray(pos),
        fixed_neg=jnp.asarray(neg),
        weight_gen=jnp.asarray(w_gen),
        weight_pos=jnp.asarray(w_pos),
        weight_neg=jnp.asarray(w_neg),
        R_list=(0.2, 0.05, 0.02),
    )
    out = {
        "gen": gen, "pos": pos, "neg": neg,
        "w_gen": w_gen, "w_pos": w_pos, "w_neg": w_neg,
        "loss": np.asarray(loss, np.float32),
    }
    for k, v in info.items():
        out[f"info_{k}"] = np.asarray(v, np.float32)

    # Variant without negatives (fixed_neg=None path).
    loss2, info2 = drift_loss(
        gen=jnp.asarray(gen), fixed_pos=jnp.asarray(pos), R_list=(0.2, 0.05, 0.02)
    )
    out["loss_noneg"] = np.asarray(loss2, np.float32)
    for k, v in info2.items():
        out[f"info_noneg_{k}"] = np.asarray(v, np.float32)

    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / "drift_loss.npz", **out)
    print(f"wrote {FIXDIR / 'drift_loss.npz'}")


def capture_lr_schedule():
    import optax

    # Inline copy of utils/model_builder.py:create_learning_rate_fn (avoids the
    # dataset import chain).
    def create(learning_rate, warmup_steps, total_steps, lr_schedule="const"):
        warmup_fn = optax.linear_schedule(
            init_value=1e-6, end_value=learning_rate, transition_steps=warmup_steps
        )
        if lr_schedule in ["cosine", "cos"]:
            cosine_steps = max(total_steps - warmup_steps, 1)
            schedule_fn = optax.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=cosine_steps, alpha=1e-6
            )
        elif lr_schedule == "const":
            schedule_fn = optax.constant_schedule(value=learning_rate)
        else:
            raise NotImplementedError(lr_schedule)
        return optax.join_schedules(
            schedules=[warmup_fn, schedule_fn], boundaries=[warmup_steps]
        )

    steps = np.array(
        [0, 1, 2, 100, 2500, 4999, 5000, 5001, 7500, 9999, 10000, 10001,
         50000, 99999, 100000, 150000, 199999, 200000],
        dtype=np.int64,
    )
    out = {"steps": steps}
    cases = {
        "const_2e4_w5000_t100000": dict(learning_rate=2e-4, warmup_steps=5000,
                                        total_steps=100000, lr_schedule="const"),
        "const_4e4_w10000_t200000": dict(learning_rate=4e-4, warmup_steps=10000,
                                         total_steps=200000, lr_schedule="const"),
        "cos_4e4_w10000_t200000": dict(learning_rate=4e-4, warmup_steps=10000,
                                       total_steps=200000, lr_schedule="cosine"),
        "cos_4e3_w4000_t200000": dict(learning_rate=4e-3, warmup_steps=4000,
                                      total_steps=200000, lr_schedule="cos"),
    }
    for name, kwargs in cases.items():
        fn = create(**kwargs)
        out[name] = np.array([float(fn(s)) for s in steps], dtype=np.float64)

    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / "lr_schedule.npz", **out)
    print(f"wrote {FIXDIR / 'lr_schedule.npz'}")


CAPTURES = {
    "gen": lambda args: [capture_generator(args.root, n) for n in args.gen_names],
    "drift": lambda args: capture_drift_loss(),
    "lr": lambda args: capture_lr_schedule(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["all"],
                    choices=list(CAPTURES) + ["all"])
    ap.add_argument("--gen-names", nargs="+", default=["ablation"],
                    help="generator artifact names for --which gen")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    which = list(CAPTURES) if "all" in args.which else args.which
    for name in which:
        CAPTURES[name](args)


if __name__ == "__main__":
    main()
