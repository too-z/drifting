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


def capture_mae(root, name="mae_latent_256"):
    import jax.numpy as jnp
    from flax import serialization

    from models.mae_model import MAEResNetJAX, patch_input

    art = Path(root) / "models" / "mae" / "jax" / name
    if not (art / "ema_params.msgpack").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Goodeat/drifting",
            allow_patterns=[f"models/mae/jax/{name}/*"],
            local_dir=root,
        )
    metadata = json.loads((art / "metadata.json").read_text())
    params = serialization.msgpack_restore((art / "ema_params.msgpack").read_bytes())
    cfg = dict(metadata["model_config"])
    num_classes = int(cfg.pop("num_classes", 1000))

    rs = np.random.RandomState(2)
    B = 2
    ips = cfg.get("input_patch_size", 1)
    H = 32 * ips  # production latent/pixel grid
    x = rs.randn(B, H, H, cfg["in_channels"]).astype(np.float32)
    labels = np.array([3, 500], dtype=np.int32)
    lambda_cls = 0.3

    # Fixed patch mask on the post-patch_input layout (H' = H / ips).
    hp = H // ips
    psz = cfg.get("patch_size", 4)
    nh = hp // psz
    noise = rs.rand(B, nh, nh)
    ratios = np.array([0.5, 0.7], dtype=np.float32)
    mask = (noise < ratios[:, None, None]).astype(np.float32)
    mask = np.repeat(np.repeat(mask, psz, axis=1), psz, axis=2)[..., None]  # (B,H',W',1)

    out = {
        "x": x, "labels": labels, "mask": mask,
        "lambda_cls": np.float32(lambda_cls),
    }

    act_kwargs = dict(patch_mean_size=[2, 4], patch_std_size=[2, 4],
                      use_std=True, use_mean=True, every_k_block=2)

    for tag, use_bf16 in [("fp32", False), ("bf16", True)]:
        model = MAEResNetJAX(num_classes=num_classes, **{**cfg, "use_bf16": use_bf16})
        dt = jnp.bfloat16 if use_bf16 else jnp.float32

        acts = model.apply({"params": params}, jnp.asarray(x),
                           method=model.get_activations, **act_kwargs)
        for k, v in acts.items():
            out[f"act_{tag}_{k}"] = np.asarray(v, dtype=np.float32)

        # Loss path with the injected mask (mirrors __call__ minus rng).
        def fwd(m, xp, mk):
            x_in = xp * (1.0 - mk)
            feats = m.encoder(x_in, train=False)
            pooled = feats["layer4"].mean(axis=(1, 2))
            logits = m.fc(pooled)
            recon = m.decoder(feats)
            return logits, recon

        xp = patch_input(jnp.asarray(x, dt), ips)
        mk = jnp.asarray(mask, dt)
        logits, recon = model.apply({"params": params}, xp, mk, method=fwd)
        out[f"logits_{tag}"] = np.asarray(logits, np.float32)
        out[f"recon_{tag}"] = np.asarray(recon, np.float32)

        # Loss composition exactly as MAEResNetJAX.__call__.
        import jax

        one_hot = jax.nn.one_hot(jnp.asarray(labels), num_classes, dtype=dt)
        cls_loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
        mse = (recon - xp) ** 2
        recon_loss = (mse * mk).sum(axis=(1, 2, 3)) / (mk.sum(axis=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        out[f"cls_loss_{tag}"] = np.asarray(cls_loss, np.float32)
        out[f"recon_loss_{tag}"] = np.asarray(recon_loss, np.float32)
        out[f"loss_{tag}"] = np.asarray(loss, np.float32)

    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / f"mae_{name}.npz", **out)
    print(f"wrote {FIXDIR / f'mae_{name}.npz'} ({len(out)} arrays)")


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


def capture_vae(args):
    """Flax SD-VAE reference: deterministic encode moments + decode pixels."""
    import jax.numpy as jnp
    from diffusers.models import FlaxAutoencoderKL

    vae, vae_params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax")

    rs = np.random.RandomState(4)
    img = (rs.rand(2, 3, 64, 64).astype(np.float32) * 2 - 1)  # BCHW in [-1, 1]
    lat = rs.randn(2, 8, 8, 4).astype(np.float32)  # BHWC latents (flax layout)

    dist = vae.apply(
        {"params": vae_params}, jnp.asarray(img), method=FlaxAutoencoderKL.encode
    ).latent_dist
    dec = vae.apply(
        {"params": vae_params}, jnp.asarray(lat) / 0.18215,
        method=FlaxAutoencoderKL.decode,
    ).sample

    out = {
        "img": img,
        "lat": lat,
        "enc_mean": np.asarray(dist.mean, np.float32),
        "enc_std": np.asarray(dist.std, np.float32),
        "decoded": np.asarray(dec, np.float32),
    }
    for k, v in out.items():
        print(f"  vae fixture {k}: {v.shape}")
    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / "vae.npz", **out)
    print(f"wrote {FIXDIR / 'vae.npz'}")


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


def capture_fid(args):
    """FID/IS/PR evaluation pipeline on deterministic fake images (CPU).

    Mirrors utils/fid_util._compute_stats piecewise: uint8 NHWC -> torch resize
    (299, (x-128)/128) -> Flax Inception (pretrained pickle via jax_fid.cvt,
    downloaded to /tmp on first run) -> pooled feats + logits. The pmap
    plumbing of _build_jax_inception is skipped (1 CPU device); the Flax module
    is applied directly in small batches, which is numerically identical.

    Also saves: FID between the two 32-image halves' feature stats, the
    inception score (same math as utils/fid_util._compute_inception_score),
    and precision/recall between the halves (k=3).
    """
    import jax
    import jax.numpy as jnp
    import torch

    from utils.jax_fid import inception as jax_inception
    from utils.jax_fid import resize as torch_resize
    from utils.jax_fid.cvt import load_all as load_inception_params
    from utils.jax_fid.fid import compute_frechet_distance
    from utils.jax_fid.precision_recall import compute_precision_recall

    rs = np.random.RandomState(3)
    images = rs.randint(0, 256, (64, 256, 256, 3), dtype=np.uint8)

    # Same preprocessing as utils/fid_util._compute_stats: uint8 NHWC ->
    # float32 BCHW -> torch resize -> NHWC for the Flax inception.
    x = torch.from_numpy(images.astype(np.float32).transpose(0, 3, 1, 2))
    resized = torch_resize.forward(x).numpy().astype(np.float32)  # (64, 3, 299, 299)

    model = jax_inception.InceptionV3(pretrained=True, include_head=True, transform_input=False)
    params = load_inception_params()

    nhwc = resized.transpose(0, 2, 3, 1)
    pooled_list, logits_list = [], []
    bs = 16  # small CPU batches
    for i in range(0, len(nhwc), bs):
        pooled_i, _, logits_i = model.apply(params, jnp.asarray(nhwc[i : i + bs]), train=False)
        pooled_list.append(np.asarray(pooled_i))
        logits_list.append(np.asarray(logits_i))
    pooled = np.concatenate(pooled_list).astype(np.float32)
    logits = np.concatenate(logits_list).astype(np.float32)

    # FID between the two halves' feature stats (float64 mu/cov, as in
    # utils/fid_util._compute_stats).
    a64 = pooled[:32].astype(np.float64)
    b64 = pooled[32:].astype(np.float64)
    fid_half = compute_frechet_distance(
        np.mean(a64, axis=0), np.mean(b64, axis=0),
        np.cov(a64, rowvar=False), np.cov(b64, rowvar=False),
    )

    # Inception score: inline copy of utils/fid_util._compute_inception_score
    # (importing utils.fid_util would pull in the multihost/dataset chain).
    def _inception_score(lg, splits=10):
        rng = np.random.RandomState(2020)
        lg = lg[rng.permutation(lg.shape[0]), :]
        probs = np.asarray(jax.nn.softmax(lg, axis=-1), dtype=np.float64)
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

    isc_mean, isc_std = _inception_score(logits)

    precision, recall = compute_precision_recall(a64, b64, k=3)

    out = {
        "images": images,
        "resized": resized,
        "pooled": pooled,
        "logits": logits,
        "fid_half": np.float64(fid_half),
        "isc_mean": np.float64(isc_mean),
        "isc_std": np.float64(isc_std),
        "precision": np.float64(precision),
        "recall": np.float64(recall),
    }
    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / "fid.npz", **out)
    print(f"wrote {FIXDIR / 'fid.npz'} ({len(out)} arrays)")


def capture_optim(args):
    """optax adamw + external clip_by_global_norm trajectory on synthetic
    tensors — pins the torch AdamW + clip_grad_norm_ + EMA port."""
    import jax
    import jax.numpy as jnp
    import optax

    rs = np.random.RandomState(7)
    init_w = rs.randn(4, 3).astype(np.float32)
    init_b = rs.randn(3).astype(np.float32)
    params = {"w": jnp.asarray(init_w), "b": jnp.asarray(init_b)}
    n_steps = 25
    grads_w = rs.randn(n_steps, 4, 3).astype(np.float32) * 3.0  # exceeds clip sometimes
    grads_b = rs.randn(n_steps, 3).astype(np.float32) * 3.0

    lr_values = np.array(
        [1e-6 + (2e-3 - 1e-6) * min(t / 10, 1.0) for t in range(n_steps)],
        dtype=np.float64,
    )
    tx = optax.adamw(
        learning_rate=lambda t: lr_values[int(t)] if int(t) < n_steps else lr_values[-1],
        weight_decay=0.01, b1=0.9, b2=0.95,
    )
    opt_state = tx.init(params)
    ema = jax.tree.map(lambda p: p, params)
    ema_decay = 0.97
    clipper = optax.clip_by_global_norm(2.0)

    traj_w, traj_b, traj_ema_w, gnorms = [], [], [], []
    for t in range(n_steps):
        grads = {"w": jnp.asarray(grads_w[t]), "b": jnp.asarray(grads_b[t])}
        gnorms.append(float(optax.global_norm(grads)))
        clipped, _ = clipper.update(grads, None)
        updates, opt_state = tx.update(clipped, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree.map(lambda e, p: e * ema_decay + p * (1 - ema_decay), ema, params)
        traj_w.append(np.asarray(params["w"]))
        traj_b.append(np.asarray(params["b"]))
        traj_ema_w.append(np.asarray(ema["w"]))

    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXDIR / "optim.npz",
        init_w=init_w, init_b=init_b,
        grads_w=grads_w, grads_b=grads_b, lr_values=lr_values,
        traj_w=np.stack(traj_w), traj_b=np.stack(traj_b),
        traj_ema_w=np.stack(traj_ema_w), gnorms=np.asarray(gnorms, np.float64),
        ema_decay=np.float64(ema_decay),
    )
    print(f"wrote {FIXDIR / 'optim.npz'}")


def capture_convnext(args):
    """Flax ConvNeXtV2-base get_activations reference (fp32 — the production
    path; configs never set convnext_bf16).

    Replicates models/convnext.py:load_convnext_jax_model WITHOUT its
    rmtree(~/.cache/huggingface) step, which would nuke shared HF caches.
    """
    import jax
    import jax.numpy as jnp

    from utils.hsdp_util import set_global_mesh

    set_global_mesh(1)

    from models.convnext import ConvNextBase, ConvNextV2, convert_weights_to_jax
    from transformers import ConvNextV2ForImageClassification

    model_jax = ConvNextBase(dtype=jnp.float32)
    dummy = jnp.ones((1, 224, 224, 3))
    params = model_jax.init(jax.random.PRNGKey(0), dummy)
    sd = ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-base-22k-224"
    ).state_dict()
    params = convert_weights_to_jax(params, sd, hf=True)

    rs = np.random.RandomState(5)
    # ImageNet-normalized-scale input, BHWC 256x256 (resized to 224 inside).
    x = (rs.randn(2, 256, 256, 3) * 0.8).astype(np.float32)

    feats = model_jax.apply(params, jnp.asarray(x), method=ConvNextV2.get_activations)

    out = {"x": x}
    for k, v in feats.items():
        out[f"cn_{k}"] = np.asarray(v, np.float32)
    FIXDIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXDIR / "convnext.npz", **out)
    print(f"wrote {FIXDIR / 'convnext.npz'} ({len(out)} arrays)")


CAPTURES = {
    "gen": lambda args: [capture_generator(args.root, n) for n in args.gen_names],
    "mae": lambda args: [capture_mae(args.root, n) for n in args.mae_names],
    "drift": lambda args: capture_drift_loss(),
    "lr": lambda args: capture_lr_schedule(),
    "vae": capture_vae,
    "convnext": capture_convnext,
    "optim": capture_optim,
    "fid": capture_fid,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["all"],
                    choices=list(CAPTURES) + ["all"])
    ap.add_argument("--gen-names", nargs="+", default=["ablation"],
                    help="generator artifact names for --which gen")
    ap.add_argument("--mae-names", nargs="+", default=["mae_latent_256"],
                    help="MAE artifact names for --which mae")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    which = list(CAPTURES) if "all" in args.which else args.which
    for name in which:
        CAPTURES[name](args)


if __name__ == "__main__":
    main()
