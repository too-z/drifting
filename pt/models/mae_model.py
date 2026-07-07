"""PyTorch port of models/mae_model.py (MAE-ResNet feature model).

Layout: NCHW everywhere (torch-native). The only layout-sensitive spot is
patch_input: Flax flattens each input patch channel-last (h2, w2, c); in NCHW
the packed channel dim must keep that (h2 w2 c) ordering so converted conv1
weights line up (HWIO->OIHW preserves input-channel ordering).

Precision: params fp32; every conv/norm/linear casts to the compute dtype at
call time (Flax dtype semantics). GroupNorm statistics are computed in fp32
(Flax _compute_stats upcasts) — eps is 1e-6, Flax's default, NOT torch's 1e-5.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class CastConv2d(nn.Conv2d):
    def __init__(self, *args, compute_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_dtype = compute_dtype

    def forward(self, x):
        cd = self.compute_dtype
        b = self.bias.to(cd) if self.bias is not None else None
        return self._conv_forward(x.to(cd), self.weight.to(cd), b)


class CastGroupNorm(nn.GroupNorm):
    """GroupNorm with fp32 statistics + affine, output cast to compute dtype
    (mirrors Flax GroupNorm: stats in fp32, result cast to `dtype`)."""

    def __init__(self, num_groups, num_channels, eps=1e-6, compute_dtype=torch.float32):
        super().__init__(num_groups, num_channels, eps=eps)
        self.compute_dtype = compute_dtype

    def forward(self, x):
        y = F.group_norm(x.float(), self.num_groups, self.weight, self.bias, self.eps)
        return y.to(self.compute_dtype)


class CastLinear(nn.Linear):
    def __init__(self, *args, compute_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_dtype = compute_dtype

    def forward(self, x):
        cd = self.compute_dtype
        b = self.bias.to(cd) if self.bias is not None else None
        return F.linear(x.to(cd), self.weight.to(cd), b)


class _BasicBlock(nn.Module):
    """ResNet basic block. Projection conv/GN exist only when the residual
    shape changes (stride!=1 or channel change) — matching the params Flax
    lazily materialized (see tests/parity/reference_trees/tree_mae_256.txt)."""

    def __init__(self, filters, in_channels, stride=1, gn_max_groups=32,
                 dropout_prob=0.0, compute_dtype=torch.float32):
        super().__init__()
        cd = compute_dtype
        self.conv1 = CastConv2d(in_channels, filters, 3, stride=stride, padding=1,
                                bias=False, compute_dtype=cd)
        self.gn1 = CastGroupNorm(_choose_gn_groups(filters, gn_max_groups), filters,
                                 compute_dtype=cd)
        self.conv2 = CastConv2d(filters, filters, 3, stride=1, padding=1,
                                bias=False, compute_dtype=cd)
        self.gn2 = CastGroupNorm(_choose_gn_groups(filters, gn_max_groups), filters,
                                 compute_dtype=cd)
        self.drop = nn.Dropout(dropout_prob)
        self.has_proj = stride != 1 or in_channels != filters
        if self.has_proj:
            self.proj_conv = CastConv2d(in_channels, filters, 1, stride=stride,
                                        bias=False, compute_dtype=cd)
            self.proj_gn = CastGroupNorm(_choose_gn_groups(filters, gn_max_groups),
                                         filters, compute_dtype=cd)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.gn2(y)

        if self.has_proj:
            residual = self.proj_gn(self.proj_conv(residual))

        return F.relu(residual + y)


class _ResNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=64, layers=(2, 2, 2, 2),
                 dropout_prob=0.0, gn_max_groups=32, compute_dtype=torch.float32):
        super().__init__()
        cd = compute_dtype
        self.conv1 = CastConv2d(in_channels, base_channels, 3, stride=1, padding=1,
                                bias=False, compute_dtype=cd)
        self.gn1 = CastGroupNorm(_choose_gn_groups(base_channels, gn_max_groups),
                                 base_channels, compute_dtype=cd)

        stages = []
        ch = base_channels
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = ch * (2 ** stage_idx) if stage_idx > 0 else ch
            in_ch = ch if stage_idx == 0 else (ch * (2 ** (stage_idx - 1)))
            blocks = [_BasicBlock(out_ch, in_channels=in_ch, stride=stride,
                                  dropout_prob=dropout_prob, compute_dtype=cd)]
            for _ in range(1, num_blocks):
                blocks.append(_BasicBlock(out_ch, in_channels=out_ch, stride=1,
                                          dropout_prob=dropout_prob, compute_dtype=cd))
            stages.append(nn.ModuleList(blocks))
            setattr(self, f"layer{stage_idx + 1}_norm",
                    CastGroupNorm(_choose_gn_groups(out_ch, gn_max_groups), out_ch,
                                  compute_dtype=cd))
        self.stages = nn.ModuleList(stages)

    def forward(self, x, return_block_outputs=False):
        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        feats["conv1"] = x

        for i, blocks in enumerate(self.stages):
            layer_name = f"layer{i + 1}"
            outs: List[torch.Tensor] = []
            for block in blocks:
                x = block(x)
                outs.append(x)
            block_outputs[layer_name] = outs
            norm_layer = getattr(self, f"{layer_name}_norm")
            x = norm_layer(x)
            feats[layer_name] = x
        if return_block_outputs:
            return feats, block_outputs
        return feats


class _ConvGNReLU(nn.Module):
    def __init__(self, in_channels, channels, kernel=3, compute_dtype=torch.float32):
        super().__init__()
        self.conv = CastConv2d(in_channels, channels, kernel, padding=kernel // 2,
                               bias=False, compute_dtype=compute_dtype)
        self.gn = CastGroupNorm(_choose_gn_groups(channels, 32), channels,
                                compute_dtype=compute_dtype)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, compute_dtype=torch.float32):
        super().__init__()
        cat_ch = in_channels + skip_channels
        self.concat_norm_fn = CastGroupNorm(32, cat_ch, compute_dtype=compute_dtype)
        self.proj = _ConvGNReLU(cat_ch, out_channels, kernel=3, compute_dtype=compute_dtype)
        self.refine = _ConvGNReLU(out_channels, out_channels, kernel=3,
                                  compute_dtype=compute_dtype)

    def forward(self, x, skip):
        # jax.image.resize(..., "bilinear") on an upscale == align_corners=False
        # (antialias only differs on downscale).
        x = F.interpolate(x.float(), size=skip.shape[2:], mode="bilinear",
                          align_corners=False).to(x.dtype)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels, out_channels, compute_dtype=torch.float32):
        super().__init__()
        cd = compute_dtype
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8
        self.bridge = _ConvGNReLU(c5, c5, compute_dtype=cd)
        self.up43 = _UpBlock(c5, c4, c4, compute_dtype=cd)
        self.up32 = _UpBlock(c4, c3, c3, compute_dtype=cd)
        self.up21 = _UpBlock(c3, c2, c2, compute_dtype=cd)
        self.up10 = _UpBlock(c2, c1, c1, compute_dtype=cd)
        self.head = CastConv2d(c1, out_channels, 1, compute_dtype=cd)

    def forward(self, feats):
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    """NCHW space-to-depth with (h2 w2 c) channel ordering — the order Flax's
    NHWC '(h2 w2 c)' flatten produces, which conv1's converted weights expect."""
    if input_patch_size == 1:
        return x
    return rearrange(
        x,
        "b c (h1 h2) (w1 w2) -> b (h2 w2 c) h1 w1",
        h2=input_patch_size,
        w2=input_patch_size,
    )


def make_patch_mask(x: torch.Tensor, mask_ratio: torch.Tensor, patch_size: int = 4,
                    generator=None) -> torch.Tensor:
    """Random patch mask, NCHW: returns (B, 1, H, W) in x.dtype."""
    b, _, h, w = x.shape
    nh, nw = h // patch_size, w // patch_size
    noise = torch.rand(b, nh, nw, device=x.device, generator=generator)
    mask = (noise < mask_ratio[:, None, None]).to(x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=1)
    mask = mask.repeat_interleave(patch_size, dim=2)
    return mask[:, None]


def safe_std(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


class MAEResNet(nn.Module):
    """Constructor keys mirror MAEResNetJAX fields; metadata model_config plus
    num_classes splat directly into this signature."""

    def __init__(self, num_classes=1000, in_channels=3, base_channels=64,
                 patch_size=4, dropout_prob=0.0, layers=(2, 2, 2, 2),
                 use_bf16=False, input_patch_size=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.input_patch_size = input_patch_size
        self.use_bf16 = use_bf16
        cd = torch.bfloat16 if use_bf16 else torch.float32
        self.compute_dtype = cd

        enc_in = in_channels * input_patch_size * input_patch_size
        self.encoder = _ResNetEncoder(
            in_channels=enc_in,
            base_channels=base_channels,
            layers=tuple(layers),
            dropout_prob=dropout_prob,
            compute_dtype=cd,
        )
        self.decoder = _UNetDecoder(
            base_channels=base_channels,
            out_channels=in_channels * input_patch_size * input_patch_size,
            compute_dtype=cd,
        )
        self.fc = CastLinear(base_channels * 8, num_classes, compute_dtype=cd)

    def forward(self, x, labels, lambda_cls=0.0, mask_ratio_min=0.75,
                mask_ratio_max=0.75, generator=None, mask=None):
        """x: (B, C, H, W). Returns (loss[B], metrics dict).

        mask: optional injected (B, 1, H', W') mask (post-patch_input layout)
        for parity tests; when None it is drawn from `generator`.
        """
        cd = self.compute_dtype
        x = x.to(cd)
        x = patch_input(x, self.input_patch_size)
        b = x.shape[0]
        if mask is None:
            mask_ratio = (
                torch.rand(b, device=x.device, generator=generator)
                * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
            ).to(cd)
            mask = make_patch_mask(x, mask_ratio, self.patch_size, generator=generator)
        else:
            mask = mask.to(cd)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in)
        top = feats["layer4"]
        pooled = top.mean(dim=(2, 3))
        logits = self.fc(pooled)
        recon = self.decoder(feats)

        log_probs = F.log_softmax(logits, dim=-1)
        cls_loss = -log_probs.gather(-1, labels[:, None]).squeeze(-1)
        mse = (recon - x) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (logits.argmax(dim=-1) == labels).to(cd),
            "mask_ratio": mask.float().mean(dim=(1, 2, 3)).to(cd),
        }
        return loss, metrics

    def get_activations(self, x, patch_mean_size: Optional[List[int]] = [2, 4],
                        patch_std_size: Optional[List[int]] = [2, 4],
                        use_std=True, use_mean=True, every_k_block=2):
        """Multi-scale features for the drift loss. x: (B, C, H, W).

        Gradients flow through (the generated-samples branch differentiates
        through this); the frozen positives/negatives branch is wrapped in
        torch.no_grad() by the caller, mirroring the external stop_gradient
        in train.py.
        """
        patch_mean_size = patch_mean_size or []
        patch_std_size = patch_std_size or []

        x = x.to(self.compute_dtype)
        x = patch_input(x, self.input_patch_size)
        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            feats, block_outputs = self.encoder(x, return_block_outputs=True)
        else:
            feats = self.encoder(x)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out["norm_x"] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]

        def process_feat(name, feat):
            b, c, h, w = feat.shape
            out[name] = feat.flatten(2).transpose(1, 2)  # b (h w) c
            if use_mean:
                out[f"{name}_mean"] = feat.mean(dim=(2, 3))[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat, dim=(2, 3))[:, None, :]

            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(
                        feat, "b c (h s1) (w s2) -> b (h w) (s1 s2) c", s1=size, s2=size
                    )
                    out[f"{name}_mean_{size}"] = reshaped.mean(dim=2)

            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(
                        feat, "b c (h s1) (w s2) -> b (h w) (s1 s2) c", s1=size, s2=size
                    )
                    out[f"{name}_std_{size}"] = safe_std(reshaped, dim=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                blocks = block_outputs.get(lname, [])
                for blk_idx, feat_i in enumerate(blocks, start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out


def mae_from_metadata(metadata) -> MAEResNet:
    model_config = dict(metadata.get("model_config", {}) or {})
    num_classes = int(model_config.pop("num_classes", 1000))
    return MAEResNet(num_classes=num_classes, **model_config)
