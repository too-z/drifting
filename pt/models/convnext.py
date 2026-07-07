"""Torch ConvNeXtV2 feature extractor for the drift loss.

Unlike the other model ports, this does NOT re-implement the architecture:
the Flax models/convnext.py was itself converted *from* the transformers
checkpoint facebook/convnextv2-base-22k-224, so we use ConvNextV2Model
directly and implement only get_activations on top of the stage outputs
(matching models/convnext.py:ConvNextV2.get_activations, including the
misspelled "convenxt_stage_{i}" keys).

Input: (B, 3, H, W) ImageNet-normalized images (mean/std applied by the
caller, see build_activation_function in pt/models/mae_model.py's trainer
counterpart). Resized to 224 with antialiased bilinear — jax.image.resize
antialiases on downscale.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pt.models.mae_model import safe_std

_MODEL_NAMES = {
    "base": "facebook/convnextv2-base-22k-224",
    "tiny": "facebook/convnextv2-tiny-22k-224",
}


class ConvNextFeatures(nn.Module):
    def __init__(self, model_name: str = "base", use_bf16: bool = False):
        super().__init__()
        from transformers import ConvNextV2Model

        self.backbone = ConvNextV2Model.from_pretrained(_MODEL_NAMES[model_name])
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
        if use_bf16:
            # Flax runs params-fp32/compute-bf16; casting the whole backbone is
            # the closest torch equivalent (weight rounding == flax's per-op
            # cast). fp32 is the only path the released configs use.
            self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.eval()

    @staticmethod
    def _normalize(y: torch.Tensor) -> torch.Tensor:
        """Per-position channel normalization (NCHW dim=1), fp32, eps on std."""
        old_dtype = y.dtype
        y = y.float()
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True, correction=0)  # jnp.std is ddof=0
        y = (y - mean) / (std + 1e-3)
        return y.to(old_dtype)

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 3, H, W) ImageNet-normalized. Returns dict of (B, T, D)."""
        x = F.interpolate(
            x.float(), size=(224, 224), mode="bilinear",
            align_corners=False, antialias=True,
        ).to(self.compute_dtype)

        feature_dict: Dict[str, torch.Tensor] = {}
        hidden = self.backbone.embeddings(x)
        for i, stage in enumerate(self.backbone.encoder.stages):
            hidden = stage(hidden)
            x_normed = self._normalize(hidden)
            tokens = x_normed.flatten(2).transpose(1, 2)  # b (h w) c
            if i > 0:
                feature_dict[f"convenxt_stage_{i}"] = tokens
            feature_dict[f"convenxt_stage_{i}_mean"] = x_normed.mean(dim=(2, 3))[:, None, :]
            feature_dict[f"convenxt_stage_{i}_std"] = safe_std(tokens, dim=1)[:, None, :]

        feature_dict["global_mean"] = self.backbone.layernorm(
            hidden.mean(dim=(2, 3))
        )[:, None, :]
        feature_dict["global_std"] = safe_std(
            self._normalize(hidden).flatten(2).transpose(1, 2), dim=1
        )[:, None, :]
        return feature_dict

    def forward(self, x):
        return self.get_activations(x)


def load_convnext_model(model_name: str = "base", use_bf16: bool = False):
    """Counterpart of models/convnext.py:load_convnext_jax_model (no param
    tree needed in torch; and no HF-cache deletion)."""
    model = ConvNextFeatures(model_name=model_name, use_bf16=use_bf16)
    model.requires_grad_(False)
    return model
