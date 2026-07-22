"""PyTorch port of models/generator.py (Flax LightningDiT / DitGen).

Numerics contract (mirrors the Flax source line by line):
  * Params are stored fp32; each linear casts input+weight to the compute dtype
    (bf16 when use_bf16) at call time — Flax's dtype/param_dtype split.
  * fp32 islands: RMSNorm variance, AdaLN modulation MLPs, TimestepEmbedder
    frequencies, and (iff attn_fp32) the attention softmax and RoPE tables.
  * The DiT operates on BHWC internally (patchify flattens patches
    channel-fastest, matching the pretrained patch-embed weights); public
    tensors at the DitGen boundary are BCHW.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# -----------------------------------------------------------------------------
# 1. Utils & Base Modules (Fixed Precision & Init)
# -----------------------------------------------------------------------------


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Sinusoidal positional encoding for a 1-D grid. Returns (len(pos), embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2-D sinusoidal positional encoding of shape (grid_size**2, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    embed_dim_half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class CastLinear(nn.Linear):
    """Linear with Flax dtype semantics: fp32 storage, cast to compute dtype at call.

    weight_init: "xavier_uniform" | "zeros" | "normal" (std 0.02); bias is
    always zero-initialized (the Flax TorchLinear only ever uses zeros).
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_init="xavier_uniform", compute_dtype=torch.float32):
        super().__init__(in_features, out_features, bias=bias)
        self.compute_dtype = compute_dtype
        with torch.no_grad():
            if weight_init == "zeros":
                nn.init.zeros_(self.weight)
            elif weight_init == "normal":
                nn.init.normal_(self.weight, std=0.02)
            else:
                nn.init.xavier_uniform_(self.weight)
            if bias:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        cd = self.compute_dtype
        b = self.bias.to(cd) if self.bias is not None else None
        return F.linear(x.to(cd), self.weight.to(cd), b)


class RMSNorm(nn.Module):
    """RMSNorm with fp32 variance (matches the Flax RMSNorm precision fix)."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        # bf16 * fp32 promotes to fp32, exactly like jnp
        normed = x * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            normed = normed * self.weight
        return normed.to(input_dtype)


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting over tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(q, k, dtype=torch.float32):
    """Rotary positional embedding on q, k of shape [B, N, H, D]."""
    B, N, H, D = q.shape
    half_dim = D // 2
    device = q.device
    freqs = (1.0 / (10000 ** (torch.arange(0, half_dim, device=device) / half_dim))).to(dtype)
    t = torch.arange(N, device=device, dtype=dtype)
    freqs = torch.outer(t, freqs)  # [N, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = torch.cos(emb)[None, :, None, :]
    sin = torch.sin(emb)[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, compute_dtype=torch.float32):
        super().__init__()
        # Attribute order matches the Flax call order: w1, w3, w2
        self.w1 = CastLinear(hidden_size, intermediate_size, compute_dtype=compute_dtype)
        self.w3 = CastLinear(hidden_size, intermediate_size, compute_dtype=compute_dtype)
        self.w2 = CastLinear(intermediate_size, hidden_size, compute_dtype=compute_dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardMLP(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_dim, compute_dtype=torch.float32):
        super().__init__()
        self.fc1 = CastLinear(hidden_size, mlp_hidden_dim, compute_dtype=compute_dtype)
        self.fc2 = CastLinear(mlp_hidden_dim, hidden_size, compute_dtype=compute_dtype)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="none"))


# -----------------------------------------------------------------------------
# 2. Core Blocks
# -----------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 use_rmsnorm=False, use_rope=False, attn_drop=0.0, proj_drop=0.0,
                 attn_fp32=True, compute_dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.attn_fp32 = attn_fp32
        self.compute_dtype = compute_dtype

        self.qkv = CastLinear(dim, dim * 3, bias=qkv_bias, compute_dtype=compute_dtype)
        if qk_norm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = CastLinear(dim, dim, compute_dtype=compute_dtype)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_qk=False):
        B, N, C = x.shape
        head_dim = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            rope_dtype = torch.float32 if self.attn_fp32 else self.compute_dtype
            q, k = apply_rope(q, k, dtype=rope_dtype)

        qk = (q, k) if return_qk else None

        attn_dtype = torch.float32 if self.attn_fp32 else self.compute_dtype
        q = q.to(attn_dtype) * (head_dim ** -0.5)
        k = k.to(attn_dtype)
        v = v.to(attn_dtype)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, qk


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_qknorm=False,
                 use_swiglu=False, use_rmsnorm=False, cond_dim=None, use_rope=False,
                 attn_fp32=True, compute_dtype=torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype

        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            attn_fp32=attn_fp32,
            compute_dtype=compute_dtype,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = int(2 / 3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size, compute_dtype=compute_dtype)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden_dim, compute_dtype=compute_dtype)

        # AdaLN modulation, computed in pure fp32 (plain nn.Linear, zero-init).
        # Index 1 in the Sequential matches the converter path `blocks_{i}/TorchLinear_0`.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        cd = self.compute_dtype
        chunks = self.adaLN(c.float()).to(cd)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks.chunk(6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa).to(cd)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)[0]

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp).to(cd)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False,
                 cond_dim=None, compute_dtype=torch.float32, tabular=False, feature_dims=None, feature_kinds=None, cat_softmax=False):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.tabular = tabular
        self.feature_dims = list(feature_dims) if feature_dims is not None else None
        self.feature_kinds = list(feature_kinds) if feature_kinds is not None else None
        self.cat_softmax = bool(cat_softmax and tabular and self.feature_kinds is not None)
        if use_rmsnorm:
            self.norm = RMSNorm(hidden_size)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

        
        if tabular:
            self.decoders = nn.ModuleList([
                CastLinear(hidden_size, w, weight_init="zeros", comput_dtype=compute_dtype) for w in self.feature_dims])
        else:
            out_dim = patch_size * patch_size * out_channels
            self.linear = CastLinear(
                hidden_size, out_dim,
                weight_init="zeros", compute_dtype=compute_dtype,
            )

    def forward(self, x, c):
        chunks = self.adaLN(c.float()).to(self.compute_dtype)
        shift, scale = chunks.chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        if self.tabular:
            if self.cat_softmax:
                blocks = []
                for i in range(len(self.decoders)):
                    b = self.decoders[i](x[:, i,:])
                    if self.feature_kinds[i] == "cat":
                        b = torch.softmax(b.float(), dim=-1).to(b.dtype)
                    blocks.append(b)
                return torch.cat(blocks, dim=1)
            return torch.cat([self.decoders[i](x[:, i, :]) for i in range(len(self.decoders))], dim=1)
        return self.linear(x)


# -----------------------------------------------------------------------------
# 3. Main Model
# -----------------------------------------------------------------------------


class LightningDiT(nn.Module):
    """DiT trunk. forward() takes/returns BHWC, matching the Flax source; the
    BCHW <-> BHWC permutes live in DitGen so the patchify reshape chain (which
    flattens patches channel-fastest) stays byte-identical to Flax."""

    def __init__(self, input_size=32, patch_size=2, in_channels=32, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4.0, out_channels=32,
                 use_qknorm=False, use_swiglu=False, use_rope=False, use_rmsnorm=False,
                 cond_dim=None, n_cls_tokens=0, attn_fp32=True,
                 compute_dtype=torch.float32, use_remat=False, tabular=False, feature_dims=None, feature_kinds=None, cat_softmax=False):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.n_cls_tokens = n_cls_tokens
        self.compute_dtype = compute_dtype
        self.use_remat = use_remat
        self.tabular = tabular

        if tabular:
            if feature_dims is None:
                feature_dims = [in_channels] * input_size
            self.feature_dims = [int(w) for w in feature_dims]
            self.num_features = len(self.feature_dims)
            self.data_dim = int(sum(self.feature_dims))
            offs = [0]
            for w in self.feature_dims:
                offs.append(offs[-1] + w)
            self.feature_offsets = offs
            num_patches = self.num_features
            self.patch_embed = nn.ModuleList([CastLinear(w, hidden_size, compute_dtype=compute_dtype) for w in self.feature_dims])
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02)
        else:
            target_grid = input_size // patch_size
            num_patches = target_grid * target_grid
    
            # effective patch dim is data-dependent (effective_p); the Linear's
            # in_features is fixed by the config as in Flax lazy init at trace time
            # (pixel models pass H == input_size so effective_p == patch_size).
            self.patch_embed = CastLinear(
                patch_size * patch_size * in_channels, hidden_size, compute_dtype=compute_dtype
            )
            pe = get_2d_sincos_pos_embed(hidden_size, target_grid)
            self.pos_embed = nn.Parameter(torch.from_numpy(pe).float()[None, :, :])

        if n_cls_tokens > 0:
            self.cls_proj = CastLinear(cond_dim, hidden_size, compute_dtype=compute_dtype)
            self.cls_embed = nn.Parameter(torch.randn(1, n_cls_tokens, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                cond_dim=cond_dim,
                use_rope=use_rope,
                attn_fp32=attn_fp32,
                compute_dtype=compute_dtype,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
            compute_dtype=compute_dtype,
            tabular=tabular,
            feature_dims = self.feature_dims if tabular else None,
            feature_kinds = feature_kinds if tabular else None,
            cat_softmax = cat_softmax,
        )

    def _forward_tabular(self, x, c):
        B = x.shape[0]
        x = x.reshape(B, self.data_dim)
        offs = self.feature_offsets 
        x = torch.stack([self.patch_embed[i](x[:,offs[i]:offs[i+1]]) for i in range(self.num_features)], dim=1)
        x = (x + self.pos_embed).to(self.compute_dtype)
        if self.n_cls_tokens > 0:
            c_in = c.to(self.compute_dtype)
            c_tokens = self.cls_proj(c_in).unsqueeze(1).expand(-1, self.n_cls_tokens, -1)
            c_tokens = (c_tokens + self.cls_embed).to(self.compute_dtype)
            x = torch.cat([c_tokens, x.to(self.compute_dtype)], dim=1)

        for block in self.blocks:
            if self.use_remat and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        # x = self.final_layer(x, c)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens:, :]
        return self.final_layer(x, c)

    
    def forward(self, x, c):
        if self.tabular:
            return self._forward_tabular(x, c)
        # x: [B, H, W, C] (BHWC, as in the Flax source)
        B, H, W, C = x.shape
        p = self.patch_size

        target_grid = self.input_size // p
        num_patches = target_grid * target_grid
        effective_p = H // target_grid
        grid_h, grid_w = target_grid, target_grid

        # Patch embed: flatten patches channel-fastest (h2, w2, C) — this
        # ordering is what the pretrained Linear weights expect.
        x = x.reshape(B, grid_h, effective_p, grid_w, effective_p, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, num_patches, effective_p * effective_p * C)
        x = self.patch_embed(x)

        x = (x + self.pos_embed).to(self.compute_dtype)

        if self.n_cls_tokens > 0:
            c_in = c.to(self.compute_dtype)
            c_tokens = self.cls_proj(c_in).unsqueeze(1)
            c_tokens = c_tokens.expand(-1, self.n_cls_tokens, -1)
            # Flax adds the fp32 cls_embed (promoting to fp32), concatenates,
            # then casts to compute dtype; casting each part first is
            # value-identical (bf16->fp32->bf16 round-trips exactly).
            c_tokens = (c_tokens + self.cls_embed).to(self.compute_dtype)
            x = torch.cat([c_tokens, x.to(self.compute_dtype)], dim=1)

        for block in self.blocks:
            if self.use_remat and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        out_size = self.input_size
        x = self.final_layer(x, c)

        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens:, :]

        x = x.reshape(B, grid_h, grid_w, p, p, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_size, out_size, self.out_channels)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, compute_dtype=torch.float32):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.compute_dtype = compute_dtype
        self.mlp = nn.Sequential(
            CastLinear(frequency_embedding_size, hidden_size, weight_init="normal",
                       compute_dtype=compute_dtype),
            nn.SiLU(),
            CastLinear(hidden_size, hidden_size, weight_init="normal",
                       compute_dtype=compute_dtype),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # cos first
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding).to(self.compute_dtype)


# -----------------------------------------------------------------------------
# 4. DitGen Wrapper
# -----------------------------------------------------------------------------


class DitGen(nn.Module):
    """One-step generator. forward() takes labels (+ optional injected noise)
    and returns {"samples": BCHW, "noise": {...}}.

    Constructor keys mirror the Flax DitGen fields exactly — metadata
    model_config and YAML config.model splat straight into this signature.
    """

    def __init__(self, cond_dim, num_classes=1001, noise_classes=0, noise_coords=1,
                 input_size=32, in_channels=3, n_cls_tokens=0, patch_size=2,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0,
                 out_channels=3, use_qknorm=False, use_swiglu=False, use_rope=False,
                 use_rmsnorm=False, use_bf16=False, attn_fp32=True, use_remat=False, tabular=False, feature_dims=None, feature_kinds=None, cat_softmax=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.input_size = input_size
        self.in_channels = in_channels
        self.tabular = tabular
        self.use_bf16 = use_bf16
        if tabular:
            patch_size = 1
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.compute_dtype = compute_dtype

        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        if noise_classes > 0:
            self.noise_embeds = nn.ModuleList()
            for _ in range(noise_coords):
                emb = nn.Embedding(noise_classes, cond_dim)
                nn.init.normal_(emb.weight, std=0.02)
                self.noise_embeds.append(emb)

        self.cfg_embedder = TimestepEmbedder(cond_dim, compute_dtype=compute_dtype)
        self.cfg_norm = RMSNorm(cond_dim)

        if tabular:
            if feature_dims is None:
                feature_dims = [in_channels] * input_size
            self.feature_dims = [int(w) for w in feature_dims]
            self.data_dim = int(sum(self.feature_dims))
            if feature_kinds is None:
                feature_kinds = ["cont"] * len(self.feature_dims)
            self.feature_kinds = list(feature_kinds)
            self.cat_softmax = bool(cat_softmax)
        else:
            self.feature_dims = None
            self.data_dim = None
            self.feature_kinds = None
            self.cat_softmax = False

        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
            n_cls_tokens=n_cls_tokens,
            attn_fp32=attn_fp32,
            compute_dtype=compute_dtype,
            use_remat=use_remat,
            tabular=tabular,
            feature_dims = self.feature_dims,
            feature_kinds = self.feature_kinds,
            cat_softmax = self.cat_softmax,
        )

    def generate_image(self, x, cond):
        return self.model(x, cond)

    def c_cfg_noise_to_cond(self, c, cfg_scale, noise_labels):
        B = c.shape[0]
        device = c.device
        # nn.Embedding lookups: Flax nn.Embed with dtype=bf16 casts the table
        # to bf16 at lookup; the sum below is therefore in compute dtype.
        cd = self.compute_dtype
        cond = self.class_embed(c).to(cd) if self.use_bf16 else self.class_embed(c)
        if self.noise_classes > 0:
            for i, embed in enumerate(self.noise_embeds):
                e = embed(noise_labels[:, i])
                cond = cond + (e.to(cd) if self.use_bf16 else e)

        if isinstance(cfg_scale, (float, int)):
            cfg_scale_t = torch.full((B,), float(cfg_scale), device=device)
        else:
            cfg_scale_t = torch.as_tensor(cfg_scale, device=device)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = cfg_scale_t[None].expand(B)
        cfg_scale_t = self.cfg_norm(self.cfg_embedder(cfg_scale_t))
        cond = cond + cfg_scale_t * 0.02

        if self.use_bf16:
            cond = cond.to(torch.bfloat16)
        return cond

    def forward(self, c, cfg_scale=1.0, temp=1.0, generator=None,
                noise=None, noise_labels=None):
        """Args:
            c: [B] int64 class labels.
            cfg_scale: float or [B] tensor.
            generator: torch.Generator for the noise stream.
            noise: optional injected latent noise, BHWC [B, S, S, C_in]
                (pre-`temp` scaling) — used by parity tests.
            noise_labels: optional injected [B, noise_coords] ints.
        """
        B = c.shape[0]
        device = c.device

        if noise is None:
            if self.tabular:
                noise = torch.randn(
                    B, self.data_dim,
                    device=device, generator=generator,
                )
            else:
                noise = torch.randn(
                    B, self.input_size, self.input_size, self.in_channels,
                    device=device, generator=generator,
                )
        x = noise * temp
        if self.use_bf16:
            x = x.to(torch.bfloat16)

        if noise_labels is None:
            noise_labels = torch.randint(
                0, max(1, self.noise_classes), (B, max(1, self.noise_coords)),
                device=device, generator=generator,
            )

        cond = self.c_cfg_noise_to_cond(c, cfg_scale, noise_labels)
        samples = self.generate_image(x, cond)

        if self.tabular:
            return {
                "samples": samples.unsqueeze(1),
                "noise": {"x": x, "noise_labels": noise_labels},
            }
        return {
            "samples": samples.permute(0, 3, 1, 2),  # BCHW at the public boundary
            "noise": {"x": x, "noise_labels": noise_labels},
        }


def build_generator_from_config(model_config) -> DitGen:
    """Build DitGen directly from a full config dict (e.g., from artifact metadata)."""
    return DitGen(**dict(model_config))
