"""Torch port of drift_loss.py.

Semantics preserved from the JAX source:
  * All inputs are cast to fp32.
  * The "goal" is computed from detached tensors under torch.no_grad()
    (jax.lax.stop_gradient); only `gen` carries gradient in the final MSE.
  * info values are scalar tensors (post .mean()), detached.

Distributed note (the one intentional addition): two scalar statistics are
means over the batch axis — `scale` (drift_loss.py:70) and the per-R force
normalizer `f_norm_val` (drift_loss.py:114). Under JAX these were computed
over the *global* sharded batch inside jit; per-rank DDP would compute them
locally and change training behavior. With ``distributed_stats=True`` (the
default when torch.distributed is initialized) they are all-reduced. Local
per-rank batches have equal shapes, so mean-of-means is the exact global
mean, and both live inside the no-grad region, so gradients are unaffected.
"""

import torch
import torch.distributed as dist


def _global_mean(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Mean over all ranks of a scalar tensor (exact when local shapes match)."""
    if enabled and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # AVG is NCCL-only
        x = x / dist.get_world_size()
    return x


def cdist(x, y, eps=1e-8):
    # [B, N, D] x [B, M, D] -> [B, N, M]
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
    distributed_stats=None,
):
    """
    Args:
        gen: [B, C_g, S]
        fixed_pos: [B, C_p, S]
        fixed_neg: [B, C_n, S] (optional, can be None)
        weight_gen: [B, C_g] (optional; if None: weight is 1)
        weight_pos: [B, C_p] (optional; if None: weight is 1)
        weight_neg: [B, C_n] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
        distributed_stats: all-reduce the batch-mean statistics across ranks
            (defaults to True when torch.distributed is initialized).
    Returns:
        loss: [batch_size]
        info: dict with entries: scale, loss_{R} for each R
    """
    if distributed_stats is None:
        distributed_stats = dist.is_available() and dist.is_initialized()

    # 1. Defaults & Casting
    B, C_g, S = gen.shape

    if fixed_neg is None:
        fixed_neg = gen[:, :0, :].detach()
    C_n = fixed_neg.shape[1]
    C_p = fixed_pos.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    # 2+3. Goal computation — entirely gradient-free (jax.lax.stop_gradient).
    with torch.no_grad():
        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

        info = {}
        d = cdist(old_gen, targets)
        weighted_dist = d * targets_w[:, None, :]  # [B, C_g, C_g + C_n + C_p]
        scale = _global_mean(weighted_dist.mean(), distributed_stats) / _global_mean(
            targets_w.mean(), distributed_stats
        )
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)  # order-1 coords
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = d / torch.clamp(scale, min=1e-3)

        # --- Masking ---
        mask_val = 100.0
        diag_mask = torch.eye(C_g, dtype=torch.float32, device=gen.device)
        block_mask = torch.nn.functional.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        # --- Force Loop ---
        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R

            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))

            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(dim=-1)  # guaranteed 0 in no_repulsion case
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = _global_mean((total_force_R ** 2).mean(), distributed_stats)

            info[f"loss_{R}"] = f_norm_val

            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    info = {k: v.mean().detach() for k, v in info.items()}

    return loss, info
