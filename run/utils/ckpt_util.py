"""Torch port of utils/ckpt_util.py.

- checkpoints/: resumable snapshots, single-file torch.save per step (under
  DDP the state is fully replicated, so rank 0 saves; the JAX multihost
  allgather is unnecessary). keep/keep_every pruning reproduces
  flax.training.checkpoints semantics: keep the newest `keep` checkpoints,
  never delete steps divisible by `keep_every`.
- params_ema/: release EMA export as {model.safetensors, metadata.json} with
  backend "torch" — loadable by pt/utils/init_util.py and (after conversion)
  interchangeable with the jax msgpack artifact.
- No RNG state is saved (parity with JAX): determinism on resume comes from
  re-deriving generators from (seed, stream, step, rank), see pt/utils/rng.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from pt.utils import dist_util
from pt.utils.logging import log_for_0

_CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def _list_checkpoints(ckpt_dir: Path):
    """Return [(step, path)] sorted by step ascending."""
    found = []
    if ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            m = _CKPT_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    return sorted(found)


def restore_checkpoint(module, optimizer, ema, step=None, workdir: Optional[str] = None) -> int:
    """Load the latest (or given-step) checkpoint into module/optimizer/ema.

    Returns the restored step (0 if nothing to restore). `ema` is a mutable
    {param_name: tensor} dict updated in place.
    """
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpts = _list_checkpoints(ckpt_dir)
    if not ckpts:
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return 0
    if step is not None:
        matches = [p for s, p in ckpts if s == int(step)]
        if not matches:
            raise FileNotFoundError(f"no checkpoint for step {step} in {ckpt_dir}")
        path = matches[0]
    else:
        step, path = ckpts[-1]

    payload = torch.load(path, map_location="cpu", weights_only=False)
    module.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and payload.get("ema") is not None:
        for k, v in payload["ema"].items():
            ema[k] = v.to(ema[k].device if k in ema else "cpu")
    log_for_0("Restored checkpoint step %d from %s", payload["step"], str(path))
    return int(payload["step"])


def save_checkpoint(module, optimizer, ema, step: int, *, ema_decay: float,
                    keep=2, keep_every=None, workdir: Optional[str] = None,
                    config: Optional[dict] = None):
    dist_util.barrier("save_checkpoint_before")
    if dist_util.is_rank_zero():
        ckpt_dir = _job_ckpt_dir(workdir=workdir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_for_0("Saving checkpoint step %d to %s", step, str(ckpt_dir))
        payload = {
            "step": int(step),
            "model": {k: v.detach().cpu() for k, v in module.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in (ema or {}).items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "ema_decay": float(ema_decay),
            "config": dict(config) if config else None,
        }
        path = ckpt_dir / f"checkpoint_{int(step)}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

        # Prune: newest `keep` retained; steps divisible by keep_every retained.
        ckpts = _list_checkpoints(ckpt_dir)
        prune = ckpts[:-keep] if keep and len(ckpts) > keep else []
        for s, p in prune:
            if keep_every and s > 0 and s % int(keep_every) == 0:
                continue
            p.unlink(missing_ok=True)
    dist_util.barrier("save_checkpoint_barrier")


def save_params_ema_artifact(
    ema: Dict[str, torch.Tensor],
    *,
    step: int,
    ema_decay: float,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the release EMA tree as a standalone restorable artifact.

    - `checkpoints/` stores resumable snapshots.
    - `params_ema/` stores the exported EMA params + metadata used by
      restore/infer/HF flows (torch-backend layout: model.safetensors).
    """
    out_dir = _output_root(workdir) / "params_ema"
    if dist_util.is_rank_zero():
        from safetensors.torch import save_file

        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in ema.items()},
            str(out_dir / "model.safetensors"),
        )
        metadata = {
            "format": "safetensors",
            "kind": kind,
            "backend": "torch",
            "ema_decay": float(ema_decay),
            "step": int(step),
            "path": "model.safetensors",
            "model_config": dict(model_config or {}),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    dist_util.barrier("save_params_ema_artifact")
    return out_dir
