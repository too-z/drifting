"""Rank-aware metric logging"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
try:
    from absl import logging as absl_logging
except ModuleNotFoundError:
    import logging as _py_logging
    absl_logging = _py_logging.getLogger("pt")
    if not absl_logging.handlers:
      _py_logging.basicConfig(level=_py_logging.INFO)

from run.utils import dist_util


def is_rank_zero() -> bool:
    return dist_util.is_rank_zero()


def log_for_0(msg, *args, **kwargs):
    if is_rank_zero():
        absl_logging.info(msg, *args, **kwargs)


def log_for_all(msg):
    absl_logging.info("[Rank %s] %s", dist_util.rank(), msg)


class MetricLogger:
    def __init__(self) -> None:
        self.step = 0
        self.log_every_k = 1
        self._buffer: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self.offline_dir = Path("log")

    def set_logging(
        self,
        offline_dir: str = "log",
        workdir: Optional[str] = None,
        log_every_k: int = 1,
        **kwargs,
    ) -> None:
        del kwargs
        self.log_every_k = int(log_every_k)
        workdir_path = Path(workdir).resolve() if workdir else None
        self.offline_dir = workdir_path / "log" if workdir_path is not None else Path(offline_dir)
        self.offline_dir.mkdir(parents=True, exist_ok=True)

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        reduced = {k: (self._buffer[k] / max(1, self._count.get(k, 1))) for k in self._buffer.keys()}
        p = self.offline_dir / "metrics.jsonl"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.step, **reduced}, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._count.clear()

    def log_dict(self, d: Dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        reduced = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = float(v.detach().float().cpu().mean())
            if isinstance(v, np.ndarray):
                v = float(np.asarray(v).mean())
            if isinstance(v, (int, float, np.floating, np.integer)):
                reduced[k] = float(v)
        for k, v in reduced.items():
            self._buffer[k] = self._buffer.get(k, 0.0) + float(v)
            self._count[k] = self._count.get(k, 0) + 1
        if self.log_every_k <= 1 or (self.step % self.log_every_k == 0):
            self._flush_buffer()

    def log_dict_dir(self, prefix: str, d: Dict[str, Any]) -> None:
        """Log a dict with keys namespaced by prefix."""
        self.log_dict({f"{prefix}/{k}": v for k, v in d.items()})

    def finish(self) -> None:
        self._flush_buffer()
        
