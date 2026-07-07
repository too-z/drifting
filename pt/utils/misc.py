"""Torch port of utils/misc.py: config loading + a best-effort profile stub.

prepare_rng/ddp_rand_func are replaced by pt/utils/rng.py; run_init by
pt/utils/dist_util.py.
"""

from __future__ import annotations

import os
import time

import torch
import yaml


# adapted from https://github.com/NVlabs/edm
class EasyDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value):
        self[name] = value


def _dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    out = EasyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_easydict(v)
        elif isinstance(v, list):
            out[k] = [_dict_to_easydict(i) for i in v]
        else:
            out[k] = v
    return out


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return _dict_to_easydict(yaml.safe_load(f))


def profile_func(fn, args, name="fn"):
    """Best-effort replacement for the XLA cost-analysis profiler: reports GPU
    peak memory and wall time for one call. FLOPs reporting is dropped
    (jax-only); metric keys keep the profile/ namespace."""
    metrics = {}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics[f"profile/{name}_memory_GB"] = torch.cuda.max_memory_allocated() / 1e9
        metrics[f"profile/{name}_time_s"] = time.time() - t0
    except Exception:
        pass
    return metrics
