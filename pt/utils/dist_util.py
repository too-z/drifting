"""torch.distributed helpers — replaces utils/hsdp_util.py's mesh/sharding and
jax.experimental.multihost_utils in the torch backend.

Mapping from the JAX release: jax.process_index() -> RANK,
jax.process_count() -> WORLD_SIZE, jax.distributed.initialize() ->
init_distributed(). NOTE the unit change: a JAX "process" was one TPU host
(8 chips); a torch rank is one GPU, so per-process batch splits divide by
WORLD_SIZE.
"""

import os

import torch
import torch.distributed as dist

_INITIALIZED = False


def init_distributed():
    """Idempotent process-group init from torchrun env vars (no-op single-process)."""
    global _INITIALIZED
    if _INITIALIZED or (dist.is_available() and dist.is_initialized()):
        _INITIALIZED = True
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    _INITIALIZED = True


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero():
    return rank() == 0


def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def barrier(tag=""):
    """Drop-in for multihost_utils.sync_global_devices(tag)."""
    del tag
    if is_dist():
        dist.barrier()


def all_reduce_mean(metrics):
    """Mean-reduce a dict of scalar tensors/floats across ranks; returns floats."""
    keys = sorted(metrics)
    vals = torch.tensor(
        [float(metrics[k]) for k in keys], dtype=torch.float64, device=device()
    )
    if is_dist():
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals = vals / world_size()
    return {k: float(v) for k, v in zip(keys, vals)}
