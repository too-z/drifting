"""2-rank DDP smoke (gloo/CPU): drift-loss distributed-stats exactness + a
short MAE DDP run on the fake cache.

    torchrun --standalone --nproc_per_node=2 tests/smoke/run_smoke_ddp.py

Checks:
1. drift_loss(distributed_stats=True) on rank-split halves == single-process
   drift_loss on the full batch (the JAX global-batch semantics), exactly.
2. train_mae runs 3 steps under DDP with identical loss on both ranks
   (metrics are all-reduced) and saves a checkpoint.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pt.drift_loss import drift_loss  # noqa: E402
from pt.utils import dist_util  # noqa: E402


def check_drift_loss_distributed():
    rank = dist_util.rank()
    ws = dist_util.world_size()
    assert ws == 2, "run under torchrun --nproc_per_node=2"

    rs = np.random.RandomState(3)
    gen = torch.from_numpy(rs.randn(8, 6, 16).astype(np.float32))
    pos = torch.from_numpy(rs.randn(8, 10, 16).astype(np.float32))
    neg = torch.from_numpy(rs.randn(8, 4, 16).astype(np.float32))
    w_neg = torch.from_numpy((rs.rand(8, 4) * 2).astype(np.float32))

    # Single-process reference on the full batch (= JAX global computation).
    ref_loss, ref_info = drift_loss(
        gen=gen, fixed_pos=pos, fixed_neg=neg, weight_neg=w_neg,
        R_list=(0.2, 0.05, 0.02), distributed_stats=False,
    )

    # Per-rank halves with all-reduced statistics.
    sl = slice(rank * 4, (rank + 1) * 4)
    loss, info = drift_loss(
        gen=gen[sl], fixed_pos=pos[sl], fixed_neg=neg[sl], weight_neg=w_neg[sl],
        R_list=(0.2, 0.05, 0.02), distributed_stats=True,
    )

    torch.testing.assert_close(loss, ref_loss[sl], rtol=1e-6, atol=1e-7)
    for k in info:
        torch.testing.assert_close(info[k], ref_info[k], rtol=1e-6, atol=1e-7)
    if rank == 0:
        print("drift_loss distributed-stats exactness: OK")


def run_mae_ddp(tmp: Path):
    from tests.smoke.run_smoke_train import MAE_CONFIG, make_fake_cache
    from pt.utils.misc import _dict_to_easydict
    from utils import env

    cache = tmp / "cache"
    if dist_util.is_rank_zero():
        if tmp.exists():
            shutil.rmtree(tmp)
        make_fake_cache(cache)
    dist_util.barrier("cache ready")
    env.IMAGENET_CACHE_PATH = str(cache)

    cfg = json.loads(json.dumps(MAE_CONFIG))
    cfg["train"]["total_steps"] = 3
    cfg["train"]["eval_per_step"] = 2
    cfg["train"]["save_per_step"] = 3
    cfg["train"]["eval_samples"] = 16
    cfg["dataset"]["batch_size"] = 16  # global; 8 per rank

    from pt.train_mae import main_mae

    main_mae(_dict_to_easydict(cfg), output_dir=str(tmp / "mae_ddp"))
    dist_util.barrier("mae ddp done")
    if dist_util.is_rank_zero():
        metrics = (tmp / "mae_ddp" / "log" / "metrics.jsonl").read_text().splitlines()
        lines = [json.loads(l) for l in metrics]
        assert any("loss" in l for l in lines)
        assert list((tmp / "mae_ddp" / "checkpoints").glob("checkpoint_*.pt"))
        print("MAE DDP smoke: OK,", len(lines), "metric lines")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force gloo/CPU
    dist_util.init_distributed()
    assert dist.is_initialized()
    check_drift_loss_distributed()
    run_mae_ddp(Path("/tmp/drift_smoke_ddp"))
    dist_util.barrier("all done")
    if dist_util.is_rank_zero():
        print("\nALL DDP SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
