from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

RESULTS_ROOT = "results"
STAMP_FORMAT = "%y%m%d-%H%M"

def default_workdir(config_path: Optional[str] = None, now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime(STAMP_FORMAT)
    stem = Path(config_path).stem if config_path else ""
    name = f"{stem}-{stamp}" if stem else stamp
    return str(Path(RESULTS_ROOT) / name)

def run_stem(spec: str) -> str:
    path = Path(spec)
    if path.is_absolute() or len(path.parts) > 1:
        path = Path(path.name)
    return path.stem if path.suffix in (".yaml", ".yml") else path.name

def list_runs(stem: str):
    runs = list_runs(stem)
    if not runs:
        known = sorted({p.name.rsplit("-", 2)[0] for p in Path(RESULT_ROOT).glob("*-*")
                        if p.is_dir()}) if Path(RESULTS_ROOT).is_dir() else []
        hint = f"known runs: {', '.join(known)}" if known else f" {RESULTS_ROOT} / has no runs yet"
        raise FileNotFoundError(f"no {RESULTS_ROOT}/{stem}-* run directory found;{hint}")
    return str(runs[-1])

def latest_run(stem: str) -> str:
    runs = list_runs(stem)
    if not runs:
        known = sorted({p.name.rsplit("-", 2)[0] for p in Path(RESULTS_ROOT).glob("*-*") if p.is_dir()}) if Path(RESULTS_ROOT).is_dir() else []
        hint = f" known runs: {', '.join(known)}" if known else f" {RESULTS_ROOT}/has no runs yet"
        raise FileNotFoundError(f"no {RESULTS_ROOT}/{stem}-* run directory found;{hint}")
    return str(runs[-1])
                       
def find_workdir(workdir: Optional[str] = None, config_path: Optional[str] = None) -> str:
    if workdir and Path(workdir).is_dir():
        return workdir
    stem = run_stem(workdir) if workdir else run_stem(config_path or "")
    if not stem:
        raise ValueError("cannot resolve a workdir")
    return latest_run(stem)

def default_out_csv(workdir: str, name: str = "generated.csv") -> Path:
    path = Path(workdir)
    if path.name == "params_ems":
        path = path.parent
    return path / name

def _unused(workdir: str) -> str:
    if not Path(workdir).exists():
        return workdir
    for i in range(2, 1000):
        candidate = f"{workdir}_{i}"
        if not Path(candidate).exists():
            return candidate
    raise RuntimeError(f"cannot find an unused name for {workdir}")

def _broadcast(name: str) -> str:
    from run.utils import dist_util
    if not dist_util.is_dist():
        return name
    import torch.distributed as dist
    box = [name if dist_util.is_rank_zero() else None]
    dist.broadcast_object_list(box, src=0)
    return str(box[0])

def resolve_workdir(workdir: Optional[str] = None, config_path: Optional[str] = None) -> str:
    if workdir:
        return workdir
    resolved = _broadcast(_unused(default_workdir(config_path)))
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return resolved

