from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from pt.utils import dist_util

_tabular_cache: dict = {}

def _as_tensor(x):
  if isinstance(x, torch.Tensor):
    return x
  return torch.as_tensor(np.asarray(x))

def worker_init_fn(worker_id:int, rank:int) -> None:
  seed = worker_id + rank * 1000
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def _resolve_csv_path(csv_path: str) -> str:
  if os.path.isabs(csv_path) and os.path.exists(csv_path):
    return csv_path
  from utils import env

  candidates = [
    csv_path, 
    os.path.join(env.TABULAR_DATA_ROOT, csv_path),
    os.path.join(str(env.REPO), csv_path),
    os.path.join(str(env.BASE), csv_path),
  ]
  for cand in candidates:
    if os.path.exists(cand):
      return os.path.abspath(cand)
  raise FileNotFoundError("tabular csv not found")

def _load_tabular(csv_path, target_col, drop_cols, val_frac, seed):
  csv_path = _resolve_csv_path(csv_path)
  key = (os.path.abspath(csv_path), target_col, tuple(sorted(drop_cols)), float(val_frac), int(seed))
  if key in _tabular_cache:
    return _tabular_cache[key]

  df = pd.read_csv(csv_path)
  if target_col not in df.columns:
    raise ValueError("target col not found")

  feat_cols = [c for c in df.columns if c != target_col and c not in set(drop_cols)]
  X = df[feat_cols].to_numpy().astype(np.float32)
  labels = df[target_col].to_numpy().astype(np.int64)

  n = X.shape[0]
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n)
  n_val = min(n - 1, max(1, int(round(n * val_frac)))) if n > 1 else 0
  val_idx = perm[:n_val]
  train_idx = perm[n_val:]

  train_X = X[train_idx]
  with np.errstate(invalid="ignore"):
    mean = np.nanmean(train_X, axis=0)
    std = np.nanstd(train_X, axis=0)
  mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
  std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)

  n_missing = int(np.isnan(X).sum())
  if n_missing:
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, mean[None, :], X).astype(np.float32)

  result = {
    "X": X,
    "labels": labels,
    "feat_cols": feat_cols,
    "n_features": len(feat_cols),
    "num_classes": int(labels.max()) + 1,
    "train_idx": train_idx,
    "val_idx": val_idx,
    "mean": mean,
    "std": std,
    "n_missing": n_missing,
  }
  _tabular_cache[key] = result
  return result

class TabularDataset(Dataset):
  def __init__(self, X, labels, indices, mean, std):
    self.X = X
    self.labels = labels
    self.indices = np.asarray(indices)
    self.mean = mean
    self.std = std

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, i:int):
    idx = int(self.indices[i])
    x = (self.X[idx] - self.mean) / self.std
    x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0) # [1, n_features]
    y = int(self.labels[idx])
    return x, y

def create_tabular_split(
  *,
  csv_path: str,
  batch_size:int,
  split: str,
  target_col: str = "Label",
  drop_cols=("Domain",),
  val_frac: float = 0.1,
  seed: int = 42,
  num_workers: int = 0,
  prefetch_factor: int = 2,
  pin_memory: bool = False,
):
  data = _load_tabular(csv_path, target_col, list(drop_cols), val_frac, seed)
  indices = data["train_idx"] if split == "train" else data["val_idx"]
  ds = TabularDataset(data["X"], data["labels"], indices= data["mean"], data["std"])
  
  rank = data_util.rank()
  sampler = DistributedSampler(ds, num_replicas=dist_util.world_size(), rank=rank, shuffle=True)
  loader = DataLoader(
    ds,
    batch_size=batch_size,
    drop_last = (split=="train"),
    worker_init_fn = partial(worker_init_fn, rank=rank),
    sampler = sampler,
    num_workers = num_workers,
    prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    pin_memory = pin_memory,
    persistent_workers = True if num_workers > 0 else False,
  )
  
  mean_t = torch.from_numpy(data["mean"]).float()
  std_t = torch.from_numpy(data["std"]).float()

  def preprocess_fn(batch, generator=None):
    del generator
    x, label = batch
    return {"images": _as_tensor(x).float(), "labels": _as_tensor(label).long()}

  def postprocess_fn(samples):
    s = samples.detach().float()
    m = mean_t.to(s.device).view(1, 1, -1)
    sd = std_t.to(s.device).view(1, 1, -1)
    return s * sd + m

  return loader, preprocess_fn, postprocess_fn

def get_tabular_postprocess_fn(*, csv_path, target_col="Label", drop_cols=("Domain",), val_frac=0.1, seed=42):
  data = _load_tabular(csv_path, target_col, list(drop_cols), val_frac, seed)
  mean_t = torch.from_numpy(data["mean"]).float()
  std_t = torch.from_numpy(data["std"]).float()

  def postprocess(samples):
    s = samples.detach().float()
    m = mean_t.to(s.device).view(1, 1, -1)
    sd = std_t.to(s.device).view(1, 1, -1)
    return s * sd + m
  return postprocess

  
    











    
    
