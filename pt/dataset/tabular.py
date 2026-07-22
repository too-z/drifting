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


def _build_features(X_full, train_idx, feat_cols, categorical_cols):
  cat_set = set(categorical_cols)
  train_X = X_full[train_idx]
  features = []
  for j, col in enumerate(feat_cols):
    if col in cat_set:
      levels = np.unique(X_full[~np.isnan(X_full[:, j]), j])
      levels = sorted(float(v) for v in levels)
      features.append({
        "name": col, "col": j, "kind": "cat", "levels": levels, "width": len(levels),})
    else:
      with np.errstate(invalid="ignore"):
        m = np.nanmean(train_X[:, j])
        s = np.nanstd(train_X[:, j])
      m = float(m) if np.isfinite(m) else 0.0
      s = float(s) if (np.isfinite(s) and s >= 1e-6) else 1.0
      features.append({
        "name": col, "col": j, "kind": "cont", "mean": m, "std": s, "width": 1,})
  return features

def _encode(X, features):
  parts = []
  for f in features:
    v = X[:, f["col"]]
    if f["kind"] == "cont":
      vv = np.where(np.isnan(v), f["mean"], v)
      parts.append(((vv - f["mean"]) / f["std"]).reshape(-1, 1).astype(np.float32))
    else:
      oh = np.zeros((len(v), f["width"]), dtype=np.float32)
      for li, lev in enumerate(f["levels"]):
        oh[:, li] = (v == lev)
      parts.append(oh)
  return np.concatenate(parts, axis=1).astype(np.float32)

def _decode(flat, features, cat_temperature=0.0, rng=None):
  flat = np.asarray(flat, dtype=np.float32)
  flat = flat.reshape(-1, flat.shape[-1])
  out = np.empty((flat.shape[0], len(features)), dtype=np.float32)
  if rng is None:
    rng = np.random.default_rng()
  off = 0
  for k, f in enumerate(features):
    w = f["width"]
    sl = flat[:, off:off+w]
    off += w
    if f["kind"] == "cont":
      out[:, k] = sl[:, 0] * f["std"] + f["mean"]
    else:
      if cat_temperature and cat_temperature > 0:
        logits = sl / float(cat_temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        u = rng.random((p.shape[0], 1))
        idx = (u > cdf).sum(axis=1)
        idx = np.clip(idx, 0, w - 1)
      else:
        idx = sl.argmax(axis=1)
      levels = np.asarray(f["levels"], dtype=np.float32)
      out[:, k] = levels[idx]
  return out

def _load_tabular(csv_path, target_col, drop_cols, val_frac, seed, categorical_cols=()):
  csv_path = _resolve_csv_path(csv_path)
  key = (os.path.abspath(csv_path), target_col, tuple(sorted(drop_cols)), float(val_frac), int(seed), tuple(sorted(categorical_cols)),)
  if key in _tabular_cache:
    return _tabular_cache[key]

  df = pd.read_csv(csv_path)
  if target_col not in df.columns:
    raise ValueError("target col not found")

  feat_cols = [c for c in df.columns if c != target_col and c not in set(drop_cols)]
  missing_cats = [c for c in categorical_cols if c not in feat_cols]
  if missing_cats:
    raise ValueError(f"cat cols not among features: {missing_cats}")
  X = df[feat_cols].to_numpy().astype(np.float32)
  labels = df[target_col].to_numpy().astype(np.int64)

  n = X.shape[0]
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n)
  n_val = min(n - 1, max(1, int(round(n * val_frac)))) if n > 1 else 0
  val_idx = perm[:n_val]
  train_idx = perm[n_val:]

  features = _build_features(X, train_idx, feat_cols, categorical_cols)
  feature_dims = [f["width"] for f in features]
  feature_kinds = [f["kind"] for f in features]
  X_enc = _encode(X, features)

  n_missing = int(np.isnan(X).sum())

  result = {
    "X": X,
    "X_enc": X_enc,  
    "labels": labels,
    "feat_cols": feat_cols,
    "features": features,
    "feature_dims": feature_dims,
    "feature_kinds": feature_kinds,
    "data_dim": int(sum(feature_dims)),
    "n_features": len(feat_cols),
    "num_classes": int(labels.max()) + 1,
    "train_idx": train_idx,
    "val_idx": val_idx,
    "n_missing": n_missing,
  }
  _tabular_cache[key] = result
  return result

def get_tabular_schema(*, csv_path, target_col="Label", drop_cols=("Domain",), val_frac=0.1, seed=42, categorical_cols=(), **_ignored):
  data = _load_tabular(csv_path, target_col, list(drop_cols), val_frac, seed, tuple(categorical_cols))
  return {
    "feature_dims": list(data["feature_dims"]),
    "feature_kinds": list(data["feature_kinds"]),
    "features": data["features"],
    "data_dim": data["data_dim"],
    "feat_cols": list(data["feat_cols"]),
    "num_classes": data["num_classes"],
  }

class TabularDataset(Dataset):
  def __init__(self, X_enc, labels, indices):
    indices = np.asarray(indices)
    self.indices = indices
    # self.X = ((np.asarray(X_enc)[indices] - mean) / std).astype(np.float32)
    self.X = np.asarray(X_enc)[indices].astype(np.float32)
    self.labels = np.asarray(labels)[indices]

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, i:int):
    x = torch.from_numpy(np.ascontiguousarray(self.X[i])).float().unsqueeze(0) # [1, D]
    y = int(self.labels[i])
    return x, y

def create_tabular_split(
  *,
  csv_path: str,
  batch_size:int,
  split: str,
  target_col: str = "Label",
  drop_cols=("Domain",),
  categorical_cols=(),
  val_frac: float = 0.1,
  seed: int = 42,
  num_workers: int = 0,
  prefetch_factor: int = 2,
  pin_memory: bool = False,
):
  data = _load_tabular(csv_path, target_col, list(drop_cols), val_frac, seed, tuple(categorical_cols))
  indices = data["train_idx"] if split == "train" else data["val_idx"]
  ds = TabularDataset(data["X_enc"], data["labels"], indices)
  
  rank = dist_util.rank()
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
  
  features = data["features"]

  def preprocess_fn(batch, generator=None):
    del generator
    x, label = batch
    return {"images": _as_tensor(x).float(), "labels": _as_tensor(label).long()}

  def postprocess_fn(samples):
    return torch.from_numpy(_decode(samples.detach().cpu().numpy(), features)).float()
  
  return loader, preprocess_fn, postprocess_fn

def get_tabular_postprocess_fn(*, csv_path, target_col="Label", drop_cols=("Domain",), val_frac=0.1, seed=42, categorical_cols=(), cat_temperature=0.0, decode_seed=None):
  data = _load_tabular(csv_path, target_col, list(drop_cols), val_frac, seed, tuple(categorical_cols))
  features = data["features"]
  rng = np.random.default_rng(decode_seed) if decode_seed is not None else None

  def postprocess(samples):
    return torch.from_numpy(_decode(samples.detach().cpu().numpy(), features, cat_temperature=cat_temperature, rng=rng,)).float()

  return postprocess

  
    











    
    

