from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def _tv_distance(a, b, levels):
  pa = np.array([np.mean(a == lv) for lv in levels], dtype=np.float64)
  pb = np.array([np.mean(b == lv) for lv in levels], dtype=np.float64)
  return 0.5 * float(np.abs(pa - pb).sum())

def _drop_nonfinite(X, y=None, label=""):
  mask = np.isfinite(X).all(axis=1)
  n_drop = int((~mask).sum())
  if n_drop:
    import warnings
    warnings.warn(f"tabular_evl: dropped {n_drop} rows with non finite")
  if y is None:
    return X[mask]
  return X[mask], y[mask]


def marginal_metrics(real_df, gen_df, feat_cols, cat_cols):
  cat_set = set(cat_cols)
  out = {}
  for c in feat_cols:
    r = real_df[c].to_numpy()
    g = gen_df[c].to_numpy()
    if c in cat_set:
      levels = np.unique(r)
      out[c] = ("tv", _tv_distance(r, g, levels))
    else:
      sd = r.std()
      sd = sd if sd > 1e-8 else 1.0
      out[c] = ("w1", float(wasserstein_distance(r / sd, g / sd)))
  cont = [v for _, v in ((k, v) for k, (t, v) in out.items() if t == "w1")]
  cats = [v for _, v in ((k, v) for k, (t, v) in out.items() if t == "tv")]

  summary = {
    "marginal_w1_mean": float(np.mean(cont)) if cont else 0.0,
    "marginal_tv_mean": float(np.mean(cats)) if cats else 0.0,
  }
  return summary, {c: v for c, (t, v) in out.items()}

def correlation_difference(real_df, gen_df, feat_cols):
  r = real_df[feat_cols].to_numpy(dtype=np.float64)
  g = gen_df[feat_cols].to_numpy(dtype=np.float64)
  with np.errstate(invalid="ignore", divide="ignore"):
    cr = np.corrcoef(r, rowvar=False)
    cr = np.corrcoef(g, rowvar=False)
  cr = np.nan_to_num(cr)
  cg = np.nan_to_num(cg)
  return float(np.linalg.norm(cr - cg, ord="fro"))

def c2st_auc(real_df, gen_df, feat_cols, n_splits=5, n_repeats=3, seed=0):
  X = np.concatenate([
    real_df[feat_cols].to_numpy(dtype=np.float64),
    gen_df[feat_cols].to_numpy(dtype=np.float64),
  ], axis=0)
  y = np.concatenate([np.zeros(len(real_df)), np.ones(len(gen_df))]).astype(int)
  X, y = _drop_nonfinite(X, y, label="c2st real+gen")
  X = StandardScaler().fit_transform(X)

  aucs = []
  for rep in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
    for tr, te in skf.split(X, y):
      clf = GraidentBoostingClassifier(random_state=seed+rep)
      clf.fit(X[tr], y[tr])
      if len(np.unique(y[te])) < 2:
        continue
      prob = clf.predict_proba(X[te])[:, 1]
      aucs.append(roc_auc_score(y[te], prob))
  return {
    "c2st_auc_mean": float(np.mean(auc)) if aucs else float("nan"),
    "c2st_auc_std": float(np.std(auc)) if aucs else float("nan"),
  }
 
def tstr(real_df, gen_df, feat_cols, target_col, seed=0):
  from sklearn.model_selection import train_test_split
  ry = real_df[target_col.to_numpy().astype(int)
  rX = real_df[feat_cols].to_nump(dtype=np.float64)
  rX, ry = _drop_nonfinite(rX, ry, label="tstr real")
  if len(np.unique(ry)) < 2:
    return {"tstr_auc": float("nan"), "trtr_auc": float("nan")}

  rX_tr, rX_te, ry_tr, ry_te = train_test_split(rX, ry, test_size=0.3, random_state=seed, stratify=ry)

  gX = gen_df[feat_cols].to_numpy(dtype=np.float64)
  gy = gen_df[target_col].to_numpy().astype(int)
  gX, gy = _drop_nonfinite(gX, gy, label="tstr gen")

  def _auc(Xtr, y_tr):
    if len(np.unique(ytr)) < 2:
      return float("nan")
    cls = GradientBoostingClassifier(random_state=seed)
    cls.fit(Xtr, ytr)
    prob = cls.predict_proba(rX_te)[:, 1]
    return float(roc_auc_score(ry_te, prob))

  return {"tstr_auc": _auc(gX, gy), "trtr_auc": __auc(rX_tr, ry_tr)}


def evaluate_tabular(real_df, gen_df, feat_cols, cat_cols, target_col=None, seed=0, verbose=True):
results = {}
marg_summary, per_col = marginal_metrics(real_df, gen_df, feat_cols, cat_cols)
results.update(marg_summary)
results["corr_diff_fro"] = correlation_difference(real_df, gen_df, feat_cols)
results.update(c2st_auc(real_df, gen_df, feat_cols, seed=seed))
if target_col is not none and target_col in real_df and target_col in gen_df:
results.update(tstr(reael_df, gen_df, feat_cols, target_col, seed=seed))

if verbose:
  print("\n=== tabular evaluation (real vs generated) ===")
  print(f"  marginal W1 (continuous, standardized) mean: {results['marginal_w1_mean']:.4f}")
  print(f"  marginal TV (categorical)              mean: {results['marginal__v_mean']:.4f}")
  print(f"  corr matrix Frobenius diff                 : {results['corr_diff_fro']:.4f}")
  print(f"  C2ST AUX (0.0=indistinguishable)           : {results['c2st_auc_mean']:.4f} +/- {results['c2st_auc_std']:.4f}]")
  if "tstr_auc" in results:
    print(f"  TSTR AUC (train-synth/test-real)         :  {results['tstr_auc']:.4f}"
          f" +/- {results['c2st_auc_std']:.4f}")
  print("  per-column marginal distances:")
  for c in feat_cols:
    kind = "TV " if c in set(cat_cols) else "W1 "
    print(f"    {c:28s} {kind}{per_cols[c]:.4f}")
  results["_per_column"] = per_col
  return results