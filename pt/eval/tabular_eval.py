from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_warned: set = set()

def _warn_once(msg):
  """The eval hook runs every eval_per_step; a per-message guard keeps the
  identical 'dropped N non-finite' notice from flooding the training log."""
  if msg not in _warned:
    _warned.add(msg)
    warnings.warn(msg)

def _tv_distance(a, b, levels):
  pa = np.array([np.mean(a == lv) for lv in levels], dtype=np.float64)
  pb = np.array([np.mean(b == lv) for lv in levels], dtype=np.float64)
  return 0.5 * float(np.abs(pa - pb).sum())

def _finite(v, label=""):
  """Drop non-finite entries of a 1-D column.

  The generator can never emit NaN (missing values are imputed at encode
  time), so a NaN in the real column would poison the whole distance. Compare
  the *observed* real values instead — that is the distribution the model was
  actually asked to fit.
  """
  v = np.asarray(v, dtype=np.float64)
  mask = np.isfinite(v)
  n_drop = int((~mask).sum())
  if n_drop:
    _warn_once(f"tabular_eval: {label}: dropped {n_drop} non-finite values")
  return v[mask]

def _drop_nonfinite(X, y=None, label=""):
  mask = np.isfinite(X).all(axis=1)
  n_drop = int((~mask).sum())
  if n_drop:
    _warn_once(f"tabular_eval: {label}: dropped {n_drop} rows with non finite")
  if y is None:
    return X[mask]
  return X[mask], y[mask]


# --------------------------------------------------------------------------- #
# shared encoding
# --------------------------------------------------------------------------- #

def _eval_columns(feat_cols, cat_cols, target_col=None, include_target=True):
  """Column list and categorical set used by every metric.

  TabSyn / TabDiff / TabCascade / TabNAT all treat the label as one more
  column of the table when scoring density, detection and privacy, so the
  default folds target_col in as a categorical column. Set include_target
  False to score the feature block alone.
  """
  cols = list(feat_cols)
  cats = set(cat_cols)
  if include_target and target_col is not None and target_col not in cols:
    cols.append(target_col)
    cats.add(target_col)
  return cols, cats


def _encode_frames(frames, cols, cat_cols, scale="zscore", cat_weight=None, fit_on=0):
  """Encode several tables into one aligned numeric space.

  Nominal columns become one-hot over the union of levels observed across
  *all* frames, so the resulting matrices are column-compatible and can be
  compared row-to-row (DCR needs synthetic, train and holdout in one space).
  Continuous scaling statistics come from frames[fit_on] — the real reference
  — so the encoding never borrows information from the synthetic table.

  scale: "zscore" for the classifier/kNN metrics, "minmax" for the distance
  based privacy metrics (DCR), matching what those papers normalize with.
  cat_weight: multiplier on the one-hot block. The default 1/sqrt(2) makes a
  category mismatch contribute exactly 1.0 of squared distance, so a nominal
  disagreement costs the same as a full-range numeric disagreement under
  minmax scaling.
  """
  if cat_weight is None:
    cat_weight = 1.0 / np.sqrt(2.0)
  cat_set = set(cat_cols)
  vals = [{c: np.asarray(f[c].to_numpy(), dtype=np.float64) for c in cols} for f in frames]
  parts = [[] for _ in frames]
  names = []
  for c in cols:
    if c in cat_set:
      seen = np.concatenate([v[c][np.isfinite(v[c])] for v in vals]) if vals else np.array([])
      for lv in np.unique(seen):
        for k, v in enumerate(vals):
          x = v[c]
          parts[k].append(np.where(np.isfinite(x), (x == lv).astype(np.float64), np.nan) * cat_weight)
        names.append(f"{c}={lv:g}")
    else:
      ref = vals[fit_on][c]
      obs = ref[np.isfinite(ref)]
      if scale == "minmax":
        lo = float(obs.min()) if obs.size else 0.0
        hi = float(obs.max()) if obs.size else 1.0
        den = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        for k, v in enumerate(vals):
          parts[k].append((v[c] - lo) / den)
      else:
        mu = float(obs.mean()) if obs.size else 0.0
        sd = float(obs.std()) if obs.size else 1.0
        sd = sd if sd > 1e-8 else 1.0
        for k, v in enumerate(vals):
          parts[k].append((v[c] - mu) / sd)
      names.append(c)
  mats = [np.stack(p, axis=1) if p else np.zeros((len(f), 0))
          for p, f in zip(parts, frames)]
  return mats, names


def _common_matrix(real_df, gen_df, cols, cat_cols, scale="zscore", cat_weight=None):
  """Two-frame shorthand for _encode_frames, fitted on the real table."""
  (R, G), names = _encode_frames([real_df, gen_df], cols, cat_cols, scale=scale,
                                 cat_weight=cat_weight, fit_on=0)
  return R, G, names


def _pairwise(A, B, chunk=2048):
  """Euclidean distances [len(A), len(B)], chunked over A to bound peak memory."""
  out = np.empty((len(A), len(B)), dtype=np.float64)
  for s in range(0, len(A), chunk):
    e = min(s + chunk, len(A))
    d = A[s:e, None, :] - B[None, :, :]
    out[s:e] = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
  return out


def _balanced(R, G, seed=0, n_max=None):
  """Subsample both blocks to a common size.

  C2ST, alpha-precision/beta-recall and density/coverage are all sensitive to
  the real:synthetic ratio, and every paper evaluates them at 1:1. n_gen is a
  free knob here (n_samples in the eval config), so without this the numbers
  would move whenever that knob moves.
  """
  n = min(len(R), len(G))
  if n_max is not None:
    n = min(n, int(n_max))
  rng = np.random.default_rng(seed)
  if len(R) > n:
    R = R[rng.choice(len(R), n, replace=False)]
  if len(G) > n:
    G = G[rng.choice(len(G), n, replace=False)]
  return R, G


# --------------------------------------------------------------------------- #
# 1. marginals
# --------------------------------------------------------------------------- #

def marginal_metrics(real_df, gen_df, feat_cols, cat_cols):
  """Per-column marginal distances.

  Continuous columns get both statistics the literature uses: W1 on
  std-normalized values (TabDDPM / CDTD / TabCascade) and the two-sample KS
  statistic (TabSyn / TabDiff / TabNAT). Categorical columns get TVD
  (TabSyn / TabDiff / TabNAT); JSD, used by TabDDPM / CDTD, is bounded by TVD
  and ranks methods the same way, so it is reported alongside for completeness.
  """
  cat_set = set(cat_cols)
  w1, ks, tv, jsd = {}, {}, {}, {}
  for c in feat_cols:
    r = _finite(real_df[c].to_numpy(), label=f"real[{c}]")
    g = _finite(gen_df[c].to_numpy(), label=f"gen[{c}]")
    if len(r) == 0 or len(g) == 0:
      if c in cat_set:
        tv[c] = jsd[c] = float("nan")
      else:
        w1[c] = ks[c] = float("nan")
      continue
    if c in cat_set:
      levels = np.unique(np.concatenate([r, g]))
      tv[c] = _tv_distance(r, g, levels)
      pa = np.array([np.mean(r == lv) for lv in levels])
      pb = np.array([np.mean(g == lv) for lv in levels])
      m = 0.5 * (pa + pb)
      with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.where(pa > 0, pa * np.log2(pa / m), 0.0)
        kl_b = np.where(pb > 0, pb * np.log2(pb / m), 0.0)
      jsd[c] = float(np.clip(0.5 * (kl_a.sum() + kl_b.sum()), 0.0, 1.0))
    else:
      sd = r.std()
      sd = sd if sd > 1e-8 else 1.0
      w1[c] = float(wasserstein_distance(r / sd, g / sd))
      ks[c] = float(ks_2samp(r, g).statistic)

  def _m(d):
    v = [x for x in d.values() if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")

  summary = {
    "marginal_w1_mean": _m(w1),
    "marginal_ks_mean": _m(ks),
    "marginal_tv_mean": _m(tv),
    "marginal_jsd_mean": _m(jsd),
  }
  # one headline number per column, in the unit that column is scored in
  per_col = {**w1, **tv}
  return summary, per_col, {"w1": w1, "ks": ks, "tv": tv, "jsd": jsd}


# --------------------------------------------------------------------------- #
# 2. pairwise dependence
# --------------------------------------------------------------------------- #

def _corr_ratio(x_cat, x_num):
  """Correlation ratio eta in [0, 1] for a nominal / continuous pair."""
  mask = np.isfinite(x_cat) & np.isfinite(x_num)
  x_cat, x_num = x_cat[mask], x_num[mask]
  if x_num.size < 2:
    return 0.0
  total = float(((x_num - x_num.mean()) ** 2).sum())
  if total <= 1e-12:
    return 0.0
  between = 0.0
  for lv in np.unique(x_cat):
    grp = x_num[x_cat == lv]
    if grp.size:
      between += grp.size * (grp.mean() - x_num.mean()) ** 2
  return float(np.sqrt(np.clip(between / total, 0.0, 1.0)))


def _cramers_v(a, b):
  """Bias-corrected Cramer's V in [0, 1] for a nominal / nominal pair."""
  mask = np.isfinite(a) & np.isfinite(b)
  a, b = a[mask], b[mask]
  if a.size < 2:
    return 0.0
  ct = pd.crosstab(a, b).to_numpy(dtype=np.float64)
  n = ct.sum()
  if n <= 0 or min(ct.shape) < 2:
    return 0.0
  row, col = ct.sum(axis=1, keepdims=True), ct.sum(axis=0, keepdims=True)
  exp = row @ col / n
  with np.errstate(divide="ignore", invalid="ignore"):
    chi2 = np.nansum(np.where(exp > 0, (ct - exp) ** 2 / exp, 0.0))
  phi2 = chi2 / n
  r, k = ct.shape
  # Bergsma correction; without it V is badly upward biased on the small
  # (n < 600) tables these datasets have.
  phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
  rc = r - (r - 1) ** 2 / (n - 1)
  kc = k - (k - 1) ** 2 / (n - 1)
  den = min(kc - 1, rc - 1)
  if den <= 1e-12:
    return 0.0
  return float(np.sqrt(np.clip(phi2c / den, 0.0, 1.0)))


def _association_matrix(df, cols, cat_set):
  """Mixed-type association matrix: Pearson (num,num), correlation ratio
  (num,cat), Cramer's V (cat,cat). This is the matrix TabDDPM and CDTD take
  the L2 norm of."""
  p = len(cols)
  M = np.eye(p, dtype=np.float64)
  vals = {c: np.asarray(df[c].to_numpy(), dtype=np.float64) for c in cols}
  for i in range(p):
    for j in range(i + 1, p):
      ci, cj = cols[i], cols[j]
      xi, xj = vals[ci], vals[cj]
      ic, jc = ci in cat_set, cj in cat_set
      if ic and jc:
        v = _cramers_v(xi, xj)
      elif ic:
        v = _corr_ratio(xi, xj)
      elif jc:
        v = _corr_ratio(xj, xi)
      else:
        m = np.isfinite(xi) & np.isfinite(xj)
        if m.sum() < 2 or xi[m].std() < 1e-12 or xj[m].std() < 1e-12:
          v = 0.0
        else:
          v = float(np.corrcoef(xi[m], xj[m])[0, 1])
          v = 0.0 if not np.isfinite(v) else v
      M[i, j] = M[j, i] = v
  return M


def association_difference(real_df, gen_df, cols, cat_cols):
  """|association(real) - association(synthetic)| over unique column pairs.

  Reports the pair-mean (what TabSyn / TabDiff / TabNAT tabulate), the
  Frobenius norm of the full matrix difference (TabDDPM / CDTD), and the
  num-num / num-cat / cat-cat breakdown (TabCascade reports mixed pairs
  separately). The pair-mean is the number to compare across datasets: the
  Frobenius norm grows like sqrt(#pairs), so it is not comparable between a
  10-column and a 25-column table.
  """
  cat_set = set(cat_cols)
  Mr = _association_matrix(real_df, cols, cat_set)
  Mg = _association_matrix(gen_df, cols, cat_set)
  D = np.abs(Mr - Mg)
  p = len(cols)
  iu = np.triu_indices(p, k=1)
  pairs = D[iu]
  kinds = np.array([
    ("cat_cat" if (cols[i] in cat_set and cols[j] in cat_set)
     else "num_num" if (cols[i] not in cat_set and cols[j] not in cat_set)
     else "num_cat")
    for i, j in zip(*iu)
  ])
  out = {
    "corr_diff_mean": float(pairs.mean()) if pairs.size else float("nan"),
    "corr_diff_max": float(pairs.max()) if pairs.size else float("nan"),
    # kept so runs logged before the mixed-type fix stay plottable
    "corr_diff_fro": float(np.linalg.norm(Mr - Mg, ord="fro")),
  }
  for k in ("num_num", "num_cat", "cat_cat"):
    sel = pairs[kinds == k]
    out[f"corr_diff_{k}"] = float(sel.mean()) if sel.size else float("nan")
  if pairs.size:
    a, b = iu[0][pairs.argmax()], iu[1][pairs.argmax()]
    out["corr_diff_argmax"] = f"{cols[a]}~{cols[b]}"
  return out


def trend_difference(real_df, gen_df, cols, cat_cols, n_bins=10):
  """SDMetrics-style pairwise "Trend": |rho_r - rho_s|/2 for numeric pairs,
  contingency TVD once either column is categorical (numeric partners
  discretized into quantile bins of the real column). This is the exact
  quantity TabSyn and TabDiff tabulate, so it is what their published tables
  can be read against."""
  cat_set = set(cat_cols)
  binned, iscat = {}, {}
  for c in cols:
    v = np.asarray(real_df[c].to_numpy(), dtype=np.float64)
    g = np.asarray(gen_df[c].to_numpy(), dtype=np.float64)
    if c in cat_set:
      binned[c] = (v, g)
      iscat[c] = True
    else:
      obs = v[np.isfinite(v)]
      qs = np.unique(np.quantile(obs, np.linspace(0, 1, n_bins + 1))) if obs.size else np.array([0.0])
      edges = qs[1:-1] if qs.size > 2 else np.array([])
      binned[c] = (np.searchsorted(edges, v).astype(np.float64),
                   np.searchsorted(edges, g).astype(np.float64))
      iscat[c] = False

  scores = []
  for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
      ci, cj = cols[i], cols[j]
      if not iscat[ci] and not iscat[cj]:
        xr = np.asarray(real_df[ci].to_numpy(), float); yr = np.asarray(real_df[cj].to_numpy(), float)
        xg = np.asarray(gen_df[ci].to_numpy(), float); yg = np.asarray(gen_df[cj].to_numpy(), float)
        mr = np.isfinite(xr) & np.isfinite(yr); mg = np.isfinite(xg) & np.isfinite(yg)
        if mr.sum() < 2 or mg.sum() < 2:
          continue
        with np.errstate(invalid="ignore", divide="ignore"):
          rr = np.corrcoef(xr[mr], yr[mr])[0, 1]
          rg = np.corrcoef(xg[mg], yg[mg])[0, 1]
        rr = 0.0 if not np.isfinite(rr) else rr
        rg = 0.0 if not np.isfinite(rg) else rg
        scores.append(abs(rr - rg) / 2.0)
      else:
        ar, ag = binned[ci]
        br, bg = binned[cj]
        mr = np.isfinite(ar) & np.isfinite(br); mg = np.isfinite(ag) & np.isfinite(bg)
        if mr.sum() == 0 or mg.sum() == 0:
          continue
        jr = pd.crosstab(ar[mr], br[mr], normalize=True)
        jg = pd.crosstab(ag[mg], bg[mg], normalize=True)
        jr, jg = jr.align(jg, fill_value=0.0)
        scores.append(0.5 * float(np.abs(jr.to_numpy() - jg.to_numpy()).sum()))
  return {"trend_diff_mean": float(np.mean(scores)) if scores else float("nan")}


def correlation_difference(real_df, gen_df, feat_cols):
  """Frobenius norm of the Pearson-only correlation difference.

  Retained for the pre-existing logged metric. Prefer
  association_difference: Pearson on integer-coded nominal columns is not a
  meaningful dependence measure, and 8 of the 13 Cleveland columns are
  nominal.
  """
  r = _drop_nonfinite(real_df[feat_cols].to_numpy(dtype=np.float64), label="corr real")
  g = _drop_nonfinite(gen_df[feat_cols].to_numpy(dtype=np.float64), label="corr gen")
  if len(r) < 2 or len(g) < 2:
    return float("nan")
  with np.errstate(invalid="ignore", divide="ignore"):
    cr = np.corrcoef(r, rowvar=False)
    cg = np.corrcoef(g, rowvar=False)
  cr = np.nan_to_num(cr)
  cg = np.nan_to_num(cg)
  return float(np.linalg.norm(cr - cg, ord="fro"))


# --------------------------------------------------------------------------- #
# 3. detection / C2ST
# --------------------------------------------------------------------------- #

def c2st_auc(real_df, gen_df, cols, cat_cols=(), n_splits=5, n_repeats=3, seed=0,
             clf="gb", balance=True):
  """Classifier two-sample test.

  Nominal columns are one-hot encoded before the critic sees them (an ordinal
  code invents an ordering the critic can exploit), and the two classes are
  subsampled to equal size so the AUC does not drift with n_samples.

  clf: "gb" (gradient boosting, the strong critic — comparable to the
  LightGBM detection score in CDTD / TabCascade) or "logreg" (the weak critic
  TabSyn / TabDiff use). A strong critic separates almost anything on a
  600-row table, so report whichever you use consistently across all methods.

  c2st_score = 1 - 2*|AUC - 0.5| in [0, 1], 1 = indistinguishable. Monotone in
  the detection scores those papers report, but confirm the normalization
  before putting it in the same column as a number copied from a paper.

  An AUC clearly *below* 0.5 is not a better-than-perfect generator: it means
  rows are shared verbatim between the real and synthetic tables, so a row the
  critic learned as "synthetic" in one fold reappears as "real" in another and
  the predictions invert. Read a sub-0.5 AUC together with dcr_zero_frac and
  authenticity, not as a fidelity win.
  """
  R, G, _ = _common_matrix(real_df, gen_df, cols, cat_cols, scale="zscore", cat_weight=1.0)
  if balance:
    R, G = _balanced(R, G, seed=seed)
  X = np.concatenate([R, G], axis=0)
  y = np.concatenate([np.zeros(len(R)), np.ones(len(G))]).astype(int)
  X, y = _drop_nonfinite(X, y, label="c2st real+gen")
  if len(np.unique(y)) < 2:
    return {"c2st_auc_mean": float("nan"), "c2st_auc_std": float("nan"),
            "c2st_acc_mean": float("nan"), "c2st_score": float("nan")}
  X = StandardScaler().fit_transform(X)

  aucs, accs = [], []
  for rep in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
    for tr, te in skf.split(X, y):
      if len(np.unique(y[te])) < 2:
        continue
      if clf == "logreg":
        model = LogisticRegression(max_iter=1000)
      else:
        model = GradientBoostingClassifier(random_state=seed + rep)
      model.fit(X[tr], y[tr])
      prob = model.predict_proba(X[te])[:, 1]
      aucs.append(roc_auc_score(y[te], prob))
      accs.append(accuracy_score(y[te], (prob >= 0.5).astype(int)))
  auc = float(np.mean(aucs)) if aucs else float("nan")
  return {
    "c2st_auc_mean": auc,
    "c2st_auc_std": float(np.std(aucs)) if aucs else float("nan"),
    "c2st_acc_mean": float(np.mean(accs)) if accs else float("nan"),
    "c2st_score": float(np.clip(1.0 - 2.0 * abs(auc - 0.5), 0.0, 1.0)) if aucs else float("nan"),
  }


# --------------------------------------------------------------------------- #
# 4. downstream utility
# --------------------------------------------------------------------------- #

def tstr(real_df, gen_df, feat_cols, target_col, real_test_df=None, cat_cols=(), seed=0):
  """TSTR / TRTR on a common test set.

  real_df is the split the generator was fit on. When real_test_df is given
  (the held-out split) both classifiers are scored on it, so neither the
  generator nor the TRTR baseline has seen the test rows. Without it we fall
  back to a 70/30 split of real_df — those test rows were in the generator's
  memory bank, which inflates TSTR.

  tstr_gap = trtr_auc - tstr_auc is the headline: MLE in TabSyn / TabDiff /
  TabCascade / TabNAT is reported as the gap to the real-data classifier, not
  as the absolute score.
  """
  from sklearn.model_selection import train_test_split
  cols = list(feat_cols)
  ry = real_df[target_col].to_numpy().astype(int)
  if len(np.unique(ry)) < 2:
    return {"tstr_auc": float("nan"), "trtr_auc": float("nan"),
            "tstr_gap": float("nan"), "tstr_n_test": 0}

  # one aligned encoding for train-real / train-synth / test-real, so the
  # three classifiers share a feature space and a nominal level ordering
  test_src = real_test_df if (real_test_df is not None and len(real_test_df) > 0) else None
  if test_src is None:
    _warn_once("tabular_eval: no held-out test split given; TSTR falls back to an in-sample split")
  frames = [real_df, gen_df] + ([test_src] if test_src is not None else [])
  mats, _ = _encode_frames(frames, cols, cat_cols, scale="zscore", cat_weight=1.0, fit_on=0)
  rX, gX = mats[0], mats[1]
  gy = gen_df[target_col].to_numpy().astype(int)

  if test_src is not None:
    rX_tr, ry_tr = rX, ry
    rX_te = mats[2]
    ry_te = test_src[target_col].to_numpy().astype(int)
  else:
    rX_tr, rX_te, ry_tr, ry_te = train_test_split(rX, ry, test_size=0.3, random_state=seed,
                                                  stratify=ry)

  rX_tr, ry_tr = _drop_nonfinite(rX_tr, ry_tr, label="tstr real train")
  rX_te, ry_te = _drop_nonfinite(rX_te, ry_te, label="tstr real test")
  gX, gy = _drop_nonfinite(gX, gy, label="tstr gen")
  if len(np.unique(ry_te)) < 2:
    return {"tstr_auc": float("nan"), "trtr_auc": float("nan"),
            "tstr_gap": float("nan"), "tstr_n_test": int(len(ry_te))}

  def _auc(Xtr, y_tr):
    if len(np.unique(y_tr)) < 2:
      return float("nan")
    cls = GradientBoostingClassifier(random_state=seed)
    cls.fit(Xtr, y_tr)
    prob = cls.predict_proba(rX_te)[:, 1]
    return float(roc_auc_score(ry_te, prob))

  a_syn, a_real = _auc(gX, gy), _auc(rX_tr, ry_tr)
  return {"tstr_auc": a_syn, "trtr_auc": a_real,
          "tstr_gap": float(a_real - a_syn) if np.isfinite(a_syn) and np.isfinite(a_real) else float("nan"),
          "tstr_n_test": int(len(ry_te))}


# --------------------------------------------------------------------------- #
# 5. fidelity / diversity
# --------------------------------------------------------------------------- #

def alpha_precision_beta_recall(real_df, gen_df, cols, cat_cols=(), seed=0, n_steps=30,
                                balance=True):
  """alpha-Precision / beta-Recall / Authenticity (Alaa et al., 2022).

  These are the three numbers TabSyn, TabDiff, TabCascade and TabNAT report
  under "Quality" — the metric that separates a model that covers the real
  support from one that only reproduces its dense core.

    alpha-Precision: sweep alpha over [0,1]; r_alpha is the alpha-quantile of
      real-to-centre distance. A faithful generator puts an alpha fraction of
      its samples inside r_alpha for every alpha, so the reported score is
      1 - 2*integral|alpha_curve - alpha|, i.e. 1 for a perfect match.
    beta-Recall: the same sweep run the other way — a real point counts as
      covered when its nearest synthetic neighbour is closer than its own
      nearest real neighbour and that neighbour sits inside the beta-support
      of the synthetic distribution.
    Authenticity: fraction of synthetic rows that are *not* near-copies. A row
      is a copy when it is closer to its nearest real row than that row is to
      its own nearest real neighbour. Low authenticity with high precision
      means memorization, so read it next to the DCR block.

  Computed in the standardized one-hot space rather than the one-class
  embedding the original paper (and synthcity) learns first. The curves are
  the same construction; absolute values are not directly comparable to a
  number copied out of a paper, so recompute every baseline with this code.
  """
  R, G, _ = _common_matrix(real_df, gen_df, cols, cat_cols, scale="zscore", cat_weight=1.0)
  R = _drop_nonfinite(R, label="alpha real")
  G = _drop_nonfinite(G, label="alpha gen")
  if balance:
    R, G = _balanced(R, G, seed=seed)
  nan = {"alpha_precision": float("nan"), "beta_recall": float("nan"),
         "authenticity": float("nan")}
  if len(R) < 3 or len(G) < 3:
    return nan

  alphas = np.linspace(0, 1, n_steps)
  centre = R.mean(axis=0)
  syn_centre = G.mean(axis=0)
  radii = np.quantile(np.linalg.norm(R - centre, axis=1), alphas)
  syn_to_centre = np.linalg.norm(G - centre, axis=1)

  nn_r = NearestNeighbors(n_neighbors=2).fit(R)
  real_to_real = nn_r.kneighbors(R)[0][:, 1]
  nn_g = NearestNeighbors(n_neighbors=1).fit(G)
  d_rg, idx_rg = nn_g.kneighbors(R)
  real_to_syn = d_rg[:, 0]
  closest_syn_to_syn_centre = np.linalg.norm(G[idx_rg[:, 0]] - syn_centre, axis=1)
  beta_radii = np.quantile(closest_syn_to_syn_centre, alphas)

  prec_curve = np.array([np.mean(syn_to_centre <= radii[k]) for k in range(n_steps)])
  rec_curve = np.array([
    np.mean((real_to_syn <= real_to_real) & (closest_syn_to_syn_centre <= beta_radii[k]))
    for k in range(n_steps)
  ])
  d_alpha = alphas[1] - alphas[0]
  a_prec = 1.0 - 2.0 * float(np.abs(alphas - prec_curve).sum()) * d_alpha
  b_rec = 1.0 - 2.0 * float(np.abs(alphas - rec_curve).sum()) * d_alpha

  # authenticity, per synthetic row (see docstring)
  d_gr, idx_gr = nn_r.kneighbors(G, n_neighbors=1)
  syn_to_real = d_gr[:, 0]
  host_nn = real_to_real[idx_gr[:, 0]]
  authenticity = float(np.mean(syn_to_real >= host_nn))

  return {"alpha_precision": float(a_prec), "beta_recall": float(b_rec),
          "authenticity": authenticity}


def density_coverage(real_df, gen_df, cols, cat_cols=(), k=5, seed=0, balance=True):
  """Density and Coverage (Naeem et al., 2020) — the improved-precision/recall
  pair TabDDPM reports. Density is 1.0 and coverage 1.0 for a perfect match;
  density above 1 means the generator over-concentrates on dense real regions.
  """
  R, G, _ = _common_matrix(real_df, gen_df, cols, cat_cols, scale="zscore", cat_weight=1.0)
  R = _drop_nonfinite(R, label="density real")
  G = _drop_nonfinite(G, label="density gen")
  if balance:
    R, G = _balanced(R, G, seed=seed)
  if len(R) < k + 1 or len(G) < 1:
    return {"density": float("nan"), "coverage": float("nan")}
  kk = min(k, len(R) - 1)
  nn_r = NearestNeighbors(n_neighbors=kk + 1).fit(R)
  radius = nn_r.kneighbors(R)[0][:, kk]           # kth-NN radius of each real point
  D = _pairwise(G, R)                              # [n_gen, n_real]
  inside = D <= radius[None, :]
  density = float(inside.sum() / (kk * len(G)))
  coverage = float(np.mean(inside.any(axis=0)))
  return {"density": density, "coverage": coverage}


# --------------------------------------------------------------------------- #
# 6. privacy
# --------------------------------------------------------------------------- #

def dcr_metrics(real_df, gen_df, cols, cat_cols=(), real_holdout_df=None, seed=0,
                n_repeats=5):
  """Distance to Closest Record.

  Two readings, because the papers report two different things under "DCR":

  dcr_share (TabSyn / TabDiff / TabCascade "DCR Share"): sample a reference
    subset of the training rows the size of the holdout, then ask, for each
    synthetic row, whether its nearest real row falls in the training half or
    the holdout half. 0.5 is ideal — the generator is no closer to the rows it
    saw than to rows it did not. Above 0.5 is memorization.

    Those papers run this on a 50/50 real split, which is what makes it sharp.
    With val_frac=0.1 the train reference has to be subsampled 10:1 to stay
    size-matched, so a memorized row only registers when the row it copied
    lands in the subsample: the share is pulled toward 0.5 by roughly
    n_holdout/n_train and a blatant copy generator scores ~0.55 instead of
    ~1.0. Averaging over n_repeats reduces the variance but cannot restore the
    lost power. For the paper's privacy table, regenerate with val_frac=0.5;
    on a 90/10 split read dcr_ratio_p5 and dcr_zero_frac instead.

  dcr_ratio_median / dcr_ratio_p5 (TabDDPM-style): median and 5th-percentile
    synthetic-to-train DCR divided by the same statistic for holdout-to-train.
    Both sides use the full training set as reference, so the size asymmetry
    cancels and this stays fully powered at val_frac=0.1. 1.0 means synthetic
    rows sit as far from the training data as genuinely fresh real rows do;
    well below 1.0 means the generator is hugging the training set. The 5th
    percentile is the one that matters — a handful of exact copies is the
    privacy failure, and the median hides them.

    The ratio is undefined when the holdout baseline is itself 0, which
    happens on indian_liver because the raw CSV contains duplicated patient
    records that straddle the split. dcr_p5_synth / dcr_p5_holdout are
    reported raw for that case.

  dcr_zero_frac: fraction of synthetic rows that exactly coincide with a
    training row. Note this interacts with the decoder: cont_transform=
    "quantile" inverts through the training ECDF and decode_round snaps onto
    the observed value grid, so continuous columns can only take values the
    training split took. Exact row-level collisions are therefore expected at
    a low rate even without memorization, and are more likely on the
    low-cardinality Cleveland columns. Report the decoder settings next to
    this number.

  Distances are L2 on min-max scaled continuous columns plus one-hot nominal
  columns weighted so that a category mismatch costs 1.0, matching the
  normalization those papers use.
  """
  nan = {"dcr_share": float("nan"), "dcr_synth_median": float("nan"),
         "dcr_holdout_median": float("nan"), "dcr_ratio_median": float("nan"),
         "dcr_ratio_p5": float("nan"), "dcr_zero_frac": float("nan"),
         "dcr_p5_synth": float("nan"), "dcr_p5_holdout": float("nan")}
  has_ho = real_holdout_df is not None and len(real_holdout_df) >= 2
  # all three tables in one shared one-hot / minmax space, scaled on train
  frames = [real_df, gen_df] + ([real_holdout_df] if has_ho else [])
  mats, _ = _encode_frames(frames, cols, cat_cols, scale="minmax", fit_on=0)
  Rtr = _drop_nonfinite(mats[0], label="dcr train")
  G = _drop_nonfinite(mats[1], label="dcr gen")
  if len(Rtr) < 2 or len(G) < 1:
    return nan

  d_syn_full = _pairwise(G, Rtr)
  d_syn_tr = d_syn_full.min(axis=1)
  out = dict(nan)
  out["dcr_synth_median"] = float(np.median(d_syn_tr))
  out["dcr_p5_synth"] = float(np.quantile(d_syn_tr, 0.05))
  out["dcr_zero_frac"] = float(np.mean(d_syn_tr <= 1e-9))

  if not has_ho:
    _warn_once("tabular_eval: no holdout split given; DCR share/ratio unavailable")
    return out

  Rho = _drop_nonfinite(mats[2], label="dcr holdout")
  if len(Rho) < 2:
    return out

  # a holdout row's distance to the training set, excluding nothing: the
  # holdout is disjoint from train by construction
  d_ho_min = _pairwise(Rho, Rtr).min(axis=1)
  med_h = float(np.median(d_ho_min))
  p5_h = float(np.quantile(d_ho_min, 0.05))
  out["dcr_holdout_median"] = med_h
  out["dcr_p5_holdout"] = p5_h
  out["dcr_ratio_median"] = float(np.median(d_syn_tr) / med_h) if med_h > 1e-12 else float("nan")
  if p5_h > 1e-12:
    out["dcr_ratio_p5"] = float(out["dcr_p5_synth"] / p5_h)
  else:
    _warn_once("tabular_eval: holdout 5th-pct DCR is 0 (duplicated real rows across the "
               "split); dcr_ratio_p5 undefined, read dcr_p5_synth / dcr_zero_frac instead")

  # DCR share, on size-matched train / holdout references
  n_ref = min(len(Rho), len(Rtr))
  if len(Rtr) > 1.5 * n_ref:
    _warn_once(f"tabular_eval: DCR share is underpowered at this split "
               f"(n_train={len(Rtr)} vs n_holdout={n_ref}); it is pulled toward 0.5 by "
               f"~{n_ref / len(Rtr):.2f}. Use val_frac=0.5 for the reported privacy table.")
  d_syn_ho = _pairwise(G, Rho).min(axis=1)
  rng = np.random.default_rng(seed)
  shares = []
  for _ in range(int(n_repeats)):
    sub = rng.choice(len(Rtr), n_ref, replace=False) if len(Rtr) > n_ref else np.arange(len(Rtr))
    d_tr = d_syn_full[:, sub].min(axis=1)
    ho = (_pairwise(G, Rho[rng.choice(len(Rho), n_ref, replace=False)]).min(axis=1)
          if len(Rho) > n_ref else d_syn_ho)
    tie = np.isclose(d_tr, ho)
    shares.append(float(np.mean(np.where(tie, 0.5, (d_tr < ho).astype(np.float64)))))
  out["dcr_share"] = float(np.mean(shares))
  return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def evaluate_tabular(real_df, gen_df, feat_cols, cat_cols, target_col=None, real_test_df=None,
                     seed=0, verbose=True, c2st_splits=5, c2st_repeats=3, c2st_clf="gb",
                     include_target=True, quality=True, privacy=True, legacy_corr=True,
                     dcr_repeats=5, density_k=5):
  """All seven metric families, on one call.

  include_target folds the label into the joint metrics as a categorical
  column, which is what every paper in the comparison does.
  quality / privacy gate the alpha-precision and DCR blocks; both are cheap
  kNN computations, so they stay on by default in the training hook too.
  """
  results = {}
  cols, cat_set = _eval_columns(feat_cols, cat_cols, target_col, include_target)
  has_target = target_col is not None and target_col in real_df and target_col in gen_df
  if include_target and not has_target:
    cols, cat_set = _eval_columns(feat_cols, cat_cols, None, False)

  marg_summary, per_col, per_kind = marginal_metrics(real_df, gen_df, cols, cat_set)
  results.update(marg_summary)

  results.update(association_difference(real_df, gen_df, cols, cat_set))
  results.update(trend_difference(real_df, gen_df, cols, cat_set))
  if legacy_corr:
    results["corr_diff_fro_pearson_only"] = correlation_difference(real_df, gen_df, feat_cols)

  results.update(c2st_auc(real_df, gen_df, cols, cat_cols=cat_set, n_splits=c2st_splits,
                          n_repeats=c2st_repeats, seed=seed, clf=c2st_clf))

  if has_target:
    results.update(tstr(real_df, gen_df, feat_cols, target_col,
                        real_test_df=real_test_df, cat_cols=cat_cols, seed=seed))

  if quality:
    results.update(alpha_precision_beta_recall(real_df, gen_df, cols, cat_cols=cat_set, seed=seed))
    results.update(density_coverage(real_df, gen_df, cols, cat_cols=cat_set, k=density_k, seed=seed))

  if privacy:
    results.update(dcr_metrics(real_df, gen_df, cols, cat_cols=cat_set,
                               real_holdout_df=real_test_df, seed=seed, n_repeats=dcr_repeats))

  results["_per_column"] = per_col
  results["_per_column_by_kind"] = per_kind
  results["_eval_columns"] = list(cols)

  if verbose:
    _print_report(results, cols, cat_set, per_kind, len(real_df), len(gen_df))
  return results


def _fmt(v, w=7, p=4):
  return f"{v:{w}.{p}f}" if isinstance(v, float) and np.isfinite(v) else f"{'nan':>{w}}"


def _print_report(res, cols, cat_set, per_kind, n_real, n_gen):
  g = res.get
  print("\n=== tabular evaluation (real vs generated) ===")
  print(f"  n_real={n_real} n_gen={n_gen} n_test={g('tstr_n_test', 0)} "
        f"n_cols={len(cols)} ({len(cat_set)} categorical)")
  print("  -- marginal (lower better) ------------------------------------")
  print(f"    W1  (continuous, std-normalized) mean : {_fmt(g('marginal_w1_mean'))}")
  print(f"    KS  (continuous)                 mean : {_fmt(g('marginal_ks_mean'))}")
  print(f"    TVD (categorical)                mean : {_fmt(g('marginal_tv_mean'))}")
  print(f"    JSD (categorical)                mean : {_fmt(g('marginal_jsd_mean'))}")
  print("  -- pairwise dependence (lower better) ------------------------")
  print(f"    |assoc| diff, pair mean               : {_fmt(g('corr_diff_mean'))}"
        f"   (num-num {_fmt(g('corr_diff_num_num'), 6)}"
        f" num-cat {_fmt(g('corr_diff_num_cat'), 6)}"
        f" cat-cat {_fmt(g('corr_diff_cat_cat'), 6)})")
  print(f"    |assoc| diff, worst pair              : {_fmt(g('corr_diff_max'))}"
        f"   ({g('corr_diff_argmax', '-')})")
  print(f"    Frobenius norm (mixed assoc matrix)   : {_fmt(g('corr_diff_fro'))}")
  print(f"    Trend diff (SDMetrics style)          : {_fmt(g('trend_diff_mean'))}")
  print("  -- detection (0.5 AUC = indistinguishable) -------------------")
  print(f"    C2ST AUC                              : {_fmt(g('c2st_auc_mean'))}"
        f" +/- {_fmt(g('c2st_auc_std'), 6)}")
  print(f"    C2ST accuracy / score (1 = ideal)     : {_fmt(g('c2st_acc_mean'))}"
        f" / {_fmt(g('c2st_score'), 6)}")
  auc = g("c2st_auc_mean")
  if isinstance(auc, float) and np.isfinite(auc) and auc < 0.45:
    print("      note: AUC << 0.5 means real and synthetic share verbatim rows,"
          " not that fidelity is perfect")
  if "tstr_auc" in res:
    print("  -- downstream utility (gap -> 0 better) ----------------------")
    print(f"    TSTR AUC (train synth / test real)    : {_fmt(g('tstr_auc'))}")
    print(f"    TRTR AUC (train real  / test real)    : {_fmt(g('trtr_auc'))}")
    print(f"    gap (TRTR - TSTR)                     : {_fmt(g('tstr_gap'))}")
  if "alpha_precision" in res:
    print("  -- fidelity / diversity (1 = ideal) --------------------------")
    print(f"    alpha-Precision                       : {_fmt(g('alpha_precision'))}")
    print(f"    beta-Recall                           : {_fmt(g('beta_recall'))}")
    print(f"    Authenticity (1 = no near-copies)     : {_fmt(g('authenticity'))}")
    print(f"    Density / Coverage (1 / 1 = ideal)    : {_fmt(g('density'))}"
          f" / {_fmt(g('coverage'), 6)}")
  if "dcr_share" in res:
    print("  -- privacy ---------------------------------------------------")
    print(f"    DCR share  (0.5 = ideal, >0.5 leak)   : {_fmt(g('dcr_share'))}")
    print(f"    DCR median synth / holdout            : {_fmt(g('dcr_synth_median'))}"
          f" / {_fmt(g('dcr_holdout_median'), 6)}")
    print(f"    DCR ratio  median / 5th pct (1=ideal) : {_fmt(g('dcr_ratio_median'))}"
          f" / {_fmt(g('dcr_ratio_p5'), 6)}")
    print(f"    DCR 5th pct raw synth / holdout       : {_fmt(g('dcr_p5_synth'))}"
          f" / {_fmt(g('dcr_p5_holdout'), 6)}")
    print(f"    exact-duplicate fraction              : {_fmt(g('dcr_zero_frac'))}")
  print("  -- per-column marginal distances -----------------------------")
  for c in cols:
    if c in cat_set:
      print(f"    {c:28s} TVD {_fmt(per_kind['tv'].get(c, float('nan')), 6)}"
            f"  JSD {_fmt(per_kind['jsd'].get(c, float('nan')), 6)}")
    else:
      print(f"    {c:28s} W1  {_fmt(per_kind['w1'].get(c, float('nan')), 6)}"
            f"  KS  {_fmt(per_kind['ks'].get(c, float('nan')), 6)}")
