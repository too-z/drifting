from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from pt.dataset.tabular import _load_tabular, get_tabular_postprocess_fn
from pt.utils.init_util import load_generator_model_and_params
from pt.utils.misc import load_config
from pt.utils.rng import make_generator

@torch.no_grad()
def generate(artifact, config_path, n, cfg_scale, seed, out_csv, cat_temperature=0.0, do_eval=True,
             real_split="train", decode_clip=False, decode_round=False, metrics_out="",
             c2st_clf="gb", c2st_repeats=3, include_target=True):
  config = load_config(config_path)
  ds_kwargs = dict(config.dataset.get("kwargs", {}))
  csv_path = ds_kwargs["csv_path"]
  target_col = ds_kwargs.get("target_col", "Label")
  drop_cols = list(ds_kwargs.get("drop_cols", ["Domain"]))
  categorical_cols = list(ds_kwargs.get("categorical_cols", []))
  cont_transform = str(ds_kwargs.get("cont_transform", "zscore"))
  val_frac = float(ds_kwargs.get("val_frac", 0.1))
  ds_seed = int(ds_kwargs.get("seed", 42))

  data = _load_tabular(csv_path, target_col, drop_cols, val_frac, ds_seed,
                       tuple(categorical_cols), cont_transform)
  feat_cols = data["feat_cols"]
  num_classes = data["num_classes"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, metadata = load_generator_model_and_params(artifact)
  model = model.to(device).eval()
  print(f"step={metadata.get('step')} tabular={getattr(model, 'tabular', False)} "
        f"cont_transform={cont_transform} real_split={real_split}")

  cat_softmax = bool(config.model.get("cat_softmax", False))
  postprocess = get_tabular_postprocess_fn(
    csv_path=csv_path, target_col=target_col, drop_cols=drop_cols, val_frac=val_frac,
    seed=ds_seed, categorical_cols=tuple(categorical_cols), cont_transform=cont_transform,
    cat_temperature=cat_temperature, cat_softmax=cat_softmax, decode_seed=seed,
    clip=decode_clip, round_grid=decode_round,
  )

  # The generator only ever saw the train split (memory banks are filled from
  # the train loader), so that is the reference distribution for fidelity
  # metrics; the val split stays held out as the TSTR/TRTR test set.
  ref_idx = {"train": data["train_idx"], "val": data["val_idx"],
             "all": np.arange(len(data["labels"]))}[real_split]
  real_labels = data["labels"][ref_idx]
  p = np.bincount(real_labels, minlength=num_classes) / len(real_labels)
  rng = np.random.default_rng(seed)
  labels = torch.from_numpy(rng.choice(num_classes, size=n, p=p)).long().to(device)

  g = make_generator(device, "generate-tabular", seed)
  samples = model(labels, cfg_scale=float(cfg_scale), generator=g)["samples"] # [n, 1, data_dim]
  table = postprocess(samples).squeeze(1).cpu().numpy() # [n, n_features], decoded

  gen_df = pd.DataFrame(table, columns=feat_cols)
  gen_df[target_col] = labels.cpu().numpy()
  if out_csv:
    gen_df.to_csv(out_csv, index=False)
    print(f"wrote {len(gen_df)} generated rows -> {out_csv}")

  real_df = pd.DataFrame(data["X"][ref_idx], columns=feat_cols)
  real_df[target_col] = real_labels
  val_df = pd.DataFrame(data["X"][data["val_idx"]], columns=feat_cols)
  val_df[target_col] = data["labels"][data["val_idx"]]

  print("marginal comparison real vs generated")
  print("column real_mean gen_mean real_std gen_std")
  for c in feat_cols:
    print(f"{c:28s} {real_df[c].mean():10.3f} {gen_df[c].mean():10.3f}"
          f"{real_df[c].std():9.3f} {gen_df[c].std():9.3f}")
  results = None
  if do_eval:
    from pt.eval.tabular_eval import evaluate_tabular
    results = evaluate_tabular(real_df, gen_df, feat_cols, cat_cols=categorical_cols,
                               target_col=target_col,
                               real_test_df=None if real_split == "val" else val_df,
                               seed=seed, verbose=True, c2st_clf=c2st_clf,
                               c2st_repeats=c2st_repeats, include_target=include_target)
    if metrics_out:
      payload = {k: v for k, v in results.items() if not k.startswith("_")}
      payload["_per_column"] = results.get("_per_column", {})
      payload["_per_column_by_kind"] = results.get("_per_column_by_kind", {})
      payload["_eval_columns"] = results.get("_eval_columns", [])
      payload["config"] = {"artifact": artifact, "config": config_path, "n": n,
                           "cfg": cfg_scale, "seed": seed, "cont_transform": cont_transform,
                           "cat_temperature": cat_temperature, "real_split": real_split,
                           "decode_clip": decode_clip, "decode_round": decode_round,
                           "c2st_clf": c2st_clf, "include_target": include_target,
                           "val_frac": val_frac, "step": metadata.get("step")}
      with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
      print(f"wrote metrics -> {metrics_out}")
  return gen_df, results

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--artifact", required=True, help="run dir or params_ema dir")
  ap.add_argument("--config", required=True)
  ap.add_argument("--n", type=int, default=400)
  ap.add_argument("--cfg", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default="")
  ap.add_argument("--cat-temp", type=float, default=0.0)
  ap.add_argument("--real-split", default="train", choices=["train", "val", "all"],
                  help="real reference split for the fidelity metrics")
  ap.add_argument("--decode-clip", action="store_true",
                  help="clip decoded continuous values to the observed training range")
  ap.add_argument("--decode-round", action="store_true",
                  help="snap decoded values onto the observed value grid (e.g. integers, 0.1 steps)")
  ap.add_argument("--metrics-out", default="", help="write evaluation results as JSON")
  ap.add_argument("--c2st-clf", default="gb", choices=["gb", "logreg"],
                  help="C2ST critic: gb (strong, CDTD/TabCascade-like) or logreg (TabSyn/TabDiff)")
  ap.add_argument("--c2st-repeats", type=int, default=3)
  ap.add_argument("--no-target-in-joint", action="store_true",
                  help="score the feature block only; by default the label is one more column")
  ap.add_argument("--no-eval", action="store_true")
  args = ap.parse_args()
  generate(args.artifact, args.config, args.n, args.cfg, args.seed, args.out,
           cat_temperature=args.cat_temp, do_eval=not args.no_eval,
           real_split=args.real_split, decode_clip=args.decode_clip,
           decode_round=args.decode_round, metrics_out=args.metrics_out,
           c2st_clf=args.c2st_clf, c2st_repeats=args.c2st_repeats,
           include_target=not args.no_target_in_joint)

if __name__ == "__main__":
  main()
