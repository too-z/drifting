from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from pt.dataset.tabular import _load_tabular, get_tabular_postprocess_fn
from pt.utils.init_util import load_generator_model_and_params
from pt.utils.misc import load_config
from pt.utils.rng import make_generator
from utils import env

@torch.no_grad()
def generate(artifact, config_path, n, cfg_scale, seed, out_csv):
  config = load_config(config_path)
  ds_kwargs = dict(config.dataset.get("kwargs", {}))
  csv_path = ds_kwargs["csv_path"]
  target_col = ds_kwargs.get("target_col", "Lable")
  drop_cols = list(ds_kwargs.get("drop_cols", ["Domain"]))
  val_frac = float(ds_kwargs.get("val_frac", 0.1))
  ds_seed = int(ds_kwargs.get("seed", 42))

  data = _load_tabular(csv_path, target_col, drop_cols, val_frac, ds_seed)
  feat_cols = data["feat_cols"]
  num_classes = data["num_classes"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, metadata = load_generator_model_and_params(artifact, env.HF_ROOT)
  model = model.to(device).eval()
  print(metadata.get("step"), getattr(model, 'tabular', False))

  postprocess = get_tabular_postprocess_fn(csv_path = csv_path, target_col=target_col, drop_cols=drop_cols, val_frac=val_frac, seed=ds_seed)
  real_labels = data["labels"]
  p = np.bincount(real_labels, minlength=num_classes) / len(real_labels)
  rng = np.random.default_rng(seed)
  labels = torch.from_numpy(rng.choice(num_classes, size=n, p=p)).long().to(device)

  g = make_generator(device, "generate-tabular", seed)
  samples = model(labels, cfg_scale=float(cfg_scale), generator=g)["samples"] # [n, 1, F]
  table = postprocess(samples).squeeze(1).cpu().numpy()

  gen_df = pd.DataFrame(table, columns=feat_cols)
  gen_df[target_col] = labels.cpu().numpy()
  if out_csv:
    gen_df.to_csv(out_csv, index=False)
    print(f"wrote {len(gen_df)} generated rows")

  real_df = pd.DataFrame(data["X"], columns=feat_cols)
  print("marginal comparison real vs generated")
  print("column real_mean gen_mean real_std gen_std")
  for c in feat_cols:
    print(f"{c:10s} {real_df[c].mean():10.3f} {gen_df[c].mean():10.3f} "
          f"{real_df[c].std():9.3f} {gen_df[c].std():9.3f}")
    return gen_df

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--artifact", required=True, help="run dir or params_ema dir")
  ap.add_argument("--config", required=True)
  ap.add_argument("--n", type=int, default=400)
  ap.add_argument("--cfg", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default="")
  args = ap.parse_args()
  generate(args.artifact, args.config, args.n, args.cfg, args.seed, args.out)

if __name__ == "__main__":
  main()
  
  
