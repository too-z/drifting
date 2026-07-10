import sys
from pathlib import Path
REPO = PATH(__file__).resolve().parent
BASE = REPO.parent
sys.path.insert(0, str(REPO))

from pt.utils.misc import load_config
from utils import env

env.IMAGENET_CACHE_PATH = str(REPO / "_sanity_cache")

import pt.train as train_mod

def _no_fid(*a, **k):
  return {"fid":float("nan")}

train_mod.evaluate_fid = _no_fid

cfg = load_config(str(REPO / "configs" / "gen" / "latent_ablation_6gb.yaml"))

cfg.dataset.batch_size = 16
cfg.dataset.eval_batch_size = 16
cfg.dataset.kwargs.num_workers = 0

cfg.train.forward_dict.gen_per_label = 2

cfg.train.total_steps = 20
cfg.train.save_per_step = 100
cfg.train.eval_per_step = 100
cfg.train.push_per_step = 16
cfg.train.cfg_list = [1.0]
cfg.feature.mae_path = "hf://mae_latent_256"

from pt.train import main_gen

main_gen(cfg, output_dir = str(BASE / "_train6gb_smoke_run")
print("\n6GB TRAIN SMOKE OK")
