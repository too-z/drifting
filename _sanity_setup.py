import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASE = REPO.parent

KAGGLE = (BASE / "imagenet-object-localization-challeng"
          / "ILSVRC" / "Data" / "CLS-LOC" / "train")
OUT = REPO / "_sanity_imagenet"
N_CLASSES = 4
N_TRAIN = 40
N_VAL = 12

if OUT.exists():
  shutil.rmtree(OUT)

wnids = sorted(p.name for p in KAGGLE.iterdir() if p.is_dir())[:N_CLASSES]
for wnid in wnids:
  imgs = sorted(p for p in (KAGGLE / wnid).glob("*.jpeg"))
  for split, sl in (("train" , slice(0, N_TRAIN)), ("val", slice(N_TRAIN, N_TRAIN + N_VAL))):
    dst = OUT / split / wnid
    dst.mkdir(parents=True, exist_ok=True)
    for src in imgs[sl]:
      shutil.copy2(src, dts/src.name)

print(f"classes: {wnids}")
print(f"subset root: {OUT}")
