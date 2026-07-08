from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pt.dataset.dataset import get_postprocess_fn
from pt.utils import dist_util
from pt_utils.init_util import load_generator_model_and_params
from pt_utils.rng import make_generator
from utils.env import HF_ROOT

def _is_latent(metadata: dict) -> bool:
    return metadata.get("model_config", {}).get("in_channels", 3) == 4

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Generate class conditional images from a drifting generator")
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--class-ids", required=True)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--num-per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="samples/out.png")
    ap.add_argument("--save-individual", action="store_true")
    ap.add_argument("--hf-root", default=None)
    return ap.parse_args(argv)

@torch.no_grad()
def generate(model, postprocess_fn, labels, cfg_scale, seed, batch_size, device):
    outs = []
    for start in range(0, len(labels), batch_size):
        chunk = labels[start: start + batch_size]
        c = torch.as_tensor(chunk, dtype=torch.long, device=device)
        g = make_generator(device, "gen_images", seed, start)
        samples = model(c, cfg_scale=float(cfg_scale), generator=g)["samples"]
        pixels = postprocess_fn(samples)
        pixels = pixels.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        outs.append((pixels * 255).round().astype(np.uint8))
    return np.concatenate(outs, axis=0)

def save_grid(images, rows, cols, path: Path):
    from PIL import Image
    n, h, w, _ = images.shape
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(min(n, rows * cols)):
        r, col = divmod(i, cols)
        grid[r * h : (r +1) * h, col * w : (col + 1) * w] = images[i]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)

def main(argv=None):
    args = parse_args(argv)
    class_ids = [int(x) for x in args.class_ids.split(",") if x.strip() != ""]
    if not class_ids:
        sys.exit("error: class ids parsed empty")

    device = dist_util.device()
    hf_root = args.hf_root or HF_ROOT
    model, metadata = load_generator_model_and_params(args.init_from, hf_cache_dir=hf_root)
    model = model.to(device).eval()
    latent = _is_latent(metadata)
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=latent)
    
    labels = [c for c in class_ids for _ in range(args.num_per_class)]
    print(args.init_from, latent, class_ids, args.num_per_class, args.cfg_scale, device)
    images = generate(model, postprocess_fn, labels, args.cfg_scale, args.seed, args.batch_size, device)
    out = Path(args.out)
    save_grid(images, rows=len(class_ids), cols=args.num_per_class, path=out)
    print(out, images.shape[0])
    
    if args.save_individual:
        from PIL import Image
        
        idir = out.with_suffix("")
        idir.mkdir(parents=True, exist_ok=True)
        for i, (lbl, img) in enumerate(zip(labels, images)):
            Image.fromarray(img).save(idir / f"class{lbl}_{i:04d}.png")
        print(images.shape[0], idir)


if __name__ == "__main__":
    main()
        
    
