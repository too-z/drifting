"""Convert a Flax MAE artifact (ema_params.msgpack) to torch safetensors.

Usage (needs flax installed — dev-time only):
    python -m pt.convert.convert_mae --name mae_latent_256 [--root ~/hf_cache]

Reads  <root>/models/mae/jax/<name>/
Writes <root>/models/mae/torch/<name>/ {model.safetensors, metadata.json}

Rename table locked against tests/parity/reference_trees/tree_mae_256.txt.
"""

import argparse
import os
from pathlib import Path

from pt.convert.common import (
    flatten_tree,
    load_flax_artifact,
    map_params,
    save_torch_artifact,
    verify_against_module,
)

MAE_RULES = [
    (r"encoder/conv1", "encoder.conv1"),
    (r"encoder/gn1", "encoder.gn1"),
    (r"encoder/layer(\d)_norm", r"encoder.layer\1_norm"),
    (r"encoder/stages_(\d+)/layers_(\d+)/(conv1|conv2|gn1|gn2|proj_conv|proj_gn)",
     r"encoder.stages.\1.\2.\3"),
    (r"decoder/bridge/(conv|gn)", r"decoder.bridge.\1"),
    (r"decoder/(up\d\d)/concat_norm_fn", r"decoder.\1.concat_norm_fn"),
    (r"decoder/(up\d\d)/(proj|refine)/(conv|gn)", r"decoder.\1.\2.\3"),
    (r"decoder/head", "decoder.head"),
    (r"fc", "fc"),
]

HF_REPO_ID = "Goodeat/drifting"


def ensure_artifact(root, name):
    art_dir = Path(root) / "models" / "mae" / "jax" / name
    if not (art_dir / "ema_params.msgpack").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"models/mae/jax/{name}/*"],
            local_dir=root,
        )
    return art_dir


def convert(name, root):
    from pt.models.mae_model import mae_from_metadata

    art_dir = ensure_artifact(root, name)
    params, metadata = load_flax_artifact(art_dir)
    flat = flatten_tree(params)
    state_dict = map_params(flat, MAE_RULES, context=f"mae/{name}")

    model = mae_from_metadata(metadata)
    n = verify_against_module(state_dict, model, context=f"mae/{name}")
    print(f"[{name}] converted {len(state_dict)} tensors, {n:,} params — all checks passed")

    out_dir = Path(root) / "models" / "mae" / "torch" / name
    save_torch_artifact(state_dict, metadata, out_dir, source_note=f"models/mae/jax/{name}")
    print(f"[{name}] wrote {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="e.g. mae_latent_256, mae_latent_640")
    ap.add_argument("--root", default=os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache")))
    args = ap.parse_args()
    convert(args.name, args.root)


if __name__ == "__main__":
    main()
