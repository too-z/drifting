"""Dump the parameter tree (paths, shapes, dtypes) of a Flax msgpack artifact.

Standalone on purpose: only needs huggingface_hub + flax.serialization, no repo
imports. The printed tree drives the Flax->torch rename table in pt/convert/.

Usage:
    python tests/parity/dump_flax_tree.py models/gen/jax/ablation [--root ~/hf_cache]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from flax import serialization
from huggingface_hub import snapshot_download

REPO_ID = "Goodeat/drifting"


def flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k in sorted(tree):
            yield from flatten(tree[k], f"{prefix}/{k}" if prefix else k)
    else:
        yield prefix, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="repo path, e.g. models/gen/jax/ablation")
    ap.add_argument("--root", default=os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache")))
    args = ap.parse_args()

    local = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{args.artifact}/*"],
        local_dir=args.root,
    )
    art_dir = Path(local) / args.artifact

    metadata = json.loads((art_dir / "metadata.json").read_text())
    print("=== metadata.json ===")
    print(json.dumps(metadata, indent=2))

    params = serialization.msgpack_restore((art_dir / "ema_params.msgpack").read_bytes())
    print("\n=== param tree ===")
    total = 0
    for path, leaf in flatten(params):
        arr = np.asarray(leaf)
        total += arr.size
        print(f"{path:100s} {str(arr.shape):24s} {arr.dtype}")
    print(f"\ntotal params: {total:,}")


if __name__ == "__main__":
    main()
