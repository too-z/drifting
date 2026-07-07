"""Convert a Flax generator artifact (ema_params.msgpack) to torch safetensors.

Usage (needs flax installed — dev-time only):
    python -m pt.convert.convert_generator --name ablation [--root ~/hf_cache]

Reads  <root>/models/gen/jax/<name>/   (downloads from HF if missing)
Writes <root>/models/gen/torch/<name>/ {model.safetensors, metadata.json}

The rename table below is locked against the real msgpack tree
(tests/parity/reference_trees/tree_gen_ablation.txt).
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

# (regex on the flax module path, torch module replacement) — first match wins.
GENERATOR_RULES = [
    (r"Embed_0", "class_embed"),
    (r"noise_embeds_(\d+)", r"noise_embeds.\1"),
    (r"TimestepEmbedder_0/TorchLinear_0/Dense_0", "cfg_embedder.mlp.0"),
    (r"TimestepEmbedder_0/TorchLinear_1/Dense_0", "cfg_embedder.mlp.2"),
    (r"RMSNorm_0", "cfg_norm"),
    (r"LightningDiT_0/TorchLinear_0/Dense_0", "model.patch_embed"),
    (r"LightningDiT_0/TorchLinear_1/Dense_0", "model.cls_proj"),
    (r"LightningDiT_0", "model"),  # bare params: pos_embed, cls_embed
    (r"LightningDiT_0/blocks_(\d+)/RMSNorm_0", r"model.blocks.\1.norm1"),
    (r"LightningDiT_0/blocks_(\d+)/RMSNorm_1", r"model.blocks.\1.norm2"),
    (r"LightningDiT_0/blocks_(\d+)/Attention_0/TorchLinear_0/Dense_0", r"model.blocks.\1.attn.qkv"),
    (r"LightningDiT_0/blocks_(\d+)/Attention_0/TorchLinear_1/Dense_0", r"model.blocks.\1.attn.proj"),
    (r"LightningDiT_0/blocks_(\d+)/Attention_0/q_norm", r"model.blocks.\1.attn.q_norm"),
    (r"LightningDiT_0/blocks_(\d+)/Attention_0/k_norm", r"model.blocks.\1.attn.k_norm"),
    (r"LightningDiT_0/blocks_(\d+)/SwiGLUFFN_0/TorchLinear_0/Dense_0", r"model.blocks.\1.mlp.w1"),
    (r"LightningDiT_0/blocks_(\d+)/SwiGLUFFN_0/TorchLinear_1/Dense_0", r"model.blocks.\1.mlp.w3"),
    (r"LightningDiT_0/blocks_(\d+)/SwiGLUFFN_0/TorchLinear_2/Dense_0", r"model.blocks.\1.mlp.w2"),
    (r"LightningDiT_0/blocks_(\d+)/StandardMLP_0/TorchLinear_0/Dense_0", r"model.blocks.\1.mlp.fc1"),
    (r"LightningDiT_0/blocks_(\d+)/StandardMLP_0/TorchLinear_1/Dense_0", r"model.blocks.\1.mlp.fc2"),
    (r"LightningDiT_0/blocks_(\d+)/TorchLinear_0/Dense_0", r"model.blocks.\1.adaLN.1"),
    (r"LightningDiT_0/FinalLayer_0/RMSNorm_0", "model.final_layer.norm"),
    (r"LightningDiT_0/FinalLayer_0/TorchLinear_0/Dense_0", "model.final_layer.adaLN.1"),
    (r"LightningDiT_0/FinalLayer_0/TorchLinear_1/Dense_0", "model.final_layer.linear"),
]

HF_REPO_ID = "Goodeat/drifting"


def ensure_artifact(root, name):
    art_dir = Path(root) / "models" / "gen" / "jax" / name
    if not (art_dir / "ema_params.msgpack").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"models/gen/jax/{name}/*"],
            local_dir=root,
        )
    return art_dir


def convert(name, root):
    from pt.models.generator import build_generator_from_config

    art_dir = ensure_artifact(root, name)
    params, metadata = load_flax_artifact(art_dir)
    flat = flatten_tree(params)
    state_dict = map_params(flat, GENERATOR_RULES, context=f"gen/{name}")

    model = build_generator_from_config(metadata["model_config"])
    n = verify_against_module(state_dict, model, context=f"gen/{name}")
    print(f"[{name}] converted {len(state_dict)} tensors, {n:,} params — all checks passed")

    out_dir = Path(root) / "models" / "gen" / "torch" / name
    save_torch_artifact(state_dict, metadata, out_dir, source_note=f"models/gen/jax/{name}")
    print(f"[{name}] wrote {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="artifact name, e.g. ablation, latent_L_sota")
    ap.add_argument("--root", default=os.environ.get("HF_ROOT", os.path.expanduser("~/hf_cache")))
    args = ap.parse_args()
    convert(args.name, args.root)


if __name__ == "__main__":
    main()
