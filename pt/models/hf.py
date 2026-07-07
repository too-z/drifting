"""Torch artifact loaders, mirroring models/hf.py.

Artifact layout mirrors the jax one with backend "torch":
    <root>/models/{gen,mae}/torch/<name>/{model.safetensors, metadata.json}

Resolution order: local converted artifact first, then HF snapshot_download
(for when torch artifacts are published), else a clear error pointing at the
converter.
"""

import json
from pathlib import Path

HF_REPO_ID = "Goodeat/drifting"

_CONVERTER = {"gen": "pt.convert.convert_generator", "mae": "pt.convert.convert_mae"}


def read_metadata(art_dir):
    return json.loads((Path(art_dir) / "metadata.json").read_text())


def _ensure_torch_artifact(kind, name, output_root):
    art_dir = Path(output_root) / "models" / kind / "torch" / name
    if (art_dir / "model.safetensors").exists():
        return art_dir
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"models/{kind}/torch/{name}/*"],
            local_dir=output_root,
        )
    except Exception:
        pass
    if (art_dir / "model.safetensors").exists():
        return art_dir
    raise FileNotFoundError(
        f"torch artifact not found: {art_dir}. Convert it from the jax release "
        f"with: python -m {_CONVERTER[kind]} --name {name} --root {output_root}"
    )


def _load_state(art_dir):
    from safetensors.torch import load_file

    return load_file(str(Path(art_dir) / "model.safetensors"))


def load_generator_torch(name, output_root):
    """Returns (model, metadata) with EMA weights loaded, in eval mode."""
    from pt.models.generator import build_generator_from_config

    art_dir = _ensure_torch_artifact("gen", name, output_root)
    metadata = read_metadata(art_dir)
    model = build_generator_from_config(metadata["model_config"])
    model.load_state_dict(_load_state(art_dir), strict=True)
    model.eval()
    return model, metadata


def load_mae_torch(name, output_root):
    """Returns (model, metadata) with EMA weights loaded, in eval mode."""
    from pt.models.mae_model import mae_from_metadata

    art_dir = _ensure_torch_artifact("mae", name, output_root)
    metadata = read_metadata(art_dir)
    model = mae_from_metadata(metadata)
    model.load_state_dict(_load_state(art_dir), strict=True)
    model.eval()
    return model, metadata
