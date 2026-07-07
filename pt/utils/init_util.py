"""Torch analog of utils/init_util.py: resolve --init-from / feature-model
artifacts from hf:// names or local dirs.

Local artifact dirs contain {model.safetensors, metadata.json} either directly
or under params_ema/ (the trainer's EMA export layout).
"""

import json
from pathlib import Path


def resolve_artifact_dir(path):
    p = Path(path)
    for cand in (p / "params_ema", p):
        if (cand / "metadata.json").exists():
            return cand
    raise FileNotFoundError(f"no torch artifact (metadata.json) under {path}")


def _load_local(kind, path):
    from safetensors.torch import load_file

    art_dir = resolve_artifact_dir(path)
    metadata = json.loads((art_dir / "metadata.json").read_text())
    if metadata.get("backend") not in (None, "torch"):
        raise ValueError(
            f"{art_dir} is a {metadata.get('backend')} artifact; convert it with "
            f"python -m pt.convert.convert_{'generator' if kind == 'gen' else 'mae'}"
        )
    fname = metadata.get("path", "model.safetensors")
    state = load_file(str(art_dir / fname))
    return state, metadata


def load_generator_model_and_params(init_from, hf_cache_dir):
    """Returns (model, metadata) for inference; weights are the EMA export."""
    from pt.models.generator import build_generator_from_config
    from pt.models.hf import load_generator_torch

    if init_from.startswith("hf://"):
        return load_generator_torch(init_from[len("hf://"):], hf_cache_dir)
    state, metadata = _load_local("gen", init_from)
    model = build_generator_from_config(metadata["model_config"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def load_mae_model_and_params(path, hf_cache_dir):
    """Returns (model, metadata) for the frozen feature model."""
    from pt.models.hf import load_mae_torch
    from pt.models.mae_model import mae_from_metadata

    if path.startswith("hf://"):
        return load_mae_torch(path[len("hf://"):], hf_cache_dir)
    state, metadata = _load_local("mae", path)
    model = mae_from_metadata(metadata)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata
