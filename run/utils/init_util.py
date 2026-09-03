"""Resolve --init-from artifacts from local dirs.

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


def _load_local(path):
    from safetensors.torch import load_file

    art_dir = resolve_artifact_dir(path)
    metadata = json.loads((art_dir / "metadata.json").read_text())
    if metadata.get("backend") not in (None, "torch"):
        raise ValueError(f"{art_dir} is a {metadata.get('backend')} artifact, not torch")
    fname = metadata.get("path", "model.safetensors")
    state = load_file(str(art_dir / fname))
    return state, metadata


def load_generator_model_and_params(init_from):
    """Returns (model, metadata) for inference; weights are the EMA export."""
    from run.models.generator import build_generator_from_config
    
    state, metadata = _load_local(init_from)
    model = build_generator_from_config(metadata["model_config"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def load_params_for_init(init_from):
    """Returns a state_dict from a local artifact. The trainer loads it
    into BOTH the live params and the EMA."""
    state, _ = _load_local(init_from)
    return state

