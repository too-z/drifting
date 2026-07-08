"""Shared machinery for converting Flax msgpack artifacts to torch state dicts.

Dev-time-only module: imports flax for msgpack decoding. The runtime torch
package (pt/) never imports this.

Value transform rules (applied after path renaming, keyed by the FLAX path):
  * ``.../kernel`` ndim==2  -> weight = kernel.T          (Dense -> Linear)
  * ``.../kernel`` ndim==4  -> weight = HWIO -> OIHW      (Conv)
  * ``.../scale``           -> weight                     (Layer/GroupNorm)
  * ``.../embedding``       -> weight                     (nn.Embed)
  * everything else         -> direct copy
The rename table maps the *module path* (path minus the trailing leaf name);
leaf names are handled uniformly here.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch


def _restore_without_flax(blob):
    import msgpack
    try:
        import ml_dtypes
    except:
        ml_dtypes = None
    def _dtype(name):
        name = name.decode() if isinstance(name, bytes) else name
        if name == "bfloat16":
            if ml_dtypes is None:
                raise ImportError("ml_dtypes required to decode bfloat16 arrays")
            return ml_dtypes.bfloat16
        return np.dtype(name)
    def _ndarray_from_bytes(data):
        shape, dtype_name, buf = msgpack.unpackb(data, raw=True)
        arr = np.frombuffer(buf, dtype=_dtype(dtype_name))
        return arr.reshape([int(s) for s in shape])
    def _ext_hook(code, data):
        return _ndarray_from_bytes(data) if code == 1 else msgpack.ExtType(code, data)
    _BIG = 2**21 - 1
    return msgpack.unpackb(
        blob, ext_hook=_ext_hook raw=False, max_bin_len=_BIG, max_str_len=_BIG, max_array_len=_BIG, max_map_len=_BIG,)
        
def load_flax_artifact(art_dir):
    art_dir = Path(art_dir)
    metadata = json.loads((art_dir / "metadata.json").read_text())
    blob = (art_dir / "ema_params.msgpack").read_bytes()
    try:
        from flax import serialization
        params = serialization.msgpack_restore(blob)
    except ImportError:
        params = _restore_without_flax(blob)
    return params, metadata


def flatten_tree(tree, prefix=""):
    """Flatten a nested dict into {"a/b/c": np.ndarray}."""
    flat = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            flat.update(flatten_tree(v, key))
    else:
        flat[prefix] = np.asarray(tree)
    return flat


LEAF_RENAMES = {"kernel": "weight", "scale": "weight", "embedding": "weight"}


def convert_leaf(flax_path, arr):
    """Apply the layout transform for one leaf. Returns a torch tensor."""
    leaf = flax_path.rsplit("/", 1)[-1]
    t = torch.from_numpy(np.array(arr))
    if leaf == "kernel":
        if t.ndim == 2:
            t = t.t().contiguous()          # Dense [in, out] -> Linear [out, in]
        elif t.ndim == 4:
            t = t.permute(3, 2, 0, 1).contiguous()  # HWIO -> OIHW
        else:
            raise ValueError(f"unexpected kernel ndim {t.ndim} at {flax_path}")
    return t


def map_params(flat_params, module_rules, context=""):
    """Map a flat Flax param dict to a torch state dict.

    module_rules: list of (regex, replacement) applied to the *module* part of
    each path (everything before the final leaf segment). First match wins;
    every path must match exactly one rule, and every produced torch key must
    be unique — violations raise.
    """
    compiled = [(re.compile(pat), rep) for pat, rep in module_rules]
    state_dict = {}
    src_of = {}
    unmatched = []
    for path in sorted(flat_params):
        module_path, leaf = path.rsplit("/", 1)
        torch_module = None
        for rx, rep in compiled:
            m = rx.fullmatch(module_path)
            if m:
                torch_module = m.expand(rep)
                break
        if torch_module is None:
            unmatched.append(path)
            continue
        torch_key = f"{torch_module}.{LEAF_RENAMES.get(leaf, leaf)}"
        if torch_key in state_dict:
            raise ValueError(
                f"{context}: torch key {torch_key} assigned twice "
                f"(from {src_of[torch_key]} and {path})"
            )
        state_dict[torch_key] = convert_leaf(path, flat_params[path])
        src_of[torch_key] = path
    if unmatched:
        raise ValueError(f"{context}: unmatched flax paths:\n  " + "\n  ".join(unmatched))
    return state_dict


def verify_against_module(state_dict, module, context=""):
    """Assert the converted dict covers the torch module exactly (names+shapes)."""
    model_sd = module.state_dict()
    missing = sorted(set(model_sd) - set(state_dict))
    extra = sorted(set(state_dict) - set(model_sd))
    problems = []
    if missing:
        problems.append("missing keys:\n  " + "\n  ".join(missing))
    if extra:
        problems.append("extra keys:\n  " + "\n  ".join(extra))
    for k in sorted(set(model_sd) & set(state_dict)):
        if tuple(model_sd[k].shape) != tuple(state_dict[k].shape):
            problems.append(f"shape mismatch {k}: model {tuple(model_sd[k].shape)} vs converted {tuple(state_dict[k].shape)}")
    if problems:
        raise ValueError(f"{context}:\n" + "\n".join(problems))
    n_model = sum(v.numel() for v in model_sd.values())
    n_conv = sum(v.numel() for v in state_dict.values())
    assert n_model == n_conv, f"{context}: param count {n_conv} != model {n_model}"
    return n_conv


def save_torch_artifact(state_dict, metadata, out_dir, source_note):
    """Write model.safetensors + metadata.json mirroring the jax artifact layout."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state_dict.items()},
              str(out_dir / "model.safetensors"))
    meta = dict(metadata)
    meta["backend"] = "torch"
    meta["format"] = "safetensors"
    meta["path"] = "model.safetensors"
    meta["converted_from"] = source_note
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir
