from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


_ATTENTION_NAMES = {
    "gate": "to_gate",
    "wk": "to_k",
    "wo": "to_out.0",
    "wq": "to_q",
    "wv": "to_v",
}
_FEED_FORWARD_NAMES = {
    "down": "down",
    "gate": "gate",
    "up": "up",
}


def original_to_diffusers_key(key: str) -> str:
    direct = {
        "first.bias": "img_in.bias",
        "first.weight": "img_in.weight",
        "last.linear.bias": "final_layer.linear.bias",
        "last.linear.weight": "final_layer.linear.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
        "last.norm.scale": "final_layer.norm.weight",
        "tmlp.0.bias": "time_embed.linear_1.bias",
        "tmlp.0.weight": "time_embed.linear_1.weight",
        "tmlp.2.bias": "time_embed.linear_2.bias",
        "tmlp.2.weight": "time_embed.linear_2.weight",
        "tproj.1.bias": "time_mod_proj.bias",
        "tproj.1.weight": "time_mod_proj.weight",
        "txtfusion.projector.weight": "text_fusion.projector.weight",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.bias": "txt_in.linear_1.bias",
        "txtmlp.1.weight": "txt_in.linear_1.weight",
        "txtmlp.3.bias": "txt_in.linear_2.bias",
        "txtmlp.3.weight": "txt_in.linear_2.weight",
    }
    if key in direct:
        return direct[key]

    parts = key.split(".")
    if parts[0] == "blocks" and len(parts) >= 4:
        prefix = f"transformer_blocks.{parts[1]}"
        suffix = parts[2:]
    elif (
        parts[0] == "txtfusion"
        and parts[1] in {"layerwise_blocks", "refiner_blocks"}
        and len(parts) >= 5
    ):
        prefix = f"text_fusion.{parts[1]}.{parts[2]}"
        suffix = parts[3:]
    else:
        raise KeyError(f"Unsupported original Krea 2 key: {key}")

    if suffix[:2] == ["attn", "qknorm"]:
        norm_name = {"knorm": "norm_k", "qnorm": "norm_q"}.get(suffix[2])
        if norm_name and suffix[3:] == ["scale"]:
            return f"{prefix}.attn.{norm_name}.weight"
    if suffix[0] == "attn" and suffix[1] in _ATTENTION_NAMES and suffix[2:] == ["weight"]:
        return f"{prefix}.attn.{_ATTENTION_NAMES[suffix[1]]}.weight"
    if suffix[0] == "mlp" and suffix[1] in _FEED_FORWARD_NAMES and suffix[2:] == ["weight"]:
        return f"{prefix}.ff.{_FEED_FORWARD_NAMES[suffix[1]]}.weight"
    if suffix == ["mod", "lin"]:
        return f"{prefix}.scale_shift_table"
    if suffix == ["prenorm", "scale"]:
        return f"{prefix}.norm1.weight"
    if suffix == ["postnorm", "scale"]:
        return f"{prefix}.norm2.weight"
    raise KeyError(f"Unsupported original Krea 2 key: {key}")


def load_pipeline_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("dm_krea2_ostris_edit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reference pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_diffusers_keys(index_path: Path) -> set[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return set(data["weight_map"])


def inspect_original_checkpoint(checkpoint: Path) -> dict[str, object]:
    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
    converted = {original_to_diffusers_key(key) for key in source_keys}
    return {
        "source_key_count": len(source_keys),
        "converted_key_count": len(converted),
        "converted_keys": converted,
    }


def load_reference_transformer(
    pipeline_module: ModuleType,
    checkpoint: Path,
    config_path: Path,
    index_path: Path,
):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    with init_empty_weights():
        transformer = pipeline_module.Krea2Transformer2DModel(**config)

    target_shapes = {key: value.shape for key, value in transformer.state_dict().items()}
    indexed_keys = expected_diffusers_keys(index_path)
    if set(target_shapes) != indexed_keys:
        missing = sorted(indexed_keys - set(target_shapes))
        extra = sorted(set(target_shapes) - indexed_keys)
        raise RuntimeError(f"Reference transformer/index mismatch; missing={missing}, extra={extra}")

    loaded = set()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for source_key in handle.keys():
            target_key = original_to_diffusers_key(source_key)
            if target_key not in target_shapes:
                raise KeyError(f"Converted key is absent from transformer: {target_key}")
            tensor = handle.get_tensor(source_key)
            expected_shape = target_shapes[target_key]
            if tensor.shape != expected_shape:
                if tensor.numel() != expected_shape.numel():
                    raise ValueError(
                        f"Shape mismatch for {source_key} -> {target_key}: "
                        f"{tuple(tensor.shape)} != {tuple(expected_shape)}"
                    )
                tensor = tensor.reshape(expected_shape)
            set_module_tensor_to_device(
                transformer,
                target_key,
                device="cpu",
                value=tensor,
                dtype=torch.bfloat16,
            )
            loaded.add(target_key)

    missing = sorted(set(target_shapes) - loaded)
    if missing:
        raise RuntimeError(f"Checkpoint did not populate {len(missing)} parameters: {missing}")
    return transformer
