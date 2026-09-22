from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from krea2_weights import original_to_diffusers_key


LORA_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class LoRAPair:
    a: object
    b: object


def _lora_input(hidden_states, pair: LoRAPair):
    """Use the base feature lanes when a spatial control appends extra channels."""
    expected = int(pair.a.shape[-1])
    actual = int(hidden_states.shape[-1])
    if actual < expected:
        raise RuntimeError(
            f"LoRA input width {actual} is smaller than checkpoint width {expected}"
        )
    return hidden_states if actual == expected else hidden_states[..., :expected]


def lora_module_to_diffusers(module: str) -> str:
    mapped = original_to_diffusers_key(f"{module}.weight")
    return mapped.removesuffix(".weight")


def musubi_module_to_original(module: str) -> str:
    if not module.startswith("lora_unet_"):
        raise KeyError(f"Unsupported Musubi Krea 2 LoRA module: {module}")
    name = module.removeprefix("lora_unet_")
    direct = {
        "first": "first",
        "last_linear": "last.linear",
        "tmlp_0": "tmlp.0",
        "tmlp_2": "tmlp.2",
        "tproj_1": "tproj.1",
        "txtmlp_1": "txtmlp.1",
        "txtmlp_3": "txtmlp.3",
        "txtfusion_projector": "txtfusion.projector",
    }
    if name in direct:
        return direct[name]
    match = re.fullmatch(r"blocks_(\d+)_(attn|mlp)_(\w+)", name)
    if match:
        index, kind, layer = match.groups()
        return f"blocks.{index}.{kind}.{layer}"
    match = re.fullmatch(
        r"txtfusion_(layerwise_blocks|refiner_blocks)_(\d+)_(attn|mlp)_(\w+)",
        name,
    )
    if match:
        block, index, kind, layer = match.groups()
        return f"txtfusion.{block}.{index}.{kind}.{layer}"
    raise KeyError(f"Unsupported Musubi Krea 2 LoRA module: {module}")


def load_lora_pairs(
    checkpoint: Path,
    *,
    module_mapper: Callable[[str], str] = lora_module_to_diffusers,
    allowed_extra_keys: frozenset[str] = frozenset(),
) -> dict[str, LoRAPair]:
    from safetensors import safe_open

    checkpoint = checkpoint.expanduser().resolve()
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        pairs = {}
        if any(key.startswith(LORA_PREFIX) for key in keys):
            modules = sorted(
                key.removeprefix(LORA_PREFIX).rsplit(".lora_", 1)[0]
                for key in keys
                if key.startswith(LORA_PREFIX) and key.endswith(".lora_A.weight")
            )
            expected = set()
            for source_module in modules:
                a_key = f"{LORA_PREFIX}{source_module}.lora_A.weight"
                b_key = f"{LORA_PREFIX}{source_module}.lora_B.weight"
                if b_key not in keys:
                    raise RuntimeError(f"Missing LoRA B tensor for {source_module}")
                expected.update((a_key, b_key))
                pairs[module_mapper(source_module)] = LoRAPair(
                    a=weights.get_tensor(a_key).contiguous(),
                    b=weights.get_tensor(b_key).contiguous(),
                )
        else:
            modules = sorted(
                key.rsplit(".lora_down.weight", 1)[0]
                for key in keys
                if key.startswith("lora_unet_") and key.endswith(".lora_down.weight")
            )
            expected = set()
            for source_module in modules:
                a_key = f"{source_module}.lora_down.weight"
                b_key = f"{source_module}.lora_up.weight"
                alpha_key = f"{source_module}.alpha"
                if b_key not in keys:
                    raise RuntimeError(f"Missing LoRA up tensor for {source_module}")
                a = weights.get_tensor(a_key).contiguous()
                b = weights.get_tensor(b_key).contiguous()
                expected.update((a_key, b_key))
                if alpha_key in keys:
                    alpha = float(weights.get_tensor(alpha_key))
                    b = b * (alpha / a.shape[0])
                    expected.add(alpha_key)
                original_module = musubi_module_to_original(source_module)
                pairs[module_mapper(original_module)] = LoRAPair(a=a, b=b)
    unexpected = sorted(keys - expected - allowed_extra_keys)
    if unexpected:
        raise RuntimeError(f"Unsupported LoRA tensors: {unexpected[:8]}")
    if not pairs:
        raise RuntimeError(f"No supported Krea 2 LoRA tensors in {checkpoint}")
    return pairs


class SwitchableLoRAController:
    """Hot-switches active BF16 LoRA residuals over a frozen transformer.

    CPU weights remain cached, while GPU residency follows the current request
    exactly. Repeating the same request therefore reuses the existing tensors;
    switching requests evicts adapters that are no longer active.
    """

    def __init__(
        self,
        model,
        adapters: dict[str, Path],
        *,
        device: str = "cuda",
        module_mapper: Callable[[str], str] = lora_module_to_diffusers,
        allowed_extra_keys: frozenset[str] = frozenset(),
    ):
        import torch

        self.model = model
        self.device = torch.device(device)
        self._cpu_adapters = {
            name: load_lora_pairs(
                path,
                module_mapper=module_mapper,
                allowed_extra_keys=allowed_extra_keys,
            )
            for name, path in adapters.items()
        }
        self._gpu_adapters: dict[str, dict[str, LoRAPair]] = {}
        module_sets = {name: set(pairs) for name, pairs in self._cpu_adapters.items()}
        if len({frozenset(modules) for modules in module_sets.values()}) > 1:
            raise RuntimeError(f"Reference LoRA module sets differ: {module_sets.keys()}")
        self.module_names = sorted(next(iter(module_sets.values()), set()))
        self._active_pairs: dict[str, tuple[LoRAPair, ...]] = {}
        self._active_names: tuple[str, ...] = ()
        self._active_scale = 1.0
        self._hooks = []
        self._install_hooks()

    @property
    def active_name(self) -> str | None:
        if not self._active_names:
            return None
        return "+".join(self._active_names)

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(self._cpu_adapters)

    def _install_hooks(self) -> None:
        import torch
        import torch.nn.functional as functional

        for module_name in self.module_names:
            module = self.model.get_submodule(module_name)

            def add_residual(_module, args, output, *, name=module_name):
                pairs = self._active_pairs.get(name, ())
                if not pairs:
                    return output
                hidden_states = args[0]
                result = output
                for pair in pairs:
                    compute = _lora_input(hidden_states, pair).to(
                        dtype=torch.bfloat16
                    )
                    residual = functional.linear(
                        functional.linear(compute, pair.a),
                        pair.b,
                    )
                    result = (
                        result
                        + residual.to(dtype=output.dtype) * self._active_scale
                    )
                return result

            self._hooks.append(module.register_forward_hook(add_residual))

    def activate(
        self,
        name: str | tuple[str, ...] | list[str] | None,
        *,
        scale: float = 1.0,
    ) -> dict[str, object]:
        import torch

        if name is None:
            names = ()
        elif isinstance(name, str):
            names = (name,)
        else:
            names = tuple(dict.fromkeys(name))
        if (
            names == self._active_names
            and float(scale) == self._active_scale
            and set(self._gpu_adapters) == set(names)
        ):
            return self.report
        unknown = [candidate for candidate in names if candidate not in self._cpu_adapters]
        if unknown:
            raise ValueError(f"Unknown LoRA adapter: {unknown[0]}")

        self.evict_except(names)
        loaded = []
        copied = False
        for candidate in names:
            if candidate not in self._gpu_adapters:
                self._gpu_adapters[candidate] = self._copy_adapter_to_device(
                    candidate
                )
                copied = True
            loaded.append(self._gpu_adapters[candidate])
        self._active_pairs = {
            module_name: tuple(adapter[module_name] for adapter in loaded)
            for module_name in self.module_names
        } if loaded else {}
        self._active_names = names
        self._active_scale = float(scale)
        if copied and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.report

    def evict_except(
        self,
        names: str | tuple[str, ...] | list[str] | None,
    ) -> dict[str, object]:
        if names is None:
            keep = ()
        elif isinstance(names, str):
            keep = (names,)
        else:
            keep = tuple(dict.fromkeys(names))
        unknown = [candidate for candidate in keep if candidate not in self._cpu_adapters]
        if unknown:
            raise ValueError(f"Unknown LoRA adapter: {unknown[0]}")

        keep_set = set(keep)
        stale = [name for name in self._gpu_adapters if name not in keep_set]
        if not stale:
            return self.report

        if any(name not in keep_set for name in self._active_names):
            self._active_pairs = {}
            self._active_names = ()
        for name in stale:
            del self._gpu_adapters[name]
        return self.report

    def _copy_adapter_to_device(self, name: str) -> dict[str, LoRAPair]:
        import torch

        return {
            module_name: LoRAPair(
                a=pair.a.to(self.device, dtype=torch.bfloat16, non_blocking=True),
                b=pair.b.to(self.device, dtype=torch.bfloat16, non_blocking=True),
            )
            for module_name, pair in self._cpu_adapters[name].items()
        }

    def preload_all(self) -> dict[str, object]:
        import torch

        for name in self.adapter_names:
            if name not in self._gpu_adapters:
                self._gpu_adapters[name] = self._copy_adapter_to_device(name)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.report

    @property
    def report(self) -> dict[str, object]:
        parameter_count = 0
        byte_count = 0
        for pairs in self._active_pairs.values():
            for pair in pairs:
                parameter_count += pair.a.numel() + pair.b.numel()
                byte_count += pair.a.numel() * pair.a.element_size()
                byte_count += pair.b.numel() * pair.b.element_size()
        cached_byte_count = 0
        for adapter in self._gpu_adapters.values():
            for pair in adapter.values():
                cached_byte_count += pair.a.numel() * pair.a.element_size()
                cached_byte_count += pair.b.numel() * pair.b.element_size()
        return {
            "active_adapter": self.active_name,
            "active_adapters": list(self._active_names),
            "active_scale": self._active_scale,
            "module_count": len(self._active_pairs),
            "parameter_count": parameter_count,
            "resident_gib": round(byte_count / 2**30, 3),
            "gpu_cached_adapters": list(self._gpu_adapters),
            "gpu_cached_gib": round(cached_byte_count / 2**30, 3),
            "gpu_residency_policy": "active_request_only",
            "available_adapters": list(self.adapter_names),
        }

    def close(self) -> None:
        self.activate(None)
        self._gpu_adapters.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class MultiLoRAController:
    """Hot-switches weighted BF16 LoRAs over one frozen INT8 transformer."""

    def __init__(
        self,
        model,
        adapters: dict[str, Path],
        *,
        device: str = "cuda",
        max_gpu_adapters: int = 4,
        module_mapper: Callable[[str], str] = lora_module_to_diffusers,
    ):
        import torch

        self.model = model
        self.device = torch.device(device)
        self.module_mapper = module_mapper
        self.max_gpu_adapters = max(1, int(max_gpu_adapters))
        self._adapter_paths = {name: Path(path).expanduser().resolve() for name, path in adapters.items()}
        self._cpu_adapters: dict[str, dict[str, LoRAPair]] = {}
        self._gpu_adapters: OrderedDict[str, dict[str, LoRAPair]] = OrderedDict()
        self._active: list[tuple[str, float]] = []
        self._active_by_module: dict[str, tuple[tuple[LoRAPair, float], ...]] = {}
        self._hooks: dict[str, object] = {}

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(self._adapter_paths)

    def _install_hook(self, module_name: str) -> None:
        if module_name in self._hooks:
            return
        import torch
        import torch.nn.functional as functional

        module = self.model.get_submodule(module_name)

        def add_residual(_module, args, output, *, name=module_name):
            active = self._active_by_module.get(name, ())
            if not active:
                return output
            hidden_states = args[0]
            result = output
            for pair, scale in active:
                compute = _lora_input(hidden_states, pair).to(
                    dtype=torch.bfloat16
                )
                residual = functional.linear(functional.linear(compute, pair.a), pair.b)
                result = result + residual.to(dtype=output.dtype) * scale
            return result

        self._hooks[module_name] = module.register_forward_hook(add_residual)

    def _load_cpu(self, name: str) -> dict[str, LoRAPair]:
        if name not in self._adapter_paths:
            raise ValueError(f"Unknown Artist LoRA adapter: {name}")
        if name not in self._cpu_adapters:
            pairs = load_lora_pairs(self._adapter_paths[name], module_mapper=self.module_mapper)
            for module_name in pairs:
                self.model.get_submodule(module_name)
                self._install_hook(module_name)
            self._cpu_adapters[name] = pairs
        return self._cpu_adapters[name]

    def _load_gpu(self, name: str) -> dict[str, LoRAPair]:
        import torch

        if name in self._gpu_adapters:
            self._gpu_adapters.move_to_end(name)
            return self._gpu_adapters[name]
        pairs = {
            module_name: LoRAPair(
                a=pair.a.to(self.device, dtype=torch.bfloat16, non_blocking=True),
                b=pair.b.to(self.device, dtype=torch.bfloat16, non_blocking=True),
            )
            for module_name, pair in self._load_cpu(name).items()
        }
        self._gpu_adapters[name] = pairs
        return pairs

    def activate(self, weighted_adapters: list[tuple[str, float]]) -> dict[str, object]:
        import torch

        combined: OrderedDict[str, float] = OrderedDict()
        for name, scale in weighted_adapters:
            scale = float(scale)
            if scale <= 0:
                continue
            combined[name] = combined.get(name, 0.0) + scale

        desired = list(combined.items())
        if desired == self._active and set(self._gpu_adapters) == set(combined):
            return self.report

        self.evict_except(list(combined))
        loaded = {name: self._load_gpu(name) for name in combined}
        by_module: dict[str, list[tuple[LoRAPair, float]]] = {}
        for name, scale in combined.items():
            for module_name, pair in loaded[name].items():
                by_module.setdefault(module_name, []).append((pair, scale))
        self._active = desired
        self._active_by_module = {name: tuple(values) for name, values in by_module.items()}

        if self.device.type == "cuda" and combined:
            torch.cuda.synchronize(self.device)
        return self.report

    def evict_except(self, names: list[str] | tuple[str, ...]) -> dict[str, object]:
        keep = tuple(dict.fromkeys(names))
        unknown = [name for name in keep if name not in self._adapter_paths]
        if unknown:
            raise ValueError(f"Unknown Artist LoRA adapter: {unknown[0]}")

        keep_set = set(keep)
        stale = [name for name in self._gpu_adapters if name not in keep_set]
        if not stale:
            return self.report

        if any(name not in keep_set for name, _scale in self._active):
            self._active = []
            self._active_by_module = {}
        for name in stale:
            del self._gpu_adapters[name]
        return self.report

    @property
    def report(self) -> dict[str, object]:
        cached_bytes = 0
        for adapter in self._gpu_adapters.values():
            for pair in adapter.values():
                cached_bytes += pair.a.numel() * pair.a.element_size()
                cached_bytes += pair.b.numel() * pair.b.element_size()
        return {
            "active_adapters": [{"name": name, "weight": weight} for name, weight in self._active],
            "active_module_count": len(self._active_by_module),
            "available_adapters": list(self.adapter_names),
            "cpu_cached_adapters": list(self._cpu_adapters),
            "gpu_cached_adapters": list(self._gpu_adapters),
            "max_gpu_adapters": self.max_gpu_adapters,
            "gpu_cached_gib": round(cached_bytes / 2**30, 3),
            "gpu_residency_policy": "active_request_only",
        }

    def close(self) -> None:
        self._active = []
        self._active_by_module = {}
        self._gpu_adapters.clear()
        self._cpu_adapters.clear()
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
