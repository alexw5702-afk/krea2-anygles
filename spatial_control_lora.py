from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from switchable_lora import SwitchableLoRAController


CONTROL_WEIGHT_KEY = "transformer.first_control.weight"


class SpatialControlLoRAController:
    """Loads an AI-Toolkit Control-LoRA and injects aligned control tokens."""

    def __init__(
        self,
        model,
        checkpoints: Path | Mapping[str, Path],
        *,
        device: str = "cuda",
    ):
        import torch
        import torch.nn.functional as functional
        from safetensors import safe_open

        self.model = model
        self.device = torch.device(device)
        self.base_channels = int(model.img_in.in_features)
        self._control_tokens = None
        self._active = False
        self._active_adapter = None
        self._original_img_in_forward = model.img_in.forward

        if isinstance(checkpoints, Mapping):
            resolved = {
                str(name): path.expanduser().resolve()
                for name, path in checkpoints.items()
            }
        else:
            resolved = {"control": checkpoints.expanduser().resolve()}
        if not resolved:
            raise ValueError("At least one spatial Control-LoRA is required")
        self.checkpoints = resolved
        self.control_weights = {}
        self._active_control_weight = None
        expected = (model.img_in.out_features, self.base_channels)
        for name, checkpoint in resolved.items():
            with safe_open(checkpoint, framework="pt", device="cpu") as weights:
                if CONTROL_WEIGHT_KEY not in weights.keys():
                    raise RuntimeError(
                        f"{name} is missing {CONTROL_WEIGHT_KEY}"
                    )
                control_weight = weights.get_tensor(CONTROL_WEIGHT_KEY).contiguous()
            if tuple(control_weight.shape) != expected:
                raise ValueError(
                    f"{name} control projection shape "
                    f"{tuple(control_weight.shape)} != {expected}"
                )
            self.control_weights[name] = control_weight

        self.lora = SwitchableLoRAController(
            model,
            resolved,
            device=device,
            allowed_extra_keys=frozenset({CONTROL_WEIGHT_KEY}),
        )
        def controlled_img_in(hidden_states):
            base = hidden_states[..., : self.base_channels]
            output = self._original_img_in_forward(base)
            if hidden_states.shape[-1] == self.base_channels:
                return output
            control = hidden_states[..., self.base_channels :]
            if control.shape[-1] != self.base_channels:
                raise ValueError(
                    f"Spatial control channels {control.shape[-1]} != "
                    f"{self.base_channels}"
                )
            control_weight = self._active_control_weight
            if control_weight is None:
                raise RuntimeError("Spatial control projection is not resident")
            residual = functional.linear(
                control.to(control_weight.dtype), control_weight
            )
            return output + residual.to(output.dtype)

        model.img_in.forward = controlled_img_in
        self._pre_hook = model.register_forward_pre_hook(
            self._inject_control_tokens, with_kwargs=True
        )

    def activate(self, adapter: str | bool | None) -> dict[str, object]:
        import torch

        if isinstance(adapter, bool):
            adapter = next(iter(self.checkpoints)) if adapter else None
        if adapter is not None and adapter not in self.checkpoints:
            raise KeyError(f"Unknown spatial Control-LoRA: {adapter}")
        if (
            adapter == self._active_adapter
            and set(self.lora.report["gpu_cached_adapters"])
            == ({adapter} if adapter is not None else set())
        ):
            return self.report

        self.evict_except(adapter)
        self._active_adapter = adapter
        self._active = adapter is not None
        if self._active:
            self.lora.activate(adapter)
            self._active_control_weight = self.control_weights[adapter].to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        else:
            self.clear_control_tokens()
            self.lora.activate(None)
        return self.report

    def evict_except(self, adapter: str | None) -> dict[str, object]:
        if adapter is not None and adapter not in self.checkpoints:
            raise KeyError(f"Unknown spatial Control-LoRA: {adapter}")
        if self._active_adapter != adapter:
            self.clear_control_tokens()
            self._active = False
            self._active_adapter = None
            self._active_control_weight = None
        self.lora.evict_except(adapter)
        return self.report

    def set_control_tokens(self, tokens) -> None:
        import torch

        if not self._active:
            raise RuntimeError("Spatial control must be active before setting tokens")
        if tokens.ndim != 3 or tokens.shape[-1] != self.base_channels:
            raise ValueError(
                "Control tokens must be BLC with the same packed channel count "
                f"as the base model; got {tuple(tokens.shape)}"
            )
        self._control_tokens = tokens.to(
            self.device, dtype=torch.bfloat16
        ).contiguous()

    def clear_control_tokens(self) -> None:
        self._control_tokens = None

    def _inject_control_tokens(self, _module, args, kwargs):
        import torch

        hidden_states = kwargs.get("hidden_states")
        if not self._active or hidden_states is None or self._control_tokens is None:
            return args, kwargs
        control = self._control_tokens
        if control.shape[1] > hidden_states.shape[1]:
            raise ValueError(
                f"Control sequence length {control.shape[1]} exceeds model sequence "
                f"length {hidden_states.shape[1]}"
            )
        if control.shape[1] < hidden_states.shape[1]:
            # Non-cached registered references trail the noisy image sequence.
            # They are clean context, not spatial-control targets.
            control = torch.cat(
                (
                    control,
                    control.new_zeros(
                        control.shape[0],
                        hidden_states.shape[1] - control.shape[1],
                        control.shape[2],
                    ),
                ),
                dim=1,
            )
        if control.shape[0] == 1 and hidden_states.shape[0] != 1:
            control = control.expand(hidden_states.shape[0], -1, -1)
        if control.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"Control batch {control.shape[0]} != image batch "
                f"{hidden_states.shape[0]}"
            )
        kwargs["hidden_states"] = torch.cat(
            (hidden_states, control.to(hidden_states.dtype)), dim=-1
        )
        return args, kwargs

    @property
    def report(self) -> dict[str, object]:
        active_weight = self._active_control_weight
        return {
            "checkpoints": {
                name: str(path) for name, path in self.checkpoints.items()
            },
            "active": self._active,
            "active_adapter": self._active_adapter,
            "base_channels": self.base_channels,
            "control_projection_shape": (
                tuple(active_weight.shape) if active_weight is not None else None
            ),
            "lora": self.lora.report,
            "control_projection_resident": active_weight is not None,
        }

    def close(self) -> None:
        self.activate(False)
        self._pre_hook.remove()
        self.model.img_in.forward = self._original_img_in_forward
        self.lora.close()
        self._active_control_weight = None
        self.control_weights.clear()
