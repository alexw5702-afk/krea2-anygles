from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

MAX_PIXELS = 2 * 1024 * 1024


def canvas_size(image: Image.Image) -> tuple[int, int]:
    """Keep the source aspect ratio and align the output to Krea 2's 16px grid."""
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Source dimensions must be positive")
    scale = min(
        max(1.0, 256 / min(width, height)),
        2048 / max(width, height),
        math.sqrt(MAX_PIXELS / (width * height)),
    )
    return (
        max(16, int(width * scale) // 16 * 16),
        max(16, int(height * scale) // 16 * 16),
    )


def camera_prompt(yaw: float, elevation: float = 0.0, distance: float = 1.0,
                  extra: str = "") -> str:
    if not all(math.isfinite(value) for value in (yaw, elevation, distance)):
        raise ValueError("Camera controls must be finite")
    if not -180 <= yaw <= 180:
        raise ValueError("Yaw must be between -180 and 180 degrees")
    if not -60 <= elevation <= 60:
        raise ValueError("Elevation must be between -60 and 60 degrees")
    if not 0.6 <= distance <= 1.8:
        raise ValueError("Distance factor must be between 0.6 and 1.8")
    parts = ["Same figure."]
    if yaw:
        parts.append(
            f"Move the camera {abs(yaw):g} degrees to the "
            f"{'left' if yaw < 0 else 'right'} relative to the input view."
        )
    else:
        parts.append("Keep the same horizontal camera angle as the input.")
    if elevation:
        parts.append(
            f"Move the camera {abs(elevation):g} degrees "
            f"{'downward' if elevation < 0 else 'upward'} relative to the input view."
        )
    if not math.isclose(distance, 1.0, abs_tol=1e-6):
        parts.append(
            "Move the camera farther away from the figure."
            if distance > 1.0 else
            "Move the camera closer to the figure."
        )
    if extra.strip():
        parts.append(extra.strip())
    return " ".join(parts)


def resize_reference(image: Image.Image, max_edge: int = 384) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, max_edge / max(image.size))
    size = (
        max(16, int(round(image.width * scale)) // 16 * 16),
        max(16, int(round(image.height * scale)) // 16 * 16),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


class AnyglesRuntime:
    """Apply the Anygles Control-LoRA to a Krea 2 reference pipeline."""

    def __init__(self, pipeline, checkpoint: str | Path, *, device: str = "cuda"):
        # Keep prompt/canvas helpers importable from the lightweight ComfyUI
        # camera node without pulling in the portable Diffusers adapter stack.
        from spatial_control_lora import SpatialControlLoRAController

        self.pipeline = pipeline
        self.device = device
        self.controller = SpatialControlLoRAController(
            pipeline.transformer,
            {"anygles": Path(checkpoint)},
            device=device,
        )
        self.controller.activate("anygles")

    def encode_normal(self, normal: Image.Image, width: int, height: int, seed: int):
        import torch

        normal = normal.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        tensor = self.pipeline._to_chw_tensor(normal).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(
            (int(seed) ^ 0x46494743) % (2**32)
        )
        with torch.inference_mode():
            latent = self.pipeline._encode_reference_latents(
                [tensor], width * height, generator, torch.device(self.device)
            )[0]
        return self.pipeline._pack_latents(latent.unsqueeze(0))

    def generate(
        self,
        source: Image.Image,
        target_normal: Image.Image,
        *,
        yaw: float,
        elevation: float = 0.0,
        distance: float = 1.0,
        prompt: str = "",
        steps: int = 8,
        seed: int = 42,
    ) -> Image.Image:
        import torch

        source = source.convert("RGB")
        width, height = canvas_size(source)
        frame = source.resize((width, height), Image.Resampling.LANCZOS)
        target_normal = target_normal.convert("RGB")
        if target_normal.size != (width, height):
            raise ValueError(
                f"Target normal must match the output canvas {(width, height)}, got {target_normal.size}"
            )
        control = self.encode_normal(target_normal, width, height, seed)
        self.controller.set_control_tokens(control)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        try:
            return self.pipeline(
                prompt=camera_prompt(yaw, elevation, distance, prompt),
                image=resize_reference(frame),
                width=width,
                height=height,
                num_inference_steps=int(steps),
                guidance_scale=0.0,
                generator=generator,
                reference_max_pixels=384 * 384,
                encode_reference_in_prompt=True,
                kv_cache=True,
            ).images[0]
        finally:
            self.controller.clear_control_tokens()

    def close(self) -> None:
        self.controller.close()
