"""ComfyUI nodes for the yijunwang2 Krea 2 Anygles adapter."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

from . import krea2_anygles_core as core
from .anygles import camera_prompt, canvas_size
from .sam3d_normal import Sam3DNormalGenerator


CONTROL_WEIGHT_KEY = "transformer.first_control.weight"
SAM3D_MODELS = Path(folder_paths.models_dir) / "sam3d_body"
folder_paths.add_model_folder_path("sam3d_body", str(SAM3D_MODELS), is_default=True)


def _sam_checkpoints() -> list[str]:
    values = [
        name for name in folder_paths.get_filename_list("sam3d_body")
        if name.endswith("model.ckpt")
    ]
    return values or ["model.ckpt"]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError(
            f"Anygles expects one ComfyUI IMAGE [1,H,W,C], got {tuple(image.shape)}"
        )
    array = (
        image[0, ..., :3].detach().float().clamp(0, 1).mul(255).round()
        .to(torch.uint8).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _reference_image(source: torch.Tensor, max_edge: int) -> torch.Tensor:
    height, width = source.shape[1:3]
    scale = min(1.0, float(max_edge) / max(height, width))
    out_h = max(16, int(round(height * scale)) // 16 * 16)
    out_w = max(16, int(round(width * scale)) // 16 * 16)
    chw = source[..., :3].movedim(-1, 1)
    if (out_h, out_w) != (height, width):
        chw = torch.nn.functional.interpolate(
            chw, (out_h, out_w), mode="bilinear", align_corners=False
        )
    return chw.movedim(1, -1)


class Krea2AnyglesCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "yaw": (
                    "FLOAT",
                    {"default": 50.0, "min": -180.0, "max": 180.0, "step": 10.0},
                ),
                "elevation": (
                    "FLOAT",
                    {"default": 0.0, "min": -60.0, "max": 60.0, "step": 5.0},
                ),
                "distance": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.6, "max": 1.8, "step": 0.1},
                ),
                "extra_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "sam3d_checkpoint": (_sam_checkpoints(),),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("source", "target_normal", "prompt", "width", "height")
    FUNCTION = "prepare"
    CATEGORY = "Krea2/Anygles"
    DESCRIPTION = (
        "Recover one human mesh, rotate the camera around it, render an aligned "
        "target normal, and assemble the relative-camera prompt."
    )

    def prepare(self, source, yaw, elevation, distance, extra_prompt, sam3d_checkpoint):
        source_pil = _tensor_to_pil(source)
        width, height = canvas_size(source_pil)
        source_pil = source_pil.resize((width, height), Image.Resampling.LANCZOS)

        checkpoint = Path(
            folder_paths.get_full_path_or_raise("sam3d_body", sam3d_checkpoint)
        )
        mhr_model = checkpoint.parent / "assets" / "mhr_model.pt"
        if not mhr_model.is_file():
            raise FileNotFoundError(
                f"Missing {mhr_model}. Download the complete gated SAM 3D Body repository."
            )

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)
        device = comfy.model_management.get_torch_device()
        generator = Sam3DNormalGenerator(
            Path(__file__).resolve().parent,
            checkpoint,
            mhr_model,
            device=str(device),
        )
        try:
            normal = generator(
                source_pil,
                yaw=float(yaw),
                elevation=float(elevation),
                distance=float(distance),
            )
        finally:
            generator.close()
            del generator
            gc.collect()
            comfy.model_management.soft_empty_cache(force=True)

        return (
            _pil_to_tensor(source_pil),
            _pil_to_tensor(normal),
            camera_prompt(float(yaw), float(elevation), float(distance), extra_prompt),
            width,
            height,
        )


class Krea2AnyglesEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "source": ("IMAGE",),
                "target_normal": ("IMAGE",),
                "reference_max_edge": (
                    "INT",
                    {"default": 384, "min": 128, "max": 768, "step": 16},
                ),
                "vlm_reference": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "ANYGLES_CONTROL")
    RETURN_NAMES = ("positive", "anygles_control")
    FUNCTION = "encode"
    CATEGORY = "Krea2/Anygles"
    DESCRIPTION = (
        "Encode the original source for Qwen3-VL and clean reference attention, "
        "plus the aligned full-canvas target normal for spatial control."
    )

    def encode(
        self,
        clip,
        vae,
        prompt,
        source,
        target_normal,
        reference_max_edge=384,
        vlm_reference=True,
    ):
        if source.shape[1:3] != target_normal.shape[1:3]:
            raise ValueError(
                "Anygles source and target normal must share the same canvas; "
                f"got {tuple(source.shape[1:3])} and {tuple(target_normal.shape[1:3])}"
            )
        reference = _reference_image(source, int(reference_max_edge))
        images_vl = [reference] if vlm_reference else []
        image_prompt = (
            "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
            if images_vl else ""
        )
        if images_vl:
            try:
                from comfy.text_encoders.krea2 import KREA2_TEMPLATE

                tokens = clip.tokenize(
                    image_prompt + prompt,
                    images=images_vl,
                    llama_template=KREA2_TEMPLATE,
                )
            except Exception:
                tokens = clip.tokenize(image_prompt + prompt, images=images_vl)
        else:
            tokens = clip.tokenize(prompt)
        positive = clip.encode_from_tokens_scheduled(tokens)
        control = {
            "reference_latents": [vae.encode(reference[..., :3])],
            "control_latent": vae.encode(target_normal[..., :3]),
        }
        return positive, control


class Krea2AnyglesLoadLoRA:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "ANYGLES_PROJECTION")
    RETURN_NAMES = ("model", "anygles_projection")
    FUNCTION = "load"
    CATEGORY = "Krea2/Anygles"
    DESCRIPTION = (
        "Apply the LoRA tensors and retain the dedicated spatial-control "
        "projection that an ordinary LoRA loader would skip."
    )

    def load(self, model, lora_name, strength=1.0):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != path:
            weights = comfy.utils.load_torch_file(path, safe_load=True)
            self.loaded_lora = (path, weights)
        else:
            weights = self.loaded_lora[1]
        if CONTROL_WEIGHT_KEY not in weights:
            raise ValueError(
                f"{lora_name} is not an Anygles adapter: missing {CONTROL_WEIGHT_KEY}"
            )
        projection = weights[CONTROL_WEIGHT_KEY].detach().cpu().contiguous()
        adapter = {key: value for key, value in weights.items() if key != CONTROL_WEIGHT_KEY}
        patched, _clip = comfy.sd.load_lora_for_models(
            model, None, adapter, float(strength), 0.0
        )
        return patched, {"weight": projection, "strength": float(strength), "path": path}


class Krea2AnyglesModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "anygles_control": ("ANYGLES_CONTROL",),
                "anygles_projection": ("ANYGLES_PROJECTION",),
                "kv_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Krea2/Anygles"
    DESCRIPTION = (
        "Inject the target-normal projection into the noisy image tokens and "
        "provide isolated clean-reference K/V to every Krea 2 block."
    )

    def patch(self, model, anygles_control, anygles_projection, kv_cache=True):
        patched = model.clone()
        base_model = patched.model
        dit = patched.get_model_object("diffusion_model")
        required = ("first", "blocks", "patch", "tmlp", "tproj", "txtfusion")
        missing = [name for name in required if not hasattr(dit, name)]
        if missing:
            raise ValueError(
                "Anygles requires current native Krea 2 ComfyUI support; "
                f"diffusion model is missing {missing}"
            )

        reference_latents = [
            base_model.process_latent_in(latent)
            for latent in anygles_control["reference_latents"]
        ]
        control_latent = base_model.process_latent_in(
            anygles_control["control_latent"]
        )
        control_weight = (
            anygles_projection["weight"] * float(anygles_projection["strength"])
        )
        state = {"last_sigma": None, "caches": {}}

        def forward(
            x,
            timesteps,
            context,
            attention_mask=None,
            transformer_options={},
            **kwargs,
        ):
            if not kv_cache:
                return core._forward_with_refs(
                    dit,
                    x,
                    timesteps,
                    context,
                    reference_latents,
                    transformer_options,
                    bbox_norm=None,
                    control_latent=control_latent,
                    control_weight=control_weight,
                )

            sigma = float(timesteps.max())
            sample_sigmas = transformer_options.get("sample_sigmas")
            new_run = state["last_sigma"] is None or sigma > state["last_sigma"]
            if (
                sample_sigmas is not None
                and sigma == float(sample_sigmas[0])
                and sigma != state["last_sigma"]
            ):
                new_run = True
            if new_run:
                state["caches"].clear()
            state["last_sigma"] = sigma

            batch_size = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
            key = core._ref_fingerprint(reference_latents, batch_size, None)
            reference_kv = state["caches"].get(key)
            if reference_kv is None:
                reference_kv = core._precompute_ref_kv(
                    dit,
                    x,
                    timesteps,
                    reference_latents,
                    transformer_options,
                    bbox_norm=None,
                )
                state["caches"][key] = reference_kv
            return core._forward_with_cached_refs(
                dit,
                x,
                timesteps,
                context,
                reference_kv,
                transformer_options,
                control_latent=control_latent,
                control_weight=control_weight,
            )

        patched.add_object_patch("diffusion_model.forward", forward)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "Krea2AnyglesCamera": Krea2AnyglesCamera,
    "Krea2AnyglesEncode": Krea2AnyglesEncode,
    "Krea2AnyglesLoadLoRA": Krea2AnyglesLoadLoRA,
    "Krea2AnyglesModelPatch": Krea2AnyglesModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2AnyglesCamera": "Krea2 Anygles Camera",
    "Krea2AnyglesEncode": "Krea2 Anygles Encode",
    "Krea2AnyglesLoadLoRA": "Krea2 Anygles Load LoRA",
    "Krea2AnyglesModelPatch": "Krea2 Anygles Model Patch",
}
