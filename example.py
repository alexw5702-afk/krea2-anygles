from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

from anyangles import AnyAnglesRuntime


REPO_ID = "yijunwang2/krea2-anyangles"
WEIGHT_NAME = "krea2_anyangles_rank32.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Krea 2 AnyAngles view from a source and aligned target normal"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = DiffusionPipeline.from_pretrained(
        "krea/Krea-2-Turbo",
        custom_pipeline=REPO_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    checkpoint = hf_hub_download(REPO_ID, WEIGHT_NAME)
    runtime = AnyAnglesRuntime(pipe, checkpoint)
    try:
        image = runtime.generate(
            Image.open(args.source),
            Image.open(args.normal),
            yaw=args.yaw,
            elevation=args.elevation,
            distance=args.distance,
            prompt=args.prompt,
            steps=args.steps,
            seed=args.seed,
        )
    finally:
        runtime.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
