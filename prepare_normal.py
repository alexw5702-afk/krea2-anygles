from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from anyangles import canvas_size
from sam3d_normal import Sam3DNormalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover one human mesh and render an aligned AnyAngles target normal"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw", type=float, default=45.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--sam3d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mhr-model", type=Path, required=True)
    parser.add_argument("--moge-model", default="Ruicheng/moge-2-vitl-normal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ImageOps.exif_transpose(Image.open(args.source)).convert("RGB")
    source = source.resize(canvas_size(source), Image.Resampling.LANCZOS)
    generator = Sam3DNormalGenerator(
        args.sam3d_root,
        args.checkpoint,
        args.mhr_model,
        moge_model=args.moge_model,
    )
    normal = generator(
        source,
        yaw=args.yaw,
        elevation=args.elevation,
        distance=args.distance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normal.save(args.output)


if __name__ == "__main__":
    main()
