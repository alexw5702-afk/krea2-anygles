# Krea 2 AnyAngles

Reference runtime and preprocessing helpers for
[`yijunwang2/krea2-anyangles`](https://huggingface.co/yijunwang2/krea2-anyangles).
AnyAngles generates a new camera view of one human image using an aligned 3D
normal as spatial control.

[Model card and weights](https://huggingface.co/yijunwang2/krea2-anyangles)

## What it controls

- horizontal orbit from −180° to +180°;
- camera elevation from −60° to +60°;
- camera distance from 0.6× to 1.8×;
- optional user text appended to the generated camera instruction.

The current model is intended for one clear human subject. It does not support
animals, arbitrary objects, or crowds.

## Install

```bash
git clone https://github.com/alexw5702-afk/krea2-anyangles
cd krea2-anyangles
pip install -e .
```

Krea 2 Turbo is gated; accept its license and authenticate with Hugging Face.
For automatic normal preparation, separately install
[`facebookresearch/sam-3d-body`](https://github.com/facebookresearch/sam-3d-body),
accept the SAM license, and download its checkpoint and MHR asset. The Krea and
SAM model weights are not stored in this repository.

## Generate an aligned normal

```bash
python prepare_normal.py \
  --source person.webp \
  --output target_normal.png \
  --yaw 50 \
  --elevation 0 \
  --distance 1.0 \
  --sam3d-root /path/to/sam-3d-body \
  --checkpoint /path/to/model.ckpt \
  --mhr-model /path/to/assets/mhr_model.pt
```

The helper preserves the source aspect ratio, aligns dimensions to 16 pixels,
recovers one human mesh, rotates it around the pelvis, and renders the control
inside the output canvas. `distance < 1` moves closer; `distance > 1` moves
farther away.

## Generate one image

```bash
python example.py \
  --source person.webp \
  --normal target_normal.png \
  --output result.webp \
  --yaw 50 \
  --elevation 0 \
  --distance 1.0 \
  --prompt "soft afternoon light" \
  --steps 8 \
  --seed 42
```

The reference pipeline sends the original image to both Qwen3-VL and Krea 2's
clean visual-reference path. The target normal enters the spatial Control-LoRA
at the output canvas resolution. The camera sentence is generated from the
three controls, and optional text is appended unchanged.

Recommended Krea 2 Turbo settings: eight steps, guidance scale 0, adapter
strength 1.0, a 384px visual reference, and one full-canvas target normal.

## Files

- `anyangles.py`: prompt, canvas, normal encoding, and generation wrapper;
- `sam3d_normal.py`: SAM 3D Body mesh recovery and normal rendering;
- `prepare_normal.py`: preprocessing CLI;
- `example.py`: one-image inference CLI;
- `spatial_control_lora.py`: spatial projection and control-token injection;
- `switchable_lora.py` / `krea2_weights.py`: Krea 2 adapter loading and key mapping.

## License

Original runtime code in this repository is Apache-2.0; see `LICENSE` and
`NOTICE`. The AnyAngles model weights are distributed separately under the
[Krea 2 Community License Agreement](https://krea.ai/krea-2-licensing).
SAM 3D Body is a separate dependency under the SAM License.

This is an unofficial community project and is not endorsed by Krea or Meta.
Training data and training infrastructure are not included.
