# ComfyUI Krea 2 Anygles

Native ComfyUI nodes for the
[`yijunwang2/krea2-anygles`](https://huggingface.co/yijunwang2/krea2-anygles)
functional adapter. Anygles turns one human image into a controlled horizontal,
elevated, lowered, closer, or farther camera view.

[Model card and portable Diffusers example](https://huggingface.co/yijunwang2/krea2-anygles)
| [Interactive Space](https://huggingface.co/spaces/yijunwang2/krea2-anygles)
| [Krea 2 functional adapters collection](https://huggingface.co/collections/yijunwang2/krea-2-functional-adapters-6a700e9e5c134888d6615a5d)

![Krea 2 Anygles horizontal orbit](showcase.gif)

## Install

Run these commands from the `ComfyUI` directory, using the Python environment
that starts ComfyUI. Check the CUDA build before installing dependencies and
again afterwards:

```bash
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'CUDA', torch.cuda.is_available()); assert torch.cuda.is_available()"
```

```bash
git clone https://github.com/alexw5702-afk/krea2-anygles custom_nodes/krea2-anygles
python -m pip install -r custom_nodes/krea2-anygles/requirements.txt
python -m pip install 'PyOpenGL==3.1.10'
```

The node requirements do not list `torch` or `torchvision` directly; ComfyUI's
CUDA-enabled builds must remain available. Other packages can still request
PyTorch transitively, so compare the printed versions before and after install.
The final PyOpenGL command restores a tested version satisfying ComfyUI's
current requirement (`>=3.1.8`). `pyrender` currently declares an older exact
version, although the normal renderer is tested with both. Pip may report that
metadata conflict.

Accept the SAM 3D Body license and download its complete checkpoint repository:

```bash
hf download facebook/sam-3d-body-dinov3 \
  --local-dir models/sam3d_body
```

Restart ComfyUI and place `krea2_anygles_rank32.safetensors` in
`ComfyUI/models/loras`.

For **Windows ComfyUI portable**, run the following from its top-level
`ComfyUI_windows_portable` directory instead:

```powershell
git clone https://github.com/alexw5702-afk/krea2-anygles .\ComfyUI\custom_nodes\krea2-anygles
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\krea2-anygles\requirements.txt
.\python_embeded\python.exe -m pip install "PyOpenGL==3.1.10"
.\python_embeded\Scripts\hf.exe download facebook/sam-3d-body-dinov3 --local-dir .\ComfyUI\models\sam3d_body
```

For portable ComfyUI, replace `python` in the CUDA check above with
`.\python_embeded\python.exe` and run it before and after the install command.

The normal renderer uses Linux EGL only on Linux. On Windows it uses
`pyrender`'s default OpenGL context. The Camera node renders in a short-lived
Python subprocess to isolate OpenGL from other ComfyUI nodes. Restart ComfyUI
after updating the node. Use a CUDA-enabled NVIDIA PyTorch build for SAM 3D
Body and the recommended Krea 2 workflow.

If the normal renderer fails, run this small check with the same Python that
starts ComfyUI, from `ComfyUI/custom_nodes/krea2-anygles`:

```bash
python tests/render_smoke.py
```

It renders a synthetic articulated mesh at three camera positions, checks
normal and depth variation, compares direct and isolated rendering, and prints
renderer package versions on failure. It does not test the full Camera/SAM/Krea
workflow, download weights, or save images. On Windows portable, run
`.\python_embeded\python.exe .\ComfyUI\custom_nodes\krea2-anygles\tests\render_smoke.py`
from the top-level portable directory.

## Requirements

- A current ComfyUI build with native Krea 2 support.
- [`krea2_turbo_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_int8_convrot.safetensors)
  in `ComfyUI/models/diffusion_models`.
- [`qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors)
  in `ComfyUI/models/text_encoders`.
- [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors)
  in `ComfyUI/models/vae`.
- [Krea 2 Anygles LoRA](https://huggingface.co/yijunwang2/krea2-anygles/blob/main/krea2_anygles_rank32.safetensors)
  in `ComfyUI/models/loras`.
- `facebook/sam-3d-body-dinov3` under `ComfyUI/models/sam3d_body`, including
  `model.ckpt`, `model_config.yaml`, and `assets/mhr_model.pt`.

MoGe is installed from its pinned upstream revision by `requirements.txt` and
downloads its public FOV model on first use.

## Nodes

### Krea2 Anygles Camera

Takes one source image plus yaw, elevation, distance, and optional user text.
It aligns the source canvas, recovers one human mesh, rotates around the pelvis,
renders the target normal, and produces the complete camera prompt.

Yaw, elevation, and distance can be changed independently or together. The
current model release improves positive-elevation behavior while retaining the
horizontal orbit and distance controls used by the original workflow.

The node deliberately unloads resident Comfy models before SAM 3D Body runs and
releases SAM before Krea 2 sampling begins. This avoids keeping both large model
families on the GPU at once.

### Krea2 Anygles Encode

Encodes the source twice according to the validated inference contract:

- a maximum-edge-384 clean Krea 2 reference and optional Qwen3-VL image input;
- a full-canvas VAE latent of the aligned target normal.

It returns the positive conditioning and one `ANYGLES_CONTROL` object for the
model patch.

### Krea2 Anygles Load LoRA

Loads the regular adapter tensors into Krea 2 and retains
`transformer.first_control.weight` separately. An ordinary LoRA loader skips
that spatial projection and therefore cannot reproduce the model.

### Krea2 Anygles Model Patch

Injects target-normal tokens through the spatial projection at the noisy image
input. The clean source reference is processed at timestep zero and its K/V is
cached once per sampling run.

## Workflow

Drag [`krea2_anygles_workflow.json`](krea2_anygles_workflow.json) into ComfyUI.
The important wiring is:

```text
Load Image -> Anygles Camera -> Anygles Encode
Load Diffusion Model -> Anygles Load LoRA -> Anygles Model Patch -> KSampler
Anygles Encode.positive -> KSampler.positive
Anygles Camera.width/height -> EmptySD3LatentImage -> KSampler.latent_image
KSampler -> VAE Decode -> Save Image
```

Recommended Krea 2 Turbo settings: 8 steps, Euler, simple scheduler, CFG 1.0,
LoRA strength 1.0, 384px source reference, VLM reference enabled, and reference
K/V cache enabled.

This graph was validated end to end at 1008×1344 against ComfyUI commit
`95539f56344958339e39b7582a476267d489b0ee`, with the three linked Comfy-Org
model files, the published Anygles adapter, SAM 3D Body normal preparation, and
the recommended eight-step sampler settings. The raw workflow output is the
model decode; no final source composite is applied.

Yaw is relative to the input: negative values move left and positive values
move right. Negative elevation moves the camera downward; positive elevation
moves it upward. Distance below 1 moves closer and distance above 1 moves
farther away. Optional text is appended after the generated camera instruction.
The node normalizes signed yaw to the model's training-time left-only prompt
representation (`left_angle = (-yaw) mod 360`), while the visible controls keep
the signed left/right convention.

## Current scope

The model supports one clear human subject. It is not designed for animals,
general objects, crowds, or exact 3D scene reconstruction. Hidden surfaces and
background details are generated and can change at large camera movements.

## Credits and license

Reference-attention and K/V-cache code is adapted from
[`RealRebelAI/ComfyUI-Rebels-Krea2-Outpaint`](https://github.com/RealRebelAI/ComfyUI-Rebels-Krea2-Outpaint)
and [`ostris/ComfyUI-Krea2-Ostris-Edit`](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit).
See `NOTICE` for the attribution chain.

Custom node code is MIT licensed. Vendored SAM 3D Body inference source remains
under `SAM_LICENSE`. The separately downloaded Anygles LoRA is a Krea 2
derivative and remains subject to the Krea 2 Community License.
