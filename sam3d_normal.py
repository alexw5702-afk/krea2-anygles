from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image


def _rotate_yaw(points: np.ndarray, keypoints: np.ndarray, degrees: float) -> np.ndarray:
    pelvis = (keypoints[9] + keypoints[10]) * 0.5
    centered = np.asarray(points, dtype=np.float32) - pelvis[None, :]
    radians = np.deg2rad(float(degrees))
    cosine, sine = np.float32(np.cos(radians)), np.float32(np.sin(radians))
    rotated = centered.copy()
    rotated[:, 0] = cosine * centered[:, 0] + sine * centered[:, 2]
    rotated[:, 2] = -sine * centered[:, 0] + cosine * centered[:, 2]
    return rotated + pelvis[None, :]


def _rotate_elevation(points: np.ndarray, keypoints: np.ndarray, degrees: float) -> np.ndarray:
    pelvis = (keypoints[9] + keypoints[10]) * 0.5
    centered = np.asarray(points, dtype=np.float32) - pelvis[None, :]
    radians = np.deg2rad(float(degrees))
    cosine, sine = np.float32(np.cos(radians)), np.float32(np.sin(radians))
    rotated = centered.copy()
    rotated[:, 1] = cosine * centered[:, 1] - sine * centered[:, 2]
    rotated[:, 2] = sine * centered[:, 1] + cosine * centered[:, 2]
    return rotated + pelvis[None, :]


def _project(points, camera, focal, width, height):
    camera_points = points + camera[None, :]
    if np.any(camera_points[:, 2] <= 1e-5):
        raise ValueError("The requested view crosses the camera plane")
    return np.column_stack((
        focal * camera_points[:, 0] / camera_points[:, 2] + width / 2,
        focal * camera_points[:, 1] / camera_points[:, 2] + height / 2,
    )).astype(np.float32)


def _base_fit(points, width, height, margin=0.08):
    low, high = points.min(0), points.max(0)
    span = high - low
    if np.any(span <= 0):
        raise ValueError("Degenerate projected human mesh")
    size = np.array([width, height], dtype=np.float32)
    return min(1.0, float(np.min(size * (1 - 2 * margin) / span)))


def _render(vertices, faces, camera, focal, width, height, scale, offset):
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.column_stack((
        np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8),
        np.full((len(normals),), 255, dtype=np.uint8),
    ))
    camera_vertices = vertices + camera[None, :]
    gl_vertices = camera_vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    render_mesh = trimesh.Trimesh(
        vertices=gl_vertices, faces=faces, vertex_colors=colors, process=False
    )
    scene = pyrender.Scene(
        bg_color=np.array([0, 0, 0, 0], dtype=np.uint8),
        ambient_light=np.ones(3, dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(render_mesh, smooth=True))
    scene.add(
        pyrender.IntrinsicsCamera(
            fx=focal * scale,
            fy=focal * scale,
            cx=width / 2 + float(offset[0]),
            cy=height / 2 + float(offset[1]),
        ),
        pose=np.eye(4, dtype=np.float32),
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        )
    finally:
        renderer.delete()
    mask = depth > 0
    valid = depth[mask]
    near = np.zeros_like(depth, dtype=np.float32)
    if valid.size:
        low, high = np.percentile(valid, (2.0, 98.0))
        near[mask] = np.clip((high - depth[mask]) / max(float(high - low), 1e-6), 0, 1)
    fused = np.zeros((height, width, 3), dtype=np.uint8)
    fused[..., :2] = color[..., :2]
    fused[..., 2] = np.rint(near * 255).astype(np.uint8)
    fused[~mask] = 0
    return Image.fromarray(fused, mode="RGB")


class Sam3DNormalGenerator:
    """Recover one human mesh and render an AnyAngles target normal."""

    def __init__(
        self,
        sam3d_root: str | Path,
        checkpoint: str | Path,
        mhr_model: str | Path,
        *,
        moge_model: str = "Ruicheng/moge-2-vitl-normal",
        device: str = "cuda",
    ):
        import torch

        sam3d_root = Path(sam3d_root).expanduser().resolve()
        if str(sam3d_root) not in sys.path:
            sys.path.insert(0, str(sam3d_root))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from tools.build_fov_estimator import FOVEstimator

        self.device = torch.device(device)
        model, config = load_sam_3d_body(
            str(Path(checkpoint).expanduser()),
            device=self.device,
            mhr_path=str(Path(mhr_model).expanduser()),
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=config,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=FOVEstimator(name="moge2", device=device, path=moge_model),
        )

    def __call__(
        self,
        image: Image.Image,
        *,
        yaw: float,
        elevation: float = 0.0,
        distance: float = 1.0,
    ) -> Image.Image:
        import torch

        image = image.convert("RGB")
        width, height = image.size
        bbox = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
        outputs = self.estimator.process_one_image(
            np.asarray(image, dtype=np.uint8), bboxes=bbox, inference_type="body"
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one human, got {len(outputs)}")
        result = outputs[0]
        keypoints = np.asarray(result["pred_keypoints_3d"], dtype=np.float32)
        vertices = np.asarray(result["pred_vertices"], dtype=np.float32)
        base_camera = np.asarray(result["pred_cam_t"], dtype=np.float32)
        focal = float(np.asarray(result["focal_length"]))
        yawed_keypoints = _rotate_yaw(keypoints, keypoints, yaw)
        yawed_vertices = _rotate_yaw(vertices, keypoints, yaw)
        orbit_vertices = _rotate_elevation(yawed_vertices, keypoints, elevation)
        orbit_keypoints = _rotate_elevation(yawed_keypoints, keypoints, elevation)

        base_projection = _project(orbit_vertices, base_camera, focal, width, height)
        scale = _base_fit(base_projection, width, height)
        camera = base_camera.copy()
        camera[2] *= np.float32(distance)
        projected = _project(orbit_vertices, camera, focal, width, height)
        scaled = (projected - np.array([width / 2, height / 2])) * scale
        center = (scaled.min(0) + scaled.max(0)) * 0.5
        offset = -center
        normal = _render(
            orbit_vertices,
            self.estimator.faces,
            camera,
            focal,
            width,
            height,
            scale,
            offset,
        )
        del outputs, result, orbit_keypoints
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return normal

    def close(self) -> None:
        """Release the mesh and FOV models before Krea 2 is loaded."""
        import gc
        import torch

        estimator = getattr(self, "estimator", None)
        if estimator is None:
            return
        for name in ("output", "image_embeddings", "batch", "model", "fov_estimator"):
            if hasattr(estimator, name):
                setattr(estimator, name, None)
        self.estimator = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
