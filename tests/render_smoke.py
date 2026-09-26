"""Check direct and isolated normal rendering without downloading weights.

The synthetic articulated mesh covers normal/depth variation, camera changes,
and repeated OpenGL context creation. It does not prove that SAM or Krea works.
No images are saved.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys

import numpy as np
import trimesh


def _renderer_module():
    source = Path(__file__).resolve().parents[1] / "sam3d_normal.py"
    spec = importlib.util.spec_from_file_location("anygles_sam3d_normal", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mesh() -> trimesh.Trimesh:
    # Off-centre arms make front and rear views genuinely different. Overlapping
    # ellipsoids require a depth buffer, unlike the original single triangle.
    parts = (
        ((0.0, 0.15, 0.0), (0.27, 0.42, 0.18)),  # torso
        ((0.0, 0.76, 0.02), (0.19, 0.22, 0.18)),  # head
        ((0.0, -0.28, 0.0), (0.28, 0.17, 0.19)),  # hips
        ((-0.36, 0.13, 0.13), (0.11, 0.37, 0.10)),
        ((0.36, 0.13, -0.10), (0.11, 0.37, 0.10)),
        ((-0.14, -0.72, 0.02), (0.12, 0.37, 0.12)),
        ((0.15, -0.72, -0.02), (0.12, 0.37, 0.12)),
    )
    meshes = []
    for center, radii in parts:
        part = trimesh.creation.icosphere(subdivisions=2)
        part.vertices = part.vertices * np.asarray(radii) + np.asarray(center)
        meshes.append(part)
    return trimesh.util.concatenate(meshes)


def _orbit(vertices: np.ndarray, yaw: float, elevation: float) -> np.ndarray:
    yaw, elevation = np.deg2rad([yaw, elevation])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    turned = vertices.copy()
    turned[:, 0] = cosine * vertices[:, 0] + sine * vertices[:, 2]
    turned[:, 2] = -sine * vertices[:, 0] + cosine * vertices[:, 2]
    cosine, sine = np.cos(elevation), np.sin(elevation)
    tilted = turned.copy()
    tilted[:, 1] = cosine * turned[:, 1] - sine * turned[:, 2]
    tilted[:, 2] = sine * turned[:, 1] + cosine * turned[:, 2]
    return tilted.astype(np.float32)


def _render_view(module, mesh, yaw, elevation, distance):
    width, height = 512, 768
    vertices = _orbit(np.asarray(mesh.vertices), yaw, elevation)
    base_camera = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    focal = 650.0
    base_projection = module._project(vertices, base_camera, focal, width, height)
    scale = module._base_fit(base_projection, width, height)
    camera = base_camera.copy()
    camera[2] *= distance
    projected = module._project(vertices, camera, focal, width, height)
    scaled = (projected - [width / 2, height / 2]) * scale
    offset = -(scaled.min(0) + scaled.max(0)) * 0.5
    return np.asarray(
        module._render(vertices, mesh.faces, camera, focal, width, height, scale, offset)
    )


def main() -> None:
    # Direct Windows rendering needs the default backend. The isolated path
    # below also checks that a stray inherited EGL setting is removed.
    if sys.platform == "win32":
        os.environ.pop("PYOPENGL_PLATFORM", None)
    module = _renderer_module()
    if sys.platform == "linux" and os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise AssertionError("Linux renderer did not select EGL")

    mesh = _mesh()
    views = ((0.0, 0.0, 1.0), (60.0, 25.0, 0.8), (180.0, -25.0, 1.4))
    rendered = []
    for yaw, elevation, distance in views:
        image = _render_view(module, mesh, yaw, elevation, distance)
        if image.shape != (768, 512, 3) or image.dtype != np.uint8:
            raise AssertionError(f"Unexpected render format: {image.shape}, {image.dtype}")
        mask = image.any(axis=2)
        pixels = int(mask.sum())
        if pixels < 10_000 or not np.any(~mask):
            raise AssertionError(f"Empty or unframed normal: {pixels} foreground pixels")
        if np.ptp(image[..., 0][mask]) < 30 or np.ptp(image[..., 1][mask]) < 30:
            raise AssertionError("Surface normal channels have no useful variation")
        if np.ptp(image[..., 2][mask]) < 30:
            raise AssertionError("Depth channel has no useful variation")
        rendered.append(image)
        print(f"yaw={yaw:g} elevation={elevation:g} distance={distance:g}: {pixels} pixels")
    if np.array_equal(rendered[0], rendered[2]):
        raise AssertionError("Front and rear normal renders are identical")
    # Reopening the renderer catches context cleanup failures on later calls.
    repeated = _render_view(module, mesh, *views[0])
    if np.max(np.abs(rendered[0].astype(np.int16) - repeated.astype(np.int16))) > 1:
        raise AssertionError("Repeated rendering changed the control image")
    if sys.platform == "win32":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        isolated = np.asarray(module._render_in_subprocess(
            _orbit(np.asarray(mesh.vertices), 0.0, 0.0), mesh.faces,
            np.array([0.0, 0.0, 3.0], dtype=np.float32), 650.0,
            512, 768, 1.0, np.zeros(2, dtype=np.float32),
        ))
    finally:
        if sys.platform == "win32":
            os.environ.pop("PYOPENGL_PLATFORM", None)
    direct = np.asarray(module._render(
        _orbit(np.asarray(mesh.vertices), 0.0, 0.0), mesh.faces,
        np.array([0.0, 0.0, 3.0], dtype=np.float32), 650.0,
        512, 768, 1.0, np.zeros(2, dtype=np.float32),
    ))
    delta = np.abs(isolated.astype(np.int16) - direct.astype(np.int16))
    if int(delta.max()) > 2:
        raise AssertionError(f"Isolated renderer changed normal/depth values: {delta.max()}")
    print("Normal/depth render smoke passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            f"Renderer environment: OS={platform.platform()} Python={sys.version.split()[0]} "
            f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '<unset>')}",
            file=sys.stderr,
        )
        for package in ("pyrender", "PyOpenGL", "pyglet", "trimesh", "numpy"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "missing"
            print(f"  {package}: {version}", file=sys.stderr)
        raise
