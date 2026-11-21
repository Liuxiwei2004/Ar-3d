
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.renderer import ARRenderer


def main():
    width = 640
    height = 480

    camera_matrix = np.array([
        [800.0, 0.0, width / 2],
        [0.0, 800.0, height / 2],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    renderer = ARRenderer(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_size=0.05,
        axis_length=0.04,
        cube_size=0.025,
        cube_height=0.04,
        cube_direction=1,
        show_axis=True,
        show_cube=True,
        show_faces=True,
        show_shadow=True,
        shadow_alpha=0.30,
        shadow_blur=21,
        shadow_scale=1.5,
    )

    frame = np.full(
        (height, width, 3),
        230,
        dtype=np.uint8,
    )

    rvec = np.array([
        [np.deg2rad(25.0)],
        [0.0],
        [0.0],
    ], dtype=np.float64)

    tvec = np.array([
        [0.0],
        [0.0],
        [0.5],
    ], dtype=np.float64)

    half = 0.05 / 2.0

    marker_corners = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    projected_corners = renderer.project_points(
        marker_corners, rvec, tvec
    )

    cv2.polylines(
        frame,
        [projected_corners.reshape(-1, 1, 2)],
        True,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )

    output = renderer.render(frame, rvec, tvec)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, frame)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "renderer_test.png"

    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError("Failed to save test image")

    print("Renderer test passed!")
    print(f"Image size: {output.shape}")
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
