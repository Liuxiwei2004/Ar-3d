import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.renderer import ARRenderer


def main():
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    renderer = ARRenderer(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        show_axis=True,
        show_cube=True,
        show_faces=True,
        show_shadow=True,
    )

    # Simulated camera frame and marker pose.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rvec = np.array([[0.3], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [0.5]])

    # Load the test OBJ model.
    model = renderer.load_obj_model(
        Path("assets/models/test_cube.obj"),
        target_size=0.04,
    )

    # Render OBJ model.
    output = renderer.render(
        frame, rvec, tvec, model=model
    )

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert np.any(output != frame), "OBJ render produced no visible changes"

    output_path = Path("results/obj_render_test.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError("Failed to save OBJ render test image")

    print("OBJ render test passed!")
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
