
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.renderer import ARRenderer


def main():
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    dist_coeffs = np.zeros((5, 1))

    renderer = ARRenderer(
        camera_matrix,
        dist_coeffs,
    )

    model_path = PROJECT_ROOT / "assets" / "models" / "test_cube.obj"

    model = renderer.load_obj_model(
        model_path,
        target_size=0.04,
    )

    vertices = model["vertices"]
    faces = model["faces"]

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)

    dimensions = vertices.max(axis=0) - vertices.min(axis=0)

    assert np.isclose(dimensions.max(), 0.04)
    assert np.allclose(vertices.mean(axis=0), 0.0)

    print("OBJ loader test passed!")
    print("Vertices:", model["vertex_count"])
    print("Faces:", model["face_count"])
    print("Dimensions:", dimensions)


if __name__ == "__main__":
    main()
