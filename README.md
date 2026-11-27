# Marker-Based Augmented Reality System

**Course:** CS460 / EIE460 / SE460 – Computer Vision Final Project  
**Institution:** Macau University of Science and Technology (MUST)  
**Topic:** Topic 2 – Augmented Reality Implementation

## 1. Project Overview

This project implements a marker-based Augmented Reality (AR) system using OpenCV and Python. It detects ArUco markers in images or video streams, estimates their 6-DoF poses, and overlays virtual 3D objects onto the real-world scene.

The system integrates camera calibration, marker detection, pose estimation, temporal smoothing, 3D model rendering, and quantitative performance evaluation.

The current implementation supports:
- Chessboard-based camera calibration.
- ArUco marker detection using `DICT_4X4_50`.
- 6-DoF pose estimation using OpenCV `solvePnP`.
- Temporal pose stabilization using Exponential Moving Average (EMA).
- 3D coordinate axes and cube rendering.
- Custom OBJ model loading and rendering.
- Semi-transparent mesh faces and ground shadows.
- Video output and CSV-based performance evaluation.

## 2. System Architecture

The system follows this processing pipeline:

1. **Camera Calibration:** Obtain camera intrinsic parameters and distortion coefficients using chessboard images.
2. **Marker Detection:** Detect the target ArUco marker in each frame.
3. **Pose Estimation:** Estimate the marker's rotation and translation relative to the camera.
4. **Pose Smoothing:** Apply EMA filtering to reduce frame-to-frame pose variations.
5. **3D Rendering:** Project virtual object vertices onto the image using `cv2.projectPoints()` and render the model.
6. **Evaluation:** Record tracking success, processing time, FPS, and pose jitter.

## 3. Repository Structure

```text
Ar-3d/
├── assets/
│   ├── calibration_images/       # Chessboard calibration images
│   ├── camera_params.npz         # Camera intrinsic and distortion parameters
│   ├── markers/                  # ArUco marker images
│   └── models/
│       └── test_cube.obj         # Test 3D model
│
├── src/
│   ├── calibrate.py              # Camera calibration
│   ├── generate_marker.py        # ArUco marker generation
│   ├── pose_estimator.py         # Marker detection and pose estimation
│   ├── renderer.py               # 3D projection and rendering
│   ├── pipeline.py               # EMA pose smoothing
│   └── eval.py                   # Performance evaluation
│
├── tests/
│   ├── test_images/
│   │   └── sample.jpg
│   ├── test_video/
│   │   └── sample.mp4
│   ├── test_renderer.py
│   ├── test_obj_loader.py
│   ├── test_obj_render.py
│   ├── test_pipeline.py
│   └── test_eval.py
│
├── results/                      # Generated output files
├── main.py                       # Main AR application
├── requirements.txt
├── README.md
└── .gitignore
```

## 4. Environment Setup

### Requirements

- Python 3.8–3.11
- OpenCV with contrib modules
- NumPy
- Matplotlib
- Pandas
- Trimesh

The project was developed and tested using Python 3.11.4 on macOS.

### Installation

Clone the repository:

```bash
git clone https://github.com/Liuxiwei2004/Ar-3d.git
cd Ar-3d
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For automated testing, install pytest:

```bash
python -m pip install pytest
```

## 5. Camera Calibration and Marker Configuration

### Camera Calibration

The system uses chessboard-based camera calibration to obtain:

- `camera_matrix`: 3 × 3 camera intrinsic matrix.
- `dist_coeffs`: Lens distortion coefficients.
- `mean_reproj_error`: Mean reprojection error.

The provided calibration images are stored in:

```text
assets/calibration_images/
```

The resulting parameters are stored in:

```text
assets/camera_params.npz
```

To recalibrate a different camera, provide a new set of chessboard images and run the calibration script:

```bash
python src/calibrate.py --help
```

Use the available command-line options to configure the calibration image directory, chessboard dimensions, square size, and output path.

### ArUco Marker

The project uses:

- Dictionary: `DICT_4X4_50`
- Target marker ID: `0`
- Physical marker side length: `0.15 m`

The marker coordinate system is centered at the marker, with +X pointing right, +Y pointing upward in the marker plane, and +Z pointing outward from the marker plane.

**Important:** The physical marker size supplied to the program must match the actual printed marker size.

## 6. Running the AR Application

The main entry point is `main.py`.

### Display available options

```bash
python main.py --help
```

### Run with a video file

```bash
python main.py \
    --source tests/test_video/sample.mp4 \
    --output results/ar_output.mp4 \
    --csv results/ar_evaluation.csv
```

### Run with a camera

```bash
python main.py --source 0
```

### Run with a custom OBJ model

```bash
python main.py \
    --source tests/test_video/sample.mp4 \
    --model assets/models/test_cube.obj \
    --output results/ar_output.mp4
```

### Disable pose smoothing

```bash
python main.py \
    --source tests/test_video/sample.mp4 \
    --no-smoothing
```

### Available options

| Argument | Description |
|---|---|
| `--source` | Camera index or input video path |
| `--model` | Path to an OBJ model |
| `--camera-params` | Camera calibration parameter file |
| `--marker-size` | Physical marker side length in meters |
| `--target-id` | Target ArUco marker ID |
| `--output` | Output video path |
| `--csv` | Evaluation CSV output path |
| `--alpha` | EMA smoothing coefficient |
| `--no-smoothing` | Disable pose smoothing |
| `--no-model` | Disable OBJ model rendering |

Press `Q` or `ESC` to stop processing.

## 7. Performance Evaluation

The evaluation module records quantitative measurements during video processing.

The current evaluation metrics include:

- **Tracking Success Rate:** Percentage of frames in which the target marker is successfully detected.
- **Average FPS:** Average processing frame rate.
- **Translation Jitter:** Frame-to-frame variation in estimated translation.
- **Rotation Jitter:** Frame-to-frame variation in estimated rotation.

The evaluation results are exported to CSV files for further analysis.

### Example Results

A test was conducted using the provided 212-frame video.

| Metric | Result |
|---|---:|
| Total Frames | 212 |
| Successful Frames | 212 |
| Tracking Success Rate | 100.00% |
| Average FPS | 19.93 |
| Average Translation Jitter | 0.004173 |
| Average Rotation Jitter | 0.031904 |

These results represent one test run under the development environment and are not intended as universal performance guarantees.

## 8. Testing

### Automated Tests

Run the pytest-compatible tests:

```bash
python -m pytest tests/ -v
```

The current pytest suite covers:

- Tracking success rate calculation.
- FPS calculation.
- Pose jitter calculation.
- CSV export.
- EMA smoothing.
- Smoothing reset and tracking failure handling.

### Additional Rendering Tests

Three rendering-related tests are implemented as standalone Python scripts:

```bash
python tests/test_renderer.py
python tests/test_obj_loader.py
python tests/test_obj_render.py
```

These tests verify basic rendering, OBJ loading, model dimensions, and OBJ projection.

## 9. Team Members and Division of Work

| Team Member | Responsibilities |
|---|---|
| Huang Yufei | Camera calibration, ArUco marker detection, and 6-DoF pose estimation |
| Liu Xiwei | 3D projection and rendering, OBJ model loading, EMA pose smoothing, quantitative evaluation, and system integration |

## 10. References

- OpenCV Documentation: https://docs.opencv.org/
- OpenCV ArUco Module: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Trimesh Documentation: https://trimesh.org/
