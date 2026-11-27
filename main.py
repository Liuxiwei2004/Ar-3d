
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.renderer import ARRenderer
from src.pipeline import PoseSmoother
from src.eval import AREvaluator


class DemoPoseEstimator:
    """
    Temporary pose estimator for integration testing.

    This class generates a stable simulated marker pose.
    Replace it with the real PoseEstimator after
    the camera calibration and pose estimation modules
    are completed.
    """

    def estimate(self, frame):
        height, width = frame.shape[:2]

        # Simulated marker pose.
        rvec = np.array([
            [0.15],
            [0.10],
            [0.0]
        ], dtype=np.float64)

        tvec = np.array([
            [0.0],
            [0.0],
            [0.5]
        ], dtype=np.float64)

        return True, rvec, tvec


def load_camera_parameters(params_path):
    """
    Load camera matrix and distortion coefficients.
    """
    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Camera parameters not found: {params_path}"
        )

    data = np.load(str(params_path))

    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError(
            "camera_params.npz must contain "
            "'camera_matrix' and 'dist_coeffs'."
        )

    return data["camera_matrix"], data["dist_coeffs"]


def create_camera_matrix(width, height):
    """
    Create temporary camera parameters for demo testing.

    These are NOT real calibrated camera parameters.
    """
    focal_length = float(max(width, height))

    return np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)


def open_video_source(source):
    """
    Open camera or video file.
    """
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source}"
        )

    return capture


def main():
    parser = argparse.ArgumentParser(
        description="Simple Augmented Reality Demo"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video file path. Default: 0"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="assets/models/test_cube.obj",
        help="Path to OBJ model."
    )

    parser.add_argument(
        "--camera-params",
        type=str,
        default="assets/camera_params.npz",
        help="Path to calibrated camera parameters."
    )

    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Use real calibrated camera parameters."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/ar_output.mp4",
        help="Output video path."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="results/ar_evaluation.csv",
        help="Evaluation CSV path."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Pose smoothing alpha."
    )

    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable pose smoothing."
    )

    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Render built-in cube instead of OBJ model."
    )

    args = parser.parse_args()

    # 1. Open video source.
    capture = open_video_source(args.source)

    ret, first_frame = capture.read()

    if not ret:
        capture.release()
        raise RuntimeError("Cannot read the first frame.")

    height, width = first_frame.shape[:2]

    # 2. Load or create camera parameters.
    if args.use_calibration:
        camera_matrix, dist_coeffs = load_camera_parameters(
            args.camera_params
        )
        print("Using calibrated camera parameters.")
    else:
        camera_matrix = create_camera_matrix(width, height)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        print("Using temporary camera parameters for demo.")

    # 3. Initialize AR components.
    renderer = ARRenderer(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )

    smoother = PoseSmoother(
        alpha=args.alpha,
        enabled=not args.no_smoothing
    )

    evaluator = AREvaluator()

    # 4. Load OBJ model.
    model = None

    if not args.no_model:
        model_path = Path(args.model)

        if not model_path.exists():
            capture.release()
            raise FileNotFoundError(
                f"OBJ model not found: {model_path}"
            )

        model = renderer.load_obj_model(str(model_path))

        print(f"Loaded OBJ model: {model_path}")
        print(f"Vertices: {model['vertex_count']}")
        print(f"Faces: {model['face_count']}")

    # 5. Initialize temporary pose estimator.
    pose_estimator = DemoPoseEstimator()

    # 6. Prepare output video.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot create output video.")

    # 7. Process video frames.
    frame = first_frame
    frame_count = 0

    print("\nStarting AR processing...")
    print("Press Q or ESC to stop.")

    try:
        while True:
            start_time = time.perf_counter()

            # Pose estimation.
            success, rvec, tvec = pose_estimator.estimate(frame)

            # Pose smoothing and tracking state management.
            success, rvec, tvec = smoother.update(
                success,
                rvec,
                tvec
            )

            # Rendering.
            if success:
                rendered_frame = renderer.render(
                    frame.copy(),
                    rvec,
                    tvec,
                    model=model
                )
            else:
                rendered_frame = frame.copy()

            processing_time = time.perf_counter() - start_time

            # Evaluation.
            evaluator.record_frame(
                success=success,
                rvec=rvec,
                tvec=tvec,
                processing_time=processing_time
            )

            # Display frame information.
            frame_count += 1

            current_fps = (
                1.0 / processing_time
                if processing_time > 0
                else 0.0
            )

            cv2.putText(
                rendered_frame,
                f"FPS: {current_fps:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                rendered_frame,
                f"Tracking: {'Success' if success else 'Lost'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if success else (0, 0, 255),
                2
            )

            # Save and display.
            writer.write(rendered_frame)

            cv2.imshow("Simple AR Demo", rendered_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            # Read next frame.
            ret, next_frame = capture.read()

            if not ret:
                break

            frame = next_frame

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    # 8. Export evaluation results.
    csv_path = evaluator.export_csv(args.csv)

    metrics = evaluator.get_metrics()

    print("\n========== AR Evaluation Results ==========")
    print(f"Total frames: {metrics['total_frames']}")
    print(f"Successful frames: {metrics['successful_frames']}")
    print(
        f"Tracking success rate: "
        f"{metrics['tracking_success_rate'] * 100:.2f}%"
    )
    print(f"Average FPS: {metrics['average_fps']:.2f}")
    print(
        f"Average translation jitter: "
        f"{metrics['average_translation_jitter']:.6f}"
    )
    print(
        f"Average rotation jitter: "
        f"{metrics['average_rotation_jitter']:.6f}"
    )

    print("\nOutput video:", output_path)
    print("Evaluation CSV:", csv_path)
    print("AR processing completed!")


if __name__ == "__main__":
    main()
