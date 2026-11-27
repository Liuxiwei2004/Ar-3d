
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.renderer import ARRenderer
from src.pose_estimator import PoseEstimator
from src.pipeline import PoseSmoother
from src.eval import AREvaluator


def open_video_source(source):
    """Open camera or video file."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

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
        "--marker-size",
        type=float,
        default=0.15,
        help="Actual Marker side length in meters. Default: 0.15"
    )

    parser.add_argument(
        "--target-id",
        type=int,
        default=0,
        help="ArUco Marker ID to track. Default: 0"
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

    if args.marker_size <= 0:
        parser.error("--marker-size must be greater than 0")

    capture = open_video_source(args.source)

    writer = None

    try:
        # 1. Read the first frame.
        ret, first_frame = capture.read()

        if not ret:
            raise RuntimeError("Cannot read the first frame.")

        height, width = first_frame.shape[:2]

        # 2. Initialize the real ArUco pose estimator.
        camera_params_path = Path(args.camera_params)

        if not camera_params_path.exists():
            raise FileNotFoundError(
                f"Camera parameters not found: {camera_params_path}"
            )

        pose_estimator = PoseEstimator(
            camera_params_path=str(camera_params_path),
            marker_size=args.marker_size,
            target_id=args.target_id
        )

        # Use exactly the same calibrated parameters for rendering.
        renderer = ARRenderer(
            camera_matrix=pose_estimator.camera_matrix,
            dist_coeffs=pose_estimator.dist_coeffs
        )

        print("Using calibrated camera parameters.")
        print(f"Marker size: {args.marker_size} m")
        print(f"Target Marker ID: {args.target_id}")

        # 3. Initialize pose smoothing and evaluation.
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
                raise FileNotFoundError(
                    f"OBJ model not found: {model_path}"
                )

            model = renderer.load_obj_model(str(model_path))

            print(f"Loaded OBJ model: {model_path}")
            print(f"Vertices: {model['vertex_count']}")
            print(f"Faces: {model['face_count']}")

        # 5. Prepare output video.
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
            raise RuntimeError("Cannot create output video.")

        # 6. Process video frames.
        frame = first_frame

        print("\nStarting AR processing...")
        print("Press Q or ESC to stop.")

        while True:
            start_time = time.perf_counter()

            # Detect ArUco Marker and estimate its pose.
            success, rvec, tvec = pose_estimator.estimate(frame)

            # Smooth pose and manage tracking state.
            success, rvec, tvec = smoother.update(
                success,
                rvec,
                tvec
            )

            # Render virtual object when tracking succeeds.
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

            # Record evaluation metrics.
            evaluator.record_frame(
                success=success,
                rvec=rvec,
                tvec=tvec,
                processing_time=processing_time
            )

            current_fps = (
                1.0 / processing_time
                if processing_time > 0
                else 0.0
            )

            # Display tracking information.
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

            # Save and display the processed frame.
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

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    # 7. Export evaluation results.
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
