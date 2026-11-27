
import csv
from pathlib import Path

import numpy as np


class AREvaluator:
    """
    Evaluate AR system performance.

    Metrics:
        - FPS
        - Marker tracking success rate
        - Translation jitter
        - Rotation jitter
        - Per-frame CSV records
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all evaluation records."""
        self.records = []

        self.total_frames = 0
        self.successful_frames = 0
        self.total_processing_time = 0.0

        self.previous_rvec = None
        self.previous_tvec = None

    def record_frame(
        self,
        success,
        rvec=None,
        tvec=None,
        processing_time=0.0
    ):
        """
        Record one frame's tracking result and processing time.

        Args:
            success: Whether marker tracking succeeded.
            rvec: Rotation vector, shape (3,) or (3, 1).
            tvec: Translation vector, shape (3,) or (3, 1).
            processing_time: Processing time in seconds.

        Returns:
            Dictionary containing this frame's evaluation data.
        """
        processing_time = float(processing_time)

        if not np.isfinite(processing_time) or processing_time < 0:
            raise ValueError(
                "processing_time must be a finite non-negative number."
            )

        self.total_frames += 1
        self.total_processing_time += processing_time

        translation_jitter = None
        rotation_jitter = None

        if success:
            if rvec is None or tvec is None:
                raise ValueError(
                    "rvec and tvec are required when success is True."
                )

            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

            if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                raise ValueError("Pose vectors must contain finite values.")

            self.successful_frames += 1

            # Calculate pose changes between consecutive successful frames.
            if self.previous_rvec is not None:
                rotation_jitter = float(
                    np.linalg.norm(rvec - self.previous_rvec)
                )

                translation_jitter = float(
                    np.linalg.norm(tvec - self.previous_tvec)
                )

            self.previous_rvec = rvec.copy()
            self.previous_tvec = tvec.copy()

        else:
            # Do not compare poses across a tracking failure.
            self.previous_rvec = None
            self.previous_tvec = None

        frame_fps = (
            1.0 / processing_time
            if processing_time > 0
            else None
        )

        record = {
            "frame": self.total_frames,
            "success": int(bool(success)),
            "processing_time": processing_time,
            "fps": frame_fps,
            "translation_jitter": translation_jitter,
            "rotation_jitter": rotation_jitter,
        }

        self.records.append(record)

        return record

    def get_metrics(self):
        """Return overall evaluation metrics."""

        tracking_success_rate = (
            self.successful_frames / self.total_frames
            if self.total_frames > 0
            else 0.0
        )

        average_fps = (
            self.total_frames / self.total_processing_time
            if self.total_processing_time > 0
            else 0.0
        )

        translation_values = [
            record["translation_jitter"]
            for record in self.records
            if record["translation_jitter"] is not None
        ]

        rotation_values = [
            record["rotation_jitter"]
            for record in self.records
            if record["rotation_jitter"] is not None
        ]

        average_translation_jitter = (
            float(np.mean(translation_values))
            if translation_values
            else 0.0
        )

        average_rotation_jitter = (
            float(np.mean(rotation_values))
            if rotation_values
            else 0.0
        )

        return {
            "total_frames": self.total_frames,
            "successful_frames": self.successful_frames,
            "tracking_success_rate": tracking_success_rate,
            "average_fps": average_fps,
            "average_translation_jitter": average_translation_jitter,
            "average_rotation_jitter": average_rotation_jitter,
        }

    def export_csv(self, output_path):
        """Export per-frame evaluation records to CSV."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame",
            "success",
            "processing_time",
            "fps",
            "translation_jitter",
            "rotation_jitter",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.records)

        return output_path
