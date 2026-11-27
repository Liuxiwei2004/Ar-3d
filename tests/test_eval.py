
import sys
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import AREvaluator


def test_tracking_success_rate():
    evaluator = AREvaluator()

    evaluator.record_frame(True, np.zeros((3, 1)), np.ones((3, 1)))
    evaluator.record_frame(False)
    evaluator.record_frame(True, np.zeros((3, 1)), np.ones((3, 1)))

    metrics = evaluator.get_metrics()

    assert metrics["total_frames"] == 3
    assert metrics["successful_frames"] == 2
    assert np.isclose(metrics["tracking_success_rate"], 2 / 3)


def test_fps():
    evaluator = AREvaluator()

    evaluator.record_frame(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1)),
        processing_time=0.1
    )

    evaluator.record_frame(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1)),
        processing_time=0.1
    )

    metrics = evaluator.get_metrics()

    assert np.isclose(metrics["average_fps"], 10.0)


def test_pose_jitter():
    evaluator = AREvaluator()

    evaluator.record_frame(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1))
    )

    evaluator.record_frame(
        True,
        np.ones((3, 1)),
        np.full((3, 1), 3.0)
    )

    metrics = evaluator.get_metrics()

    assert np.isclose(
        metrics["average_translation_jitter"],
        np.sqrt(12)
    )

    assert np.isclose(
        metrics["average_rotation_jitter"],
        np.sqrt(3)
    )


def test_tracking_failure_resets_jitter():
    evaluator = AREvaluator()

    evaluator.record_frame(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1))
    )

    evaluator.record_frame(False)

    evaluator.record_frame(
        True,
        np.full((3, 1), 5.0),
        np.full((3, 1), 6.0)
    )

    metrics = evaluator.get_metrics()

    assert metrics["average_translation_jitter"] == 0.0
    assert metrics["average_rotation_jitter"] == 0.0


def test_csv_export():
    evaluator = AREvaluator()

    evaluator.record_frame(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1)),
        processing_time=0.05
    )

    output_path = Path("results/test_evaluation.csv")
    evaluator.export_csv(output_path)

    assert output_path.exists()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert len(rows) == 1
    assert rows[0]["success"] == "1"

    print("CSV export test passed!")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    test_tracking_success_rate()
    test_fps()
    test_pose_jitter()
    test_tracking_failure_resets_jitter()
    test_csv_export()

    print("All evaluation tests passed!")
