
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import PoseSmoother


def test_first_pose():
    smoother = PoseSmoother(alpha=0.6)

    rvec = np.array([[0.1], [0.2], [0.3]])
    tvec = np.array([[1.0], [2.0], [3.0]])

    result_rvec, result_tvec = smoother.smooth(rvec, tvec)

    assert np.allclose(result_rvec, rvec)
    assert np.allclose(result_tvec, tvec)


def test_ema_smoothing():
    smoother = PoseSmoother(alpha=0.6)

    smoother.smooth(
        np.zeros((3, 1)),
        np.ones((3, 1))
    )

    rvec, tvec = smoother.smooth(
        np.ones((3, 1)),
        np.full((3, 1), 3.0)
    )

    assert np.allclose(rvec, 0.6)
    assert np.allclose(tvec, 2.2)


def test_disabled_smoothing():
    smoother = PoseSmoother(alpha=0.6, enabled=False)

    rvec = np.ones((3, 1))
    tvec = np.full((3, 1), 3.0)

    result_rvec, result_tvec = smoother.smooth(rvec, tvec)

    assert np.allclose(result_rvec, rvec)
    assert np.allclose(result_tvec, tvec)


def test_reset():
    smoother = PoseSmoother(alpha=0.6)

    smoother.smooth(
        np.zeros((3, 1)),
        np.ones((3, 1))
    )

    smoother.reset()

    rvec = np.full((3, 1), 2.0)
    tvec = np.full((3, 1), 4.0)

    result_rvec, result_tvec = smoother.smooth(rvec, tvec)

    assert np.allclose(result_rvec, rvec)
    assert np.allclose(result_tvec, tvec)


def test_invalid_alpha():
    try:
        PoseSmoother(alpha=0)
    except ValueError:
        return

    raise AssertionError("Invalid alpha should raise ValueError.")



def test_update_success():
    smoother = PoseSmoother(alpha=0.6)

    rvec = np.zeros((3, 1))
    tvec = np.ones((3, 1))

    success, result_rvec, result_tvec = smoother.update(
        True, rvec, tvec
    )

    assert success is True
    assert np.allclose(result_rvec, rvec)
    assert np.allclose(result_tvec, tvec)


def test_update_tracking_failure():
    smoother = PoseSmoother(alpha=0.6)

    smoother.update(
        True,
        np.zeros((3, 1)),
        np.ones((3, 1))
    )

    success, rvec, tvec = smoother.update(False)

    assert success is False
    assert rvec is None
    assert tvec is None

    # The next detected pose should initialize the filter again.
    new_rvec = np.full((3, 1), 2.0)
    new_tvec = np.full((3, 1), 4.0)

    success, result_rvec, result_tvec = smoother.update(
        True, new_rvec, new_tvec
    )

    assert success is True
    assert np.allclose(result_rvec, new_rvec)
    assert np.allclose(result_tvec, new_tvec)

if __name__ == "__main__":
    test_first_pose()
    test_ema_smoothing()
    test_disabled_smoothing()
    test_reset()
    test_invalid_alpha()
    test_update_success()
    test_update_tracking_failure()

    print("All pipeline tests passed!")
