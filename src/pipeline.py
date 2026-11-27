
import numpy as np


class PoseSmoother:
    """
    Smooth AR marker poses across consecutive frames.

    The smoother applies Exponential Moving Average (EMA)
    independently to rotation and translation vectors.
    """

    def __init__(self, alpha=0.6, enabled=True):
        """
        Args:
            alpha: Weight of the new pose.
                   Higher values respond faster but smooth less.
            enabled: Whether smoothing is enabled.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1].")

        self.alpha = float(alpha)
        self.enabled = bool(enabled)

        self.previous_rvec = None
        self.previous_tvec = None

    def smooth(self, rvec, tvec):
        """
        Smooth the current pose.

        Args:
            rvec: Rotation vector, shape (3,) or (3, 1).
            tvec: Translation vector, shape (3,) or (3, 1).

        Returns:
            Tuple of smoothed (rvec, tvec).
            The returned vectors have shape (3, 1).
        """
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            raise ValueError("Pose vectors must contain finite values.")

        if not self.enabled:
            self.previous_rvec = rvec.copy()
            self.previous_tvec = tvec.copy()
            return rvec.copy(), tvec.copy()

        # First valid pose: initialize the filter.
        if self.previous_rvec is None:
            self.previous_rvec = rvec.copy()
            self.previous_tvec = tvec.copy()
            return rvec.copy(), tvec.copy()

        # Exponential Moving Average (EMA).
        smoothed_rvec = (
            self.alpha * rvec
            + (1.0 - self.alpha) * self.previous_rvec
        )

        smoothed_tvec = (
            self.alpha * tvec
            + (1.0 - self.alpha) * self.previous_tvec
        )

        self.previous_rvec = smoothed_rvec.copy()
        self.previous_tvec = smoothed_tvec.copy()

        return smoothed_rvec, smoothed_tvec

    def reset(self):
        """
        Reset the filter when marker tracking is lost.
        """
        self.previous_rvec = None
        self.previous_tvec = None
    
    
    def update(self, success, rvec=None, tvec=None):
        """
        Update pose using the marker tracking result.

        Args:
            success: Whether marker pose estimation succeeded.
            rvec: Current rotation vector.
            tvec: Current translation vector.

        Returns:
            (True, smoothed_rvec, smoothed_tvec) on success.
            (False, None, None) on tracking failure.
        """
        if not success:
            self.reset()
            return False, None, None

        if rvec is None or tvec is None:
            raise ValueError("rvec and tvec are required when success is True.")

        smoothed_rvec, smoothed_tvec = self.smooth(rvec, tvec)

        return True, smoothed_rvec, smoothed_tvec

    def set_enabled(self, enabled):
        """
        Enable or disable pose smoothing.
        """
        self.enabled = bool(enabled)
        self.reset()
