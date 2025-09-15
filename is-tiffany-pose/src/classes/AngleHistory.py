from collections import deque
import numpy as np
import time

class AngleHistory:
    """
    Maintains a history of angles and provides a smoothed output, optionally replacing outliers.

    Attributes:
        max_history (int): Maximum number of angles to keep in history.
        max_age_seconds (float | None): Maximum age of angles in seconds; old angles are purged.
        replace_outliers (bool): Whether to replace detected outliers with the current mean.
        history (deque): Stores tuples of (angle, timestamp).
    """

    def __init__(self, max_history=30, max_age_seconds=None, replace_outliers=True):
        self.history = deque(maxlen=max_history)
        self.max_age_seconds = max_age_seconds
        self.replace_outliers = replace_outliers

    def _purge_old(self):
        if self.max_age_seconds is None:
            return
        now = time.time()
        while self.history and (now - self.history[0][1]) > self.max_age_seconds:
            self.history.popleft()

    def add_angle(self, angle, timestamp=None):
        """Add an angle to history, optionally replacing it if it's an outlier."""
        if timestamp is None:
            timestamp = time.time()

        if self.replace_outliers and self.is_outlier(angle):
            mean_angle = self.mean()
            if mean_angle is not None:
                angle = mean_angle

        self.history.append((angle, timestamp))
        self._purge_old()

    def _get_angles(self):
        self._purge_old()
        return [a for a, _ in self.history]

    def mean(self):
        angles = self._get_angles()
        return np.mean(angles) if angles else None

    def std(self):
        angles = self._get_angles()
        return np.std(angles) if angles else None

    def is_outlier(self, angle, threshold=1.0):
        angles = self._get_angles()
        if len(angles) < 2:
            return False
        mean = np.mean(angles)
        std = np.std(angles)
        if std == 0:
            return False
        return abs(angle - mean) > threshold * std

    def add_and_check(self, angle, timestamp, threshold=1.0):
        """Add angle and return (outlier_flag, smoothed_mean, std)."""
        outlier = self.is_outlier(angle, threshold)
        self.add_angle(angle, timestamp)
        return outlier, self.mean(), self.std()