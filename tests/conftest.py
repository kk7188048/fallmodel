"""Synthetic fixtures: fake keypoint arrays, fake video."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.pose import LEFT_HIP, LEFT_SHOULDER, NUM_LANDMARKS, RIGHT_HIP, RIGHT_SHOULDER


@pytest.fixture
def standing_keypoints():
    """A (T, 33, 4) sequence of an upright, stationary stick figure:
    hips at y=0.6, shoulders directly above at y=0.3, full visibility.
    """
    def _make(t: int = 20) -> np.ndarray:
        kp = np.zeros((t, NUM_LANDMARKS, 4))
        kp[:, :, 3] = 1.0  # visibility
        kp[:, LEFT_HIP] = [0.45, 0.6, 0.0, 1.0]
        kp[:, RIGHT_HIP] = [0.55, 0.6, 0.0, 1.0]
        kp[:, LEFT_SHOULDER] = [0.45, 0.3, 0.0, 1.0]
        kp[:, RIGHT_SHOULDER] = [0.55, 0.3, 0.0, 1.0]
        return kp

    return _make


@pytest.fixture
def lying_keypoints():
    """Same stick figure, but shoulders horizontally offset from hips at
    the same height - a torso lying flat.
    """
    def _make(t: int = 20) -> np.ndarray:
        kp = np.zeros((t, NUM_LANDMARKS, 4))
        kp[:, :, 3] = 1.0
        kp[:, LEFT_HIP] = [0.4, 0.8, 0.0, 1.0]
        kp[:, RIGHT_HIP] = [0.5, 0.8, 0.0, 1.0]
        kp[:, LEFT_SHOULDER] = [0.7, 0.8, 0.0, 1.0]
        kp[:, RIGHT_SHOULDER] = [0.8, 0.8, 0.0, 1.0]
        return kp

    return _make


@pytest.fixture
def make_synthetic_video(tmp_path):
    """Write a short synthetic .mp4 (solid color frames) and return its path."""

    def _make(seconds: float = 2.0, fps: int = 10, size: tuple[int, int] = (64, 48)) -> str:
        import cv2

        path = tmp_path / "synthetic.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        n_frames = int(seconds * fps)
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), (i * 5) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return str(path)

    return _make
