"""M2 - sliding window logic over per-frame feature sequences."""
from __future__ import annotations

import numpy as np


def sliding_windows(
    features: np.ndarray, window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chop a (T, D) feature sequence into overlapping fixed-size windows.

    Returns (windows, frame_ranges): windows is (N, window_size, D);
    frame_ranges is (N, 2) of [start, end] 0-indexed frame indices
    (inclusive) that each window covers. A video shorter than window_size
    yields zero windows rather than a padded partial one, since a padded
    window would show the model a fall (or non-fall) it never actually
    saw play out for its full duration.
    """
    t = features.shape[0]
    if t < window_size:
        return (
            np.empty((0, window_size, features.shape[1])),
            np.empty((0, 2), dtype=int),
        )

    starts = np.arange(0, t - window_size + 1, stride)
    windows = np.stack([features[s : s + window_size] for s in starts])
    frame_ranges = np.stack([starts, starts + window_size - 1], axis=1)
    return windows, frame_ranges


def label_windows(
    frame_ranges: np.ndarray,
    fall_start_frame: int | None,
    fall_end_frame: int | None,
) -> np.ndarray:
    """Label each window 1 (fall) if it overlaps the annotated fall
    interval at all, else 0. Le2i annotation frame numbers are 1-indexed;
    frame_ranges (from sliding_windows) are 0-indexed, so we shift here.

    ADL videos (fall_start_frame/fall_end_frame None, or the Le2i "no
    fall" convention of both being 0) label every window 0.
    """
    if not fall_start_frame and not fall_end_frame:
        return np.zeros(len(frame_ranges), dtype=int)

    fall_start = fall_start_frame - 1
    fall_end = fall_end_frame - 1
    starts, ends = frame_ranges[:, 0], frame_ranges[:, 1]
    overlaps = ~((ends < fall_start) | (starts > fall_end))
    return overlaps.astype(int)
