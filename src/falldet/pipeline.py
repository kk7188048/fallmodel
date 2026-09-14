"""M2 - process_video() orchestrates pose extraction, normalization,
feature assembly, and windowing for a single video.
"""
from __future__ import annotations

import numpy as np

from falldet import features, pose, windowing
from falldet.normalize import interpolate_missing_frames, normalize_pose_sequence


def process_video(
    video_path: str,
    fall_start_frame: int | None = None,
    fall_end_frame: int | None = None,
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    window_size: int = 30,
    stride: int = 15,
) -> dict:
    """Run the full M2 pipeline on one video and return its windows.

    Returns a dict with:
      windows: (N, window_size, D) feature windows
      labels: (N,) 0/1 fall label per window
      frame_ranges: (N, 2) [start, end] 0-indexed frame indices per window
      num_frames: total frames in the video
      detection_rate: fraction of frames where a person was detected
    """
    raw_keypoints = pose.estimate_video(
        video_path,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    num_frames = raw_keypoints.shape[0]
    if num_frames == 0:
        return {
            "windows": np.empty((0, window_size, features.FEATURE_DIM)),
            "labels": np.empty((0,), dtype=int),
            "frame_ranges": np.empty((0, 2), dtype=int),
            "num_frames": 0,
            "detection_rate": 0.0,
        }

    missing = np.isnan(raw_keypoints).any(axis=(1, 2))
    detection_rate = 1.0 - missing.mean()

    keypoints = interpolate_missing_frames(raw_keypoints)
    keypoints = normalize_pose_sequence(keypoints)
    feature_matrix = features.assemble_feature_vector(keypoints)

    windows, frame_ranges = windowing.sliding_windows(feature_matrix, window_size, stride)
    labels = windowing.label_windows(frame_ranges, fall_start_frame, fall_end_frame)

    return {
        "windows": windows,
        "labels": labels,
        "frame_ranges": frame_ranges,
        "num_frames": num_frames,
        "detection_rate": detection_rate,
    }
