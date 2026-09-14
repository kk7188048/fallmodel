"""M2 - hip-center translation and torso-scale normalization of keypoints."""
from __future__ import annotations

import numpy as np

from falldet.pose import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

MIN_TORSO_SCALE = 1e-3


def interpolate_missing_frames(keypoints: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where pose estimation found no person.

    A dropped detection is far more likely than a person truly vanishing
    mid-clip (occlusion, motion blur during a fall, a frame MediaPipe just
    missed) - all common in this dataset in exactly the moment we care
    most about, the fall itself. Interpolating keeps a window's frame
    count intact instead of collapsing gaps into a shorter, easier
    sequence, which would look artificially cleaner to the model than
    real footage ever is. Leading/trailing gaps are filled with the
    nearest valid frame since there is nothing to interpolate between.
    """
    keypoints = keypoints.copy()
    t = keypoints.shape[0]
    valid = ~np.isnan(keypoints).any(axis=(1, 2))

    if not valid.any():
        return keypoints  # no detections at all; nothing to interpolate from

    valid_idx = np.flatnonzero(valid)
    for i in range(t):
        if valid[i]:
            continue
        # nearest valid neighbors on either side
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if left.size and right.size:
            lo, hi = left[-1], right[0]
            frac = (i - lo) / (hi - lo)
            keypoints[i] = keypoints[lo] * (1 - frac) + keypoints[hi] * frac
        elif right.size:
            keypoints[i] = keypoints[right[0]]
        else:
            keypoints[i] = keypoints[left[-1]]

    return keypoints


def normalize_pose_sequence(keypoints: np.ndarray) -> np.ndarray:
    """Translate by hip-center and scale by torso length, per frame.

    keypoints: (T, 33, 4) array of (x, y, z, visibility).
    Returns: (T, 33, 4) array; x/y/z are hip-centered and torso-scaled,
    visibility is passed through unchanged.

    This makes the representation invariant to a person's position in
    frame and distance from the camera - both vary freely across the six
    Le2i scenes and carry no information about whether someone fell.
    """
    hip_center = (keypoints[:, LEFT_HIP, :3] + keypoints[:, RIGHT_HIP, :3]) / 2
    shoulder_center = (keypoints[:, LEFT_SHOULDER, :3] + keypoints[:, RIGHT_SHOULDER, :3]) / 2
    torso_scale = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    torso_scale = np.maximum(torso_scale, MIN_TORSO_SCALE)

    normalized = keypoints.copy()
    normalized[:, :, :3] = (keypoints[:, :, :3] - hip_center[:, None, :]) / torso_scale[:, None, None]
    return normalized
