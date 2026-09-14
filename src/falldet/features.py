"""M2 - velocity, torso angle, and feature vector assembly.

All functions take normalized keypoints (see normalize.py): (T, 33, 4)
arrays of (x, y, z, visibility), hip-centered and torso-scaled, with any
missing-detection frames already interpolated.
"""
from __future__ import annotations

import numpy as np

from falldet.pose import LEFT_HIP, LEFT_SHOULDER, NUM_LANDMARKS, RIGHT_HIP, RIGHT_SHOULDER

# Image y increases downward, so an upright torso vector (hip -> shoulder)
# points in -y. This is the reference we measure torso tilt against.
UP_VECTOR = np.array([0.0, -1.0])


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """Frame-to-frame finite difference along axis 0. First frame is zero
    (no prior frame to diff against) rather than dropped, so the output
    keeps the same length as the input.
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return velocity


def compute_torso_angle(keypoints: np.ndarray) -> np.ndarray:
    """Angle (radians, in [0, pi]) between the torso and vertical.

    0 = standing upright, pi/2 = torso horizontal (lying), the single
    strongest per-frame signal for a fall in this feature set.
    """
    hip_center = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2
    shoulder_center = (keypoints[:, LEFT_SHOULDER, :2] + keypoints[:, RIGHT_SHOULDER, :2]) / 2
    torso = shoulder_center - hip_center

    norm = np.linalg.norm(torso, axis=-1)
    norm = np.maximum(norm, 1e-8)
    cos_angle = (torso @ UP_VECTOR) / norm
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def assemble_feature_vector(keypoints: np.ndarray) -> np.ndarray:
    """Build the per-frame feature matrix fed to the model.

    Returns (T, D) with D = 33*2 (xy) + 33 (visibility) + 33*2 (xy velocity)
    + 1 (torso angle) + 1 (torso angular velocity) = 167.
    """
    xy = keypoints[:, :, :2].reshape(keypoints.shape[0], NUM_LANDMARKS * 2)
    visibility = keypoints[:, :, 3]
    xy_velocity = compute_velocity(xy)

    torso_angle = compute_torso_angle(keypoints)
    torso_angular_velocity = compute_velocity(torso_angle[:, None])[:, 0]

    return np.concatenate(
        [
            xy,
            visibility,
            xy_velocity,
            torso_angle[:, None],
            torso_angular_velocity[:, None],
        ],
        axis=1,
    )


FEATURE_DIM = NUM_LANDMARKS * 2 + NUM_LANDMARKS + NUM_LANDMARKS * 2 + 1 + 1
