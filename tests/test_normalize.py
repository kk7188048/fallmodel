import numpy as np

from falldet.normalize import interpolate_missing_frames, normalize_pose_sequence
from falldet.pose import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER


def test_normalize_centers_hips_at_origin(standing_keypoints):
    kp = standing_keypoints()
    normalized = normalize_pose_sequence(kp)
    hip_center = (normalized[:, LEFT_HIP, :2] + normalized[:, RIGHT_HIP, :2]) / 2
    assert np.allclose(hip_center, 0, atol=1e-8)


def test_normalize_scales_torso_to_unit_length(standing_keypoints):
    kp = standing_keypoints()
    normalized = normalize_pose_sequence(kp)
    hip_center = (normalized[:, LEFT_HIP, :2] + normalized[:, RIGHT_HIP, :2]) / 2
    shoulder_center = (normalized[:, LEFT_SHOULDER, :2] + normalized[:, RIGHT_SHOULDER, :2]) / 2
    torso_len = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    assert np.allclose(torso_len, 1.0, atol=1e-6)


def test_normalize_preserves_visibility_channel(standing_keypoints):
    kp = standing_keypoints()
    kp[:, :, 3] = 0.73
    normalized = normalize_pose_sequence(kp)
    assert np.allclose(normalized[:, :, 3], 0.73)


def test_normalize_handles_degenerate_zero_torso():
    """Hips and shoulders coincide -> torso scale would be 0; must not
    divide by zero / produce inf or NaN."""
    kp = np.zeros((3, 33, 4))
    kp[:, LEFT_HIP] = [0.5, 0.5, 0, 1]
    kp[:, RIGHT_HIP] = [0.5, 0.5, 0, 1]
    kp[:, LEFT_SHOULDER] = [0.5, 0.5, 0, 1]
    kp[:, RIGHT_SHOULDER] = [0.5, 0.5, 0, 1]
    normalized = normalize_pose_sequence(kp)
    assert np.isfinite(normalized).all()


def test_interpolate_fills_interior_gap():
    kp = np.zeros((5, 33, 4))
    for i in range(5):
        kp[i, 0, 0] = i  # a simple linear ramp on one coordinate
    kp[2] = np.nan  # drop the middle frame
    filled = interpolate_missing_frames(kp)
    assert not np.isnan(filled).any()
    assert np.isclose(filled[2, 0, 0], 2.0)


def test_interpolate_fills_leading_and_trailing_gaps_with_nearest():
    kp = np.ones((4, 33, 4)) * 5.0
    kp[0] = np.nan
    kp[3] = np.nan
    filled = interpolate_missing_frames(kp)
    assert np.allclose(filled[0], filled[1])
    assert np.allclose(filled[3], filled[2])


def test_interpolate_all_missing_returns_unchanged():
    kp = np.full((3, 33, 4), np.nan)
    filled = interpolate_missing_frames(kp)
    assert np.isnan(filled).all()
