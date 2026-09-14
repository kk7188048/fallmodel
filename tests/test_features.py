import numpy as np

from falldet.features import (
    FEATURE_DIM,
    assemble_feature_vector,
    compute_torso_angle,
    compute_velocity,
)


def test_compute_velocity_first_frame_is_zero():
    seq = np.arange(20).reshape(5, 4).astype(float)
    vel = compute_velocity(seq)
    assert np.allclose(vel[0], 0)


def test_compute_velocity_matches_finite_difference():
    seq = np.array([[0.0], [1.0], [3.0], [6.0]])
    vel = compute_velocity(seq)
    assert np.allclose(vel, [[0.0], [1.0], [2.0], [3.0]])


def test_torso_angle_zero_when_upright(standing_keypoints):
    kp = standing_keypoints()
    angle = compute_torso_angle(kp)
    assert np.allclose(angle, 0.0, atol=1e-6)


def test_torso_angle_is_90_degrees_when_lying(lying_keypoints):
    kp = lying_keypoints()
    angle = compute_torso_angle(kp)
    assert np.allclose(angle, np.pi / 2, atol=1e-6)


def test_torso_angle_handles_degenerate_zero_length_torso():
    kp = np.zeros((2, 33, 4))
    angle = compute_torso_angle(kp)
    assert np.isfinite(angle).all()


def test_assemble_feature_vector_shape(standing_keypoints):
    kp = standing_keypoints(t=10)
    feats = assemble_feature_vector(kp)
    assert feats.shape == (10, FEATURE_DIM)
    assert np.isfinite(feats).all()


def test_assemble_feature_vector_torso_angle_column_matches(standing_keypoints):
    kp = standing_keypoints(t=10)
    feats = assemble_feature_vector(kp)
    torso_angle_col = feats[:, -2]
    assert np.allclose(torso_angle_col, compute_torso_angle(kp))
