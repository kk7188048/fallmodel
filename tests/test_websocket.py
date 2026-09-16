"""WebSocket tests for /ws/predict.

`falldet.pose.estimate_frame` is mocked at the point api.ws imports it
(not `create_pose`, which still opens a real - if unused - Pose context)
so these tests don't depend on MediaPipe actually detecting a person in a
fake JPEG - same "mock the pose-extraction boundary, not the API logic"
pattern used in test_api_predict.py.
"""
import importlib

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from falldet.features import FEATURE_DIM, assemble_feature_vector
from falldet.model import FallDetectionModel
from falldet.normalize import interpolate_missing_frames, normalize_pose_sequence
from falldet.pose import NUM_CHANNELS, NUM_LANDMARKS


def _make_checkpoint(tmp_path, window_size=5):
    config = {
        "model": {
            "hidden_dim": 8, "num_layers": 1, "bidirectional": True,
            "attention_dim": 4, "dropout": 0.0, "num_classes": 2,
        },
        "pose": {"model_complexity": 0, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
        "windowing": {"window_size": window_size, "stride": 1},
    }
    model = FallDetectionModel.from_config(config, input_dim=FEATURE_DIM)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": FEATURE_DIM, "config": config},
        checkpoint_path,
    )
    return checkpoint_path


def _standing_keypoints():
    kp = np.zeros((NUM_LANDMARKS, NUM_CHANNELS))
    kp[:, 3] = 1.0
    kp[23] = [0.45, 0.6, 0, 1]  # left hip
    kp[24] = [0.55, 0.6, 0, 1]  # right hip
    kp[11] = [0.45, 0.3, 0, 1]  # left shoulder
    kp[12] = [0.55, 0.3, 0, 1]  # right shoulder
    return kp


def _make_app(tmp_path, monkeypatch, window_size=5):
    checkpoint_path = _make_checkpoint(tmp_path, window_size=window_size)
    monkeypatch.setenv("FALLDET_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("FALLDET_THRESHOLD", "0.5")

    import api.ws as ws_module

    monkeypatch.setattr(ws_module, "estimate_frame", lambda pose, frame: _standing_keypoints())

    import api.main as main_module

    importlib.reload(main_module)
    return main_module.app


def _fake_jpeg_bytes():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_websocket_reports_buffer_filling_before_full(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, window_size=5)
    client = TestClient(app)
    jpeg = _fake_jpeg_bytes()

    with client.websocket_connect("/ws/predict") as ws:
        for i in range(4):
            ws.send_bytes(jpeg)
            msg = ws.receive_json()
            assert msg["buffer_full"] is False
            assert msg["frames_collected"] == i + 1
            assert msg["prediction"] is None
            assert msg["confidence"] is None


def test_websocket_predicts_once_buffer_is_full(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, window_size=5)
    client = TestClient(app)
    jpeg = _fake_jpeg_bytes()

    with client.websocket_connect("/ws/predict") as ws:
        for _ in range(4):
            ws.send_bytes(jpeg)
            ws.receive_json()

        ws.send_bytes(jpeg)
        msg = ws.receive_json()

        assert msg["buffer_full"] is True
        assert msg["frames_collected"] == 5
        assert msg["prediction"] in {"fall", "adl"}
        assert 0.0 <= msg["confidence"] <= 1.0


def test_websocket_keeps_predicting_on_subsequent_frames(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, window_size=3)
    client = TestClient(app)
    jpeg = _fake_jpeg_bytes()

    with client.websocket_connect("/ws/predict") as ws:
        for _ in range(3):
            ws.send_bytes(jpeg)
            msg = ws.receive_json()
        assert msg["buffer_full"] is True

        # sliding window: one more frame, still full, predicts again
        ws.send_bytes(jpeg)
        msg = ws.receive_json()
        assert msg["buffer_full"] is True
        assert msg["prediction"] in {"fall", "adl"}


def test_websocket_handles_undecodable_frame(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, window_size=5)
    client = TestClient(app)

    with client.websocket_connect("/ws/predict") as ws:
        ws.send_bytes(b"not a real jpeg")
        msg = ws.receive_json()
        assert "error" in msg


def test_live_engine_matches_manually_run_feature_pipeline(tmp_path):
    """Sanity cross-check (the plan's step 2 of the WebSocket exit
    checks): LiveInferenceEngine.predict_window must use the exact same
    normalize -> features pipeline as everything else in the project, not
    a subtly different one written just for the live path.
    """
    from api.inference import LiveInferenceEngine

    checkpoint_path = _make_checkpoint(tmp_path, window_size=5)
    live_engine = LiveInferenceEngine(str(checkpoint_path), threshold=0.5)

    raw_keypoints = np.stack([_standing_keypoints() for _ in range(5)], axis=0)

    sequence = normalize_pose_sequence(interpolate_missing_frames(raw_keypoints))
    features = assemble_feature_vector(sequence)
    with torch.no_grad():
        logits = live_engine.model(torch.from_numpy(features[None]).float())
        expected_prob = torch.softmax(logits, dim=-1)[0, 1].item()
    expected_prediction = "fall" if expected_prob >= 0.5 else "adl"
    expected_confidence = expected_prob if expected_prediction == "fall" else 1 - expected_prob

    prediction, confidence = live_engine.predict_window(raw_keypoints)

    assert prediction == expected_prediction
    assert confidence == pytest.approx(expected_confidence)


def test_live_engine_returns_none_for_all_nan_window(tmp_path):
    from api.inference import LiveInferenceEngine

    checkpoint_path = _make_checkpoint(tmp_path, window_size=5)
    live_engine = LiveInferenceEngine(str(checkpoint_path), threshold=0.5)

    raw_keypoints = np.full((5, NUM_LANDMARKS, NUM_CHANNELS), np.nan)

    prediction, confidence = live_engine.predict_window(raw_keypoints)

    assert prediction is None
    assert confidence is None
