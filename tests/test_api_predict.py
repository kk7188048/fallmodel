"""FastAPI TestClient tests for /predict.

The model is real (tiny, untrained - its actual predictions are
meaningless here, only the wiring matters). `process_video` is mocked so
these tests run in milliseconds with no MediaPipe/real-video dependency -
that pipeline is already covered by tests/test_pipeline.py; this file's
job is to prove the API layer (upload handling, error mapping, response
shape, cleanup) is correct, not to re-test pose extraction.
"""
import importlib
import os

import numpy as np
import torch
from fastapi.testclient import TestClient

from falldet.features import FEATURE_DIM
from falldet.model import FallDetectionModel


def _make_checkpoint(tmp_path):
    config = {
        "model": {
            "hidden_dim": 8, "num_layers": 1, "bidirectional": True,
            "attention_dim": 4, "dropout": 0.0, "num_classes": 2,
        },
        "pose": {"model_complexity": 0, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
        "windowing": {"window_size": 30, "stride": 15},
    }
    model = FallDetectionModel.from_config(config, input_dim=FEATURE_DIM)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": FEATURE_DIM, "config": config},
        checkpoint_path,
    )
    return checkpoint_path


def _make_app(tmp_path, monkeypatch, fake_process_video):
    checkpoint_path = _make_checkpoint(tmp_path)
    monkeypatch.setenv("FALLDET_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("FALLDET_THRESHOLD", "0.5")

    import api.inference as inference_module

    monkeypatch.setattr(inference_module, "process_video", fake_process_video)

    import api.main as main_module

    importlib.reload(main_module)
    return main_module.app


def _result(n_windows, all_nan=False):
    if all_nan:
        windows = np.full((n_windows, 30, FEATURE_DIM), np.nan, dtype=np.float32)
    else:
        windows = np.random.randn(n_windows, 30, FEATURE_DIM).astype(np.float32)
    return {
        "windows": windows,
        "labels": np.zeros(n_windows, dtype=int),
        "frame_ranges": np.zeros((n_windows, 2), dtype=int),
        "num_frames": 90,
        "detection_rate": 0.0 if all_nan else 1.0,
    }


def _fake_process_video_ok(video_path, **kwargs):
    return _result(3)


def _fake_process_video_no_detection(video_path, **kwargs):
    return _result(3, all_nan=True)


def _fake_process_video_too_short(video_path, **kwargs):
    return _result(0)


def test_predict_returns_valid_response(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {"fall", "adl"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0
    assert body["n_windows_evaluated"] == 3


def test_predict_rejects_bad_extension(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.txt", b"not a video", "text/plain")})

    assert response.status_code == 400


def test_predict_handles_no_detection(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_no_detection)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    assert response.status_code == 422
    assert "no person detected" in response.json()["detail"]


def test_predict_handles_too_short_video(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_too_short)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    assert response.status_code == 422
    assert "too short" in response.json()["detail"] or "shorter" in response.json()["detail"]


def test_predict_cleans_up_temp_file(tmp_path, monkeypatch):
    captured = {}

    def fake_process_video(video_path, **kwargs):
        captured["path"] = video_path
        return _result(3)

    app = _make_app(tmp_path, monkeypatch, fake_process_video)
    client = TestClient(app)

    client.post("/predict", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    assert "path" in captured
    assert not os.path.exists(captured["path"])


def test_health_endpoint(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_api_key_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("FALLDET_API_KEY", "secret123")
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_predict_succeeds_with_correct_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FALLDET_API_KEY", "secret123")
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        headers={"X-API-Key": "secret123"},
    )

    assert response.status_code == 200


def test_predict_rejects_wrong_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FALLDET_API_KEY", "secret123")
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        headers={"X-API-Key": "wrong"},
    )

    assert response.status_code == 401


def test_predict_open_when_no_api_key_configured(tmp_path, monkeypatch):
    # FALLDET_API_KEY intentionally not set - default, backward-compatible behavior.
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    assert response.status_code == 200


def test_predict_rejects_oversized_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("FALLDET_MAX_UPLOAD_BYTES", "10")  # tiny limit for the test
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("clip.mp4", b"x" * 1000, "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_predict_accepts_upload_within_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("FALLDET_MAX_UPLOAD_BYTES", "10000")
    app = _make_app(tmp_path, monkeypatch, _fake_process_video_ok)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
    )

    assert response.status_code == 200
