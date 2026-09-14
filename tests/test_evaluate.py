import numpy as np
import torch

from falldet.evaluate import evaluate, predict_probs, sweep_thresholds
from falldet.model import FallDetectionModel


def _write_npz(path, windows, labels, split, video_id="scene/video"):
    np.savez_compressed(
        path,
        windows=windows,
        labels=labels,
        frame_ranges=np.zeros((len(labels), 2), dtype=int),
        scene="scene",
        split=split,
        video_id=video_id,
    )


def _make_checkpoint(tmp_path, config, input_dim=8):
    model = FallDetectionModel.from_config(config, input_dim=input_dim)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": input_dim, "config": config},
        checkpoint_path,
    )
    return checkpoint_path


def _config(processed_dir):
    return {
        "paths": {"processed_dir": str(processed_dir)},
        "model": {
            "hidden_dim": 8,
            "num_layers": 1,
            "bidirectional": True,
            "attention_dim": 4,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "train": {"batch_size": 8},
    }


def test_predict_probs_shapes(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write_npz(processed_dir / "a.npz", np.random.randn(10, 5, 8), [0, 1] * 5, "test")

    config = _config(processed_dir)
    checkpoint_path = _make_checkpoint(tmp_path, config)

    labels, probs = predict_probs(config, checkpoint_path, split="test")
    assert labels.shape == (10,)
    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_probs_raises_on_missing_split(tmp_path):
    import pytest

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write_npz(processed_dir / "a.npz", np.random.randn(4, 5, 8), [0, 1, 0, 1], "train")

    config = _config(processed_dir)
    checkpoint_path = _make_checkpoint(tmp_path, config)

    with pytest.raises(ValueError):
        predict_probs(config, checkpoint_path, split="test")


def test_evaluate_returns_expected_keys(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write_npz(processed_dir / "a.npz", np.random.randn(20, 5, 8), [0, 1] * 10, "test")

    config = _config(processed_dir)
    checkpoint_path = _make_checkpoint(tmp_path, config)

    metrics = evaluate(config, checkpoint_path)
    assert set(metrics) == {"classification_report", "confusion_matrix", "pr_auc", "num_test_windows"}
    assert metrics["num_test_windows"] == 20
    assert len(metrics["confusion_matrix"]) == 2


def test_sweep_thresholds_extremes_are_monotonic(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    # Perfectly separable synthetic windows so threshold behavior is exact.
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([0, 1] * (n // 2))
    windows = rng.normal(size=(n, 5, 8)).astype(np.float32)
    windows[labels == 1] += 5.0
    _write_npz(processed_dir / "a.npz", windows, labels, "test")

    config = _config(processed_dir)
    model = FallDetectionModel.from_config(config, input_dim=8)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": 8, "config": config}, checkpoint_path
    )

    results = sweep_thresholds(config, checkpoint_path, thresholds=np.array([0.05, 0.5, 0.95]))
    assert [r["threshold"] for r in results] == [0.05, 0.5, 0.95]
    # recall must not increase as threshold rises
    recalls = [r["recall"] for r in results]
    assert recalls == sorted(recalls, reverse=True)
    for r in results:
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == n


def test_sweep_thresholds_min_recall_filter_logic():
    results = [
        {"threshold": 0.3, "precision": 0.2, "recall": 0.95, "f1": 0.33},
        {"threshold": 0.5, "precision": 0.4, "recall": 0.85, "f1": 0.54},
        {"threshold": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55},
    ]
    candidates = [r for r in results if r["recall"] >= 0.8]
    best = max(candidates, key=lambda r: r["precision"])
    assert best["threshold"] == 0.5
