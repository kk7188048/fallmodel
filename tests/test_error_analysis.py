import numpy as np
import torch

from falldet.error_analysis import (
    compute_diagnostic_features,
    get_predictions,
    plot_skeleton_sequence,
)
from falldet.features import FEATURE_DIM
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


def test_compute_diagnostic_features_shapes_and_ranges():
    window = np.random.randn(30, FEATURE_DIM).astype(np.float32)
    diag = compute_diagnostic_features(window)
    assert set(diag) == {"max_hip_velocity", "torso_angle_range", "low_visibility_frac"}
    assert diag["max_hip_velocity"] >= 0
    assert diag["torso_angle_range"] >= 0
    assert 0.0 <= diag["low_visibility_frac"] <= 1.0


def test_get_predictions_columns_and_length(tmp_path):
    from falldet.dataset import FallWindowDataset

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    windows = np.random.randn(10, 30, FEATURE_DIM).astype(np.float32)
    labels = np.array([0, 1] * 5)
    _write_npz(processed_dir / "a.npz", windows, labels, "test", video_id="Coffee_room_01/video (1)")

    config = {
        "model": {
            "hidden_dim": 8, "num_layers": 1, "bidirectional": True,
            "attention_dim": 4, "dropout": 0.0, "num_classes": 2,
        },
    }
    model = FallDetectionModel.from_config(config, input_dim=FEATURE_DIM)
    model.eval()

    ds = FallWindowDataset(processed_dir, "test")
    df = get_predictions(model, ds, threshold=0.5)

    assert len(df) == 10
    assert set(df["scene"]) == {"Coffee_room_01"}
    assert df["max_attention_frame"].between(0, 29).all()
    assert set(df["max_attention_near_edge"].unique()).issubset({True, False})


def test_plot_skeleton_sequence_writes_file(tmp_path):
    window = np.random.randn(30, FEATURE_DIM).astype(np.float32)
    attn = np.random.rand(30).astype(np.float32)
    attn /= attn.sum()
    out_path = tmp_path / "plot.png"

    plot_skeleton_sequence(window, out_path, attention_weights=attn, title="test")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_skeleton_sequence_without_attention(tmp_path):
    window = np.random.randn(30, FEATURE_DIM).astype(np.float32)
    out_path = tmp_path / "plot_no_attn.png"

    plot_skeleton_sequence(window, out_path)

    assert out_path.exists()
