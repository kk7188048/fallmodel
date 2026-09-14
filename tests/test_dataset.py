import numpy as np
import torch

from falldet.dataset import FallWindowDataset, make_class_weights


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


def test_dataset_filters_by_split_and_concatenates(tmp_path):
    _write_npz(tmp_path / "a.npz", np.zeros((3, 5, 4)), [0, 1, 0], "train", "s/a")
    _write_npz(tmp_path / "b.npz", np.zeros((2, 5, 4)), [1, 1], "train", "s/b")
    _write_npz(tmp_path / "c.npz", np.zeros((4, 5, 4)), [0, 0, 0, 0], "val", "s/c")

    train_ds = FallWindowDataset(tmp_path, "train")
    val_ds = FallWindowDataset(tmp_path, "val")

    assert len(train_ds) == 5
    assert len(val_ds) == 4
    assert train_ds.input_dim == 4


def test_dataset_skips_empty_windows(tmp_path):
    _write_npz(tmp_path / "empty.npz", np.zeros((0, 5, 4)), [], "test")
    test_ds = FallWindowDataset(tmp_path, "test")
    assert len(test_ds) == 0
    assert test_ds.input_dim == 0


def test_dataset_getitem_returns_tensors(tmp_path):
    windows = np.arange(2 * 5 * 4).reshape(2, 5, 4).astype(float)
    _write_npz(tmp_path / "a.npz", windows, [1, 0], "train")

    ds = FallWindowDataset(tmp_path, "train")
    window, label = ds[0]
    assert isinstance(window, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert window.shape == (5, 4)
    assert window.dtype == torch.float32
    assert label.item() == 1


def test_dataset_missing_split_is_empty(tmp_path):
    _write_npz(tmp_path / "a.npz", np.zeros((3, 5, 4)), [0, 1, 0], "train")
    test_ds = FallWindowDataset(tmp_path, "test")
    assert len(test_ds) == 0


def test_class_counts(tmp_path):
    _write_npz(tmp_path / "a.npz", np.zeros((4, 5, 4)), [0, 0, 0, 1], "train")
    ds = FallWindowDataset(tmp_path, "train")
    assert ds.class_counts() == {0: 3, 1: 1}


def test_make_class_weights_favors_rare_class(tmp_path):
    _write_npz(tmp_path / "a.npz", np.zeros((10, 5, 4)), [0] * 9 + [1], "train")
    ds = FallWindowDataset(tmp_path, "train")
    weights = make_class_weights(ds)
    assert weights[1] > weights[0]


def test_make_class_weights_handles_missing_class(tmp_path):
    _write_npz(tmp_path / "a.npz", np.zeros((3, 5, 4)), [0, 0, 0], "train")
    ds = FallWindowDataset(tmp_path, "train")
    weights = make_class_weights(ds)
    assert weights[1] == 0.0
    assert torch.isfinite(weights).all()


def test_make_class_weights_empty_dataset_returns_ones(tmp_path):
    ds = FallWindowDataset(tmp_path, "train")
    weights = make_class_weights(ds)
    assert torch.allclose(weights, torch.ones(2))
