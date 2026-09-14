"""M3 - PyTorch Dataset/DataLoader over extracted feature windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FallWindowDataset(Dataset):
    """Loads windows/labels from the per-video .npz files written by
    scripts/run_extraction.py, filtered to one split ("train"/"val"/"test").

    Each .npz already carries its own `split` (assigned by M1's per-scene,
    per-label stratified split) and `scene`, so this just concatenates
    windows across every matching file - no additional splitting happens
    here, which keeps the train/val/test boundary defined in exactly one
    place (inventory.py).
    """

    def __init__(self, processed_dir: str | Path, split: str):
        self.processed_dir = Path(processed_dir)
        self.split = split

        windows, labels, video_ids = [], [], []
        for npz_path in sorted(self.processed_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            if str(data["split"]) != split or data["windows"].shape[0] == 0:
                continue
            windows.append(data["windows"])
            labels.append(data["labels"])
            video_ids.extend([str(data["video_id"])] * data["windows"].shape[0])

        if windows:
            self.windows = np.concatenate(windows, axis=0).astype(np.float32)
            self.labels = np.concatenate(labels, axis=0).astype(np.int64)
        else:
            self.windows = np.empty((0, 0, 0), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int64)
        self.video_ids = video_ids

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.windows[idx]), torch.tensor(self.labels[idx])

    @property
    def input_dim(self) -> int:
        return self.windows.shape[-1] if self.windows.size else 0

    def class_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))


def make_class_weights(dataset: FallWindowDataset, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Le2i windows skew heavily ADL (a fall is a few seconds inside a much
    longer clip), so an unweighted loss would let the model coast to a
    high accuracy that never predicts "fall" at all.
    """
    counts = dataset.class_counts()
    weights = torch.ones(num_classes)
    total = sum(counts.values())
    if total == 0:
        return weights
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        weights[cls] = total / (num_classes * count) if count > 0 else 0.0
    return weights
