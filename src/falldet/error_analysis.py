"""M4 - error analysis: find and visualize misclassified test windows.

Everything here operates on the *normalized feature windows* already saved
to `.npz` by M2 (see features.py for the exact column layout), not raw
video - so no re-extraction or original video access is needed to run
this. The (x, y) skeleton positions used for plotting are the first 66
columns of the feature vector: hip-centered, torso-scaled coordinates, the
same space the model itself sees.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from falldet.dataset import FallWindowDataset
from falldet.model import FallDetectionModel
from falldet.pose import NUM_LANDMARKS

# (landmark_a, landmark_b) index pairs forming the standard MediaPipe pose
# skeleton - just enough edges to make a recognizable stick figure.
SKELETON_EDGES = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    (0, 11), (0, 12),  # nose to shoulders (rough head/neck line)
]

# Column offsets into the 167-dim feature vector (see features.py).
_XY_END = NUM_LANDMARKS * 2  # 66
_VIS_END = _XY_END + NUM_LANDMARKS  # 99
_VEL_END = _VIS_END + NUM_LANDMARKS * 2  # 165
_TORSO_ANGLE_COL = _VEL_END  # 165


def load_model(checkpoint_path: str | Path) -> FallDetectionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FallDetectionModel.from_config(checkpoint["config"], input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_diagnostic_features(window: np.ndarray) -> dict:
    """Cheap heuristics to help bucket a failure without re-watching every
    clip. window: (T, 167) feature array for one window.
    """
    xy_velocity = window[:, _VIS_END:_VEL_END].reshape(-1, NUM_LANDMARKS, 2)
    visibility = window[:, _XY_END:_VIS_END]
    torso_angle = window[:, _TORSO_ANGLE_COL]

    hip_y_velocity = xy_velocity[:, 23:25, 1]  # left/right hip, y-component
    return {
        "max_hip_velocity": float(np.abs(hip_y_velocity).max()),
        "torso_angle_range": float(torso_angle.max() - torso_angle.min()),
        "low_visibility_frac": float((visibility < 0.5).mean()),
    }


def get_predictions(model: FallDetectionModel, dataset: FallWindowDataset, threshold: float = 0.5) -> pd.DataFrame:
    """Run model over every window in `dataset`. One row per window,
    including correct predictions - filter to true_label != pred_label for
    misclassified-only analysis.
    """
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            window_t, label_t = dataset[i]
            logits, attn = model(window_t.unsqueeze(0), return_attention=True)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = int(prob >= threshold)
            video_id = dataset.video_ids[i]
            diag = compute_diagnostic_features(window_t.numpy())

            attn_weights = attn[0].numpy()
            max_attn_idx = int(attn_weights.argmax())
            window_size = attn_weights.shape[0]

            rows.append(
                {
                    "window_id": i,
                    "source_video": video_id,
                    "scene": video_id.split("/")[0],
                    "true_label": int(label_t.item()),
                    "pred_label": pred,
                    "confidence": prob,
                    "max_attention_frame": max_attn_idx,
                    "max_attention_near_edge": max_attn_idx < 2 or max_attn_idx >= window_size - 2,
                    "attention_weights": attn_weights,
                    **diag,
                }
            )
    return pd.DataFrame(rows)


def plot_skeleton_sequence(
    window: np.ndarray,
    out_path: Path,
    attention_weights: np.ndarray | None = None,
    n_frames: int = 8,
    title: str | None = None,
) -> None:
    """Plot a stick figure per frame, subsampled to n_frames across the
    window. If attention_weights given, each subplot's background is
    shaded by that frame's attention weight - a lighter/whiter background
    means the model paid little attention to that frame, a darker red
    means it paid a lot - directly answering "does attention find the
    moment of impact?" in one glance.
    """
    t = window.shape[0]
    idxs = np.linspace(0, t - 1, n_frames).astype(int)
    xy = window[:, :_XY_END].reshape(t, NUM_LANDMARKS, 2)

    fig, axes = plt.subplots(1, n_frames, figsize=(2.4 * n_frames, 2.6))
    if n_frames == 1:
        axes = [axes]

    max_w = float(attention_weights.max()) if attention_weights is not None else None

    for ax, idx in zip(axes, idxs):
        pts = xy[idx]
        if attention_weights is not None:
            w = float(attention_weights[idx])
            ax.set_facecolor(plt.cm.Reds(0.15 + 0.65 * (w / max_w if max_w > 0 else 0)))
        for a, b in SKELETON_EDGES:
            ax.plot([pts[a, 0], pts[b, 0]], [-pts[a, 1], -pts[b, 1]], "b-", linewidth=2)
        ax.scatter(pts[:, 0], -pts[:, 1], c="black", s=10, zorder=3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"t={idx}"
        if attention_weights is not None:
            label += f"\nattn={attention_weights[idx]:.2f}"
        ax.set_xlabel(label, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
