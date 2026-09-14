"""M3 - test-set metrics: confusion matrix, PR-AUC, classification report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from falldet.dataset import FallWindowDataset
from falldet.model import FallDetectionModel


def _load_model(checkpoint: dict) -> FallDetectionModel:
    model = FallDetectionModel.from_config(checkpoint["config"], input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_probs(config: dict, checkpoint_path: str | Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Run the checkpointed model over one split and return (labels, fall_probs)."""
    paths = config["paths"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ds = FallWindowDataset(paths["processed_dir"], split)
    if len(ds) == 0:
        raise ValueError(
            f"no {split} windows found under {paths['processed_dir']} - "
            "check the split assignment in data/inventory.json"
        )

    model = _load_model(checkpoint)
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for windows, labels in loader:
            logits = model(windows)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def evaluate(config: dict, checkpoint_path: str | Path) -> dict:
    all_labels, all_probs = predict_probs(config, checkpoint_path, split="test")
    all_preds = (all_probs >= 0.5).astype(int)

    report = classification_report(
        all_labels, all_preds, target_names=["adl", "fall"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    pr_auc = (
        average_precision_score(all_labels, all_probs)
        if len(set(all_labels.tolist())) > 1
        else float("nan")
    )

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "pr_auc": pr_auc,
        "num_test_windows": len(all_labels),
    }


def sweep_thresholds(
    config: dict,
    checkpoint_path: str | Path,
    thresholds: np.ndarray | None = None,
    split: str = "test",
) -> list[dict]:
    """Score the fall class at each decision threshold instead of the
    default argmax (0.5).

    A lower threshold trades fall-precision for fall-recall (fewer missed
    falls, more false alarms) and vice versa - useful here since the
    trained model has a large recall/precision gap (0.85/0.42 at 0.5) that
    a fixed argmax can't fix, but picking a different operating point can,
    with zero retraining.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    labels, probs = predict_probs(config, checkpoint_path, split=split)

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return results
