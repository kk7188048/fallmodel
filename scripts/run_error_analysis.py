#!/usr/bin/env python
"""CLI: error analysis on a trained checkpoint's test-set predictions.

Usage:
    python scripts/run_error_analysis.py
    python scripts/run_error_analysis.py --model checkpoints/best_model.pt --out outputs/error_analysis
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.dataset import FallWindowDataset
from falldet.error_analysis import get_predictions, load_model, plot_skeleton_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="defaults to <checkpoints_dir>/best_model.pt")
    parser.add_argument("--out", default="outputs/error_analysis")
    parser.add_argument("--threshold", type=float, default=0.55, help="see scripts/threshold_sweep.py")
    parser.add_argument(
        "--max_plots", type=int, default=20, help="cap on misclassified skeleton plots (can be a lot)"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_path = args.model or str(Path(config["paths"]["checkpoints_dir"]) / "best_model.pt")
    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    examples_dir = out_dir / "attention_examples"
    plots_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model from {checkpoint_path} ...")
    model = load_model(checkpoint_path)
    dataset = FallWindowDataset(config["paths"]["processed_dir"], "test")
    if len(dataset) == 0:
        raise ValueError("no test windows found - run scripts/run_extraction.py first")

    print(f"running model over {len(dataset)} test windows ...")
    df = get_predictions(model, dataset, threshold=args.threshold)

    misclassified = df[df["true_label"] != df["pred_label"]].copy()
    false_negatives = misclassified[(misclassified["true_label"] == 1) & (misclassified["pred_label"] == 0)]
    false_positives = misclassified[(misclassified["true_label"] == 0) & (misclassified["pred_label"] == 1)]

    print(f"\ntotal windows: {len(df)}")
    print(f"misclassified: {len(misclassified)} ({len(misclassified) / len(df) * 100:.1f}%)")
    print(f"  false negatives (missed falls, prioritize these): {len(false_negatives)}")
    print(f"  false positives (false alarms): {len(false_positives)}")

    misclassified["bucket"] = ""
    misclassified["notes"] = ""
    csv_cols = [
        "window_id", "source_video", "scene", "true_label", "pred_label", "confidence",
        "max_hip_velocity", "torso_angle_range", "low_visibility_frac",
        "max_attention_frame", "max_attention_near_edge", "bucket", "notes",
    ]
    csv_path = out_dir / "misclassified_index.csv"
    misclassified[csv_cols].to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path} - fill in the 'bucket' and 'notes' columns by hand after looking at the plots")

    # Skeleton+attention plots for misclassified windows, false negatives first
    # (missed falls matter more - see M3's class-weighted loss for the same reasoning).
    to_plot = pd.concat([false_negatives, false_positives]).head(args.max_plots) if len(misclassified) else misclassified
    for _, row in to_plot.iterrows():
        window = dataset.windows[row["window_id"]]
        kind = "FN" if row["true_label"] == 1 else "FP"
        out_path = plots_dir / f"{row['window_id']}_{kind}_{row['source_video'].replace('/', '__')}.png"
        plot_skeleton_sequence(
            window,
            out_path,
            attention_weights=row["attention_weights"],
            title=f"{row['source_video']} true={row['true_label']} pred={row['pred_label']} conf={row['confidence']:.2f}",
        )
    print(f"wrote {len(to_plot)} misclassified skeleton plots -> {plots_dir}")

    # One high-confidence correct fall, for comparison.
    correct_falls = df[(df["true_label"] == 1) & (df["pred_label"] == 1)]
    if len(correct_falls):
        best = correct_falls.loc[correct_falls["confidence"].idxmax()]
        window = dataset.windows[best["window_id"]]
        plot_skeleton_sequence(
            window,
            examples_dir / "correct_fall_example.png",
            attention_weights=best["attention_weights"],
            title=f"CORRECT: {best['source_video']} conf={best['confidence']:.2f}",
        )
        print(f"wrote {examples_dir / 'correct_fall_example.png'}")

    # Worst failure: prioritize the most-confidently-wrong false negative
    # (missed a fall while being very sure it wasn't one); fall back to the
    # worst false positive if there are no false negatives.
    if len(false_negatives):
        worst = false_negatives.loc[false_negatives["confidence"].idxmin()]
    elif len(false_positives):
        worst = false_positives.loc[false_positives["confidence"].idxmax()]
    else:
        worst = None
    if worst is not None:
        window = dataset.windows[worst["window_id"]]
        plot_skeleton_sequence(
            window,
            examples_dir / "failure_example.png",
            attention_weights=worst["attention_weights"],
            title=f"FAILURE: {worst['source_video']} true={worst['true_label']} conf={worst['confidence']:.2f}",
        )
        print(f"wrote {examples_dir / 'failure_example.png'}")


if __name__ == "__main__":
    main()
