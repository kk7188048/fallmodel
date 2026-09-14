#!/usr/bin/env python
"""CLI: sweep the fall-decision threshold on an existing checkpoint.

The trained model uses argmax (threshold 0.5) by default, which is rarely
the right operating point for a rare, safety-critical positive class like
"fall" - this reuses the same checkpoint (no retraining) and reports
precision/recall/F1 at every threshold, plus its F1-maximizing pick, so
you can choose the actual tradeoff you want (e.g. bias toward recall to
minimize missed falls) instead of accepting whatever argmax happens to
give you.

Usage:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --split val
    python scripts/threshold_sweep.py --min_recall 0.9
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.evaluate import sweep_thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="defaults to <checkpoints_dir>/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--min_recall",
        type=float,
        default=None,
        help="if set, also report the highest-precision threshold that keeps fall recall >= this value",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_path = args.checkpoint or str(Path(config["paths"]["checkpoints_dir"]) / "best_model.pt")

    results = sweep_thresholds(config, checkpoint_path, split=args.split)

    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'f1':>7} {'tp':>5} {'fp':>5} {'fn':>5} {'tn':>5}")
    for r in results:
        print(
            f"{r['threshold']:>9.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>7.3f} "
            f"{r['tp']:>5d} {r['fp']:>5d} {r['fn']:>5d} {r['tn']:>5d}"
        )

    best_f1 = max(results, key=lambda r: r["f1"])
    print(
        f"\nbest F1: threshold={best_f1['threshold']:.2f} "
        f"precision={best_f1['precision']:.3f} recall={best_f1['recall']:.3f} f1={best_f1['f1']:.3f}"
    )

    if args.min_recall is not None:
        candidates = [r for r in results if r["recall"] >= args.min_recall]
        if not candidates:
            print(f"\nno threshold reaches recall >= {args.min_recall} on this split")
        else:
            best = max(candidates, key=lambda r: r["precision"])
            print(
                f"best precision with recall >= {args.min_recall}: threshold={best['threshold']:.2f} "
                f"precision={best['precision']:.3f} recall={best['recall']:.3f} f1={best['f1']:.3f}"
            )


if __name__ == "__main__":
    main()
