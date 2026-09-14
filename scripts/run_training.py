#!/usr/bin/env python
"""CLI: train one config, log to MLflow."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.evaluate import evaluate
from falldet.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    result = train(config)
    print(f"\nbest val_loss: {result['best_val_loss']:.4f}")
    print(f"checkpoint: {result['checkpoint_path']}")

    try:
        metrics = evaluate(config, result["checkpoint_path"])
    except ValueError as e:
        print(f"\nskipping test evaluation: {e}")
        return

    print(f"\ntest set ({metrics['num_test_windows']} windows):")
    print(json.dumps(metrics["classification_report"], indent=2))
    print("confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
