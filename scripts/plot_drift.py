#!/usr/bin/env python
"""CLI: plot rolling-mean prediction confidence from the inference log.

This is a STUB, not a full monitoring system: no alerting, no thresholds,
just a plot. The reasoning: if the *input distribution* changes (a new
camera, a new room, different lighting than anything in training), the
model's confidence tends to drift down even before you have ground-truth
labels to measure real accuracy against - so a sustained drop in
rolling-mean confidence is a cheap, label-free proxy for "something about
the inputs changed, go look." It is not proof of a problem, and it can't
tell you *what* changed - it's a prompt to go investigate, nothing more.

Usage:
    python scripts/plot_drift.py
    python scripts/plot_drift.py --log outputs/monitoring/inference_log.csv --window 50
"""
import argparse
import sys
from pathlib import Path

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from api.monitoring import read_recent_logs


def _load_dataframe(log_path: Path, limit: int) -> pd.DataFrame:
    rows = read_recent_logs(limit=limit, csv_path=log_path)
    if not rows:
        raise FileNotFoundError(
            f"no inference log rows found (checked Postgres if FALLDET_DATABASE_URL is set, "
            f"else {log_path}) - run some /predict requests first"
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("inference log has no rows yet")

    return df.sort_values("timestamp").reset_index(drop=True)


def latest_rolling_confidence(log_path: Path, window: int = 50, limit: int = 5000) -> float:
    """The most recent rolling-mean confidence value - what
    check_drift_alert() compares against a threshold.
    """
    df = _load_dataframe(Path(log_path), limit)
    effective_window = min(window, len(df))
    rolling = df["confidence"].rolling(effective_window, min_periods=1).mean()
    return float(rolling.iloc[-1])


def check_drift_alert(rolling_confidence: float, threshold: float) -> bool:
    """True if the latest rolling-mean confidence has dropped below
    `threshold` - the cheap, label-free drift signal this script is built
    around (see module docstring). True means "go investigate", not
    "proven broken."
    """
    return rolling_confidence < threshold


def plot_confidence_drift(log_path: Path, window: int = 50, out_path: Path = None, limit: int = 5000) -> Path:
    """Reads from Postgres if FALLDET_DATABASE_URL is set, else from the
    CSV at log_path - see api.monitoring.read_recent_logs(). log_path is
    only used for the CSV fallback and for error messages.
    """
    log_path = Path(log_path)
    df = _load_dataframe(log_path, limit)
    effective_window = min(window, len(df))
    df["rolling_confidence"] = df["confidence"].rolling(effective_window, min_periods=1).mean()

    out_path = Path(out_path) if out_path else log_path.parent / "confidence_drift.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["confidence"], "o", alpha=0.25, markersize=3, label="per-request confidence")
    ax.plot(df.index, df["rolling_confidence"], "r-", linewidth=2, label=f"rolling mean (window={effective_window})")
    ax.set_xlabel("request # (chronological)")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1)
    ax.set_title(f"Inference confidence over time - {len(df)} requests")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/monitoring/inference_log.csv")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--alert_threshold",
        type=float,
        default=None,
        help="if set, exit 1 and print a warning when the latest rolling-mean confidence "
        "drops below this value - cheap, automatable (cron/CI) drift signal, NOT real "
        "alerting infrastructure (no paging, no dashboards) - see module docstring",
    )
    args = parser.parse_args()

    out_path = plot_confidence_drift(Path(args.log), window=args.window, out_path=args.out)
    print(f"wrote {out_path}")

    if args.alert_threshold is not None:
        latest = latest_rolling_confidence(Path(args.log), window=args.window)
        if check_drift_alert(latest, args.alert_threshold):
            print(
                f"ALERT: rolling-mean confidence {latest:.3f} is below threshold "
                f"{args.alert_threshold:.3f} - this does not prove a problem, but the input "
                f"distribution may have shifted (new camera/room/lighting) since training; "
                f"go investigate.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"ok: rolling-mean confidence {latest:.3f} >= threshold {args.alert_threshold:.3f}")


if __name__ == "__main__":
    main()
