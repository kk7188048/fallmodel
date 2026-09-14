#!/usr/bin/env python
"""CLI: batch process_video() over all raw videos in the inventory."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.pipeline import process_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N videos (debugging)"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    inventory_path = Path(config["paths"].get("inventory_file", "data/inventory.json"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(inventory_path.read_text())
    if args.limit:
        inventory = inventory[: args.limit]

    pose_cfg = config["pose"]
    window_cfg = config["windowing"]

    total_windows = 0
    total_positive = 0
    low_detection = []

    for i, record in enumerate(inventory, start=1):
        video_path = raw_dir / record["path"]
        out_path = processed_dir / f"{record['video_id'].replace('/', '__')}.npz"

        print(f"[{i}/{len(inventory)}] {record['video_id']} ({record['label']}) ...", end=" ")

        result = process_video(
            str(video_path),
            fall_start_frame=record.get("fall_start_frame"),
            fall_end_frame=record.get("fall_end_frame"),
            model_complexity=pose_cfg["model_complexity"],
            min_detection_confidence=pose_cfg["min_detection_confidence"],
            min_tracking_confidence=pose_cfg["min_tracking_confidence"],
            window_size=window_cfg["window_size"],
            stride=window_cfg["stride"],
        )

        np.savez_compressed(
            out_path,
            windows=result["windows"],
            labels=result["labels"],
            frame_ranges=result["frame_ranges"],
            scene=record["scene"],
            split=record["split"],
            video_id=record["video_id"],
        )

        n_windows = result["windows"].shape[0]
        n_positive = int(result["labels"].sum())
        total_windows += n_windows
        total_positive += n_positive
        if result["detection_rate"] < 0.8:
            low_detection.append((record["video_id"], result["detection_rate"]))

        print(
            f"{result['num_frames']} frames, "
            f"detection_rate={result['detection_rate']:.2f}, "
            f"{n_windows} windows ({n_positive} positive) -> {out_path.name}"
        )

    print(f"\nwrote {len(inventory)} .npz files to {processed_dir}")
    print(f"total windows: {total_windows} ({total_positive} positive, "
          f"{total_windows - total_positive} negative)")
    if low_detection:
        print(f"\n{len(low_detection)} video(s) had pose detection_rate < 0.8:")
        for video_id, rate in low_detection:
            print(f"  {video_id}: {rate:.2f}")


if __name__ == "__main__":
    main()
