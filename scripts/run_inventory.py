#!/usr/bin/env python
"""CLI: python scripts/run_inventory.py --raw_dir data/raw"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.inventory import build_inventory, write_inventory, write_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no_probe",
        action="store_true",
        help="skip opening each video with OpenCV to read frame_count/fps/size",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    train_frac = args.train_frac if args.train_frac is not None else 0.7
    val_frac = args.val_frac if args.val_frac is not None else 0.15
    seed = args.seed if args.seed is not None else config["train"]["seed"]

    records, summary = build_inventory(
        raw_dir=args.raw_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        probe=not args.no_probe,
    )

    inventory_path = Path(config["paths"].get("inventory_file", "data/inventory.json"))
    splits_path = Path(config["paths"]["splits_file"])
    write_inventory(records, inventory_path)
    write_splits(records, splits_path)

    print(f"discovered {summary['total_videos']} videos")
    print("\nby scene:")
    for scene, stats in sorted(summary["by_scene"].items()):
        print(f"  {scene:16s} fall={stats['fall']:3d}  adl={stats['adl']:3d}  warnings={stats['warnings']}")
    print("\nby split:")
    for split, stats in summary["by_split"].items():
        print(f"  {split:12s} fall={stats['fall']:3d}  adl={stats['adl']:3d}")

    warned = [r for r in records if r.warnings]
    if warned:
        print(f"\n{len(warned)} video(s) had warnings, e.g.:")
        for r in warned[:10]:
            print(f"  {r.video_id}: {'; '.join(r.warnings)}")

    print(f"\nwrote inventory -> {inventory_path}")
    print(f"wrote splits    -> {splits_path}")


if __name__ == "__main__":
    main()
