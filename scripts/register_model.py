#!/usr/bin/env python
"""CLI: register a trained checkpoint in the model registry (Postgres).

Requires FALLDET_DATABASE_URL to be set.

Usage:
    python scripts/register_model.py --checkpoint checkpoints_window45/best_model.pt --tag production
    python scripts/register_model.py --list
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.registry import list_models, register_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path to a best_model.pt checkpoint")
    parser.add_argument("--tag", default="production", help="e.g. production, staging, canary")
    parser.add_argument("--list", action="store_true", help="list all registered models instead of registering one")
    args = parser.parse_args()

    if args.list:
        models = list_models()
        if not models:
            print("no models registered yet")
            return
        for m in models:
            marker = "ACTIVE" if m["is_active"] else ""
            print(
                f"[{m['id']}] tag={m['tag']:12s} window={m['window_size']} stride={m['stride']} "
                f"hidden_dim={m['hidden_dim']} sha256={m['sha256'][:12]}... {marker}"
            )
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless using --list")

    info = register_model(args.checkpoint, args.tag)
    print(f"registered model id={info['id']} tag={info['tag']}")
    print(f"  file_path={info['file_path']}")
    print(f"  sha256={info['sha256']}")
    print(f"  window_size={info['window_size']} stride={info['stride']} hidden_dim={info['hidden_dim']}")


if __name__ == "__main__":
    main()
