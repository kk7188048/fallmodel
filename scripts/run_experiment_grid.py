#!/usr/bin/env python
"""CLI: loop over M3.5 experiment configs."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_dir", default="configs/grid")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
