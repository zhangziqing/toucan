#!/usr/bin/env python3
"""Assemble a movable Nanhu suite without compiling or launching any GPU work."""
import argparse
from pathlib import Path

from package import Package, assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="Root containing models/p10, models/p20, etc.")
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="Portable toucanGPURT source tree")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(Package(Path(__file__).resolve().parent, args.artifact_root),
             args.runtime_root, args.output)
    print("Verified movable package: " + str(args.output))


if __name__ == "__main__":
    main()
