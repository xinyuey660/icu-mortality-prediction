#!/usr/bin/env python3
"""Download public PhysioNet Challenge 2012 Set A and its outcomes."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    archive_path = output_dir / "set-a.tar.gz"
    outcomes_path = output_dir / "Outcomes-a.txt"
    if args.force or not archive_path.exists():
        download(f"{BASE_URL}/set-a.tar.gz", archive_path)
    if args.force or not outcomes_path.exists():
        download(f"{BASE_URL}/Outcomes-a.txt", outcomes_path)
    set_dir = output_dir / "set-a"
    if args.force or not set_dir.exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()

