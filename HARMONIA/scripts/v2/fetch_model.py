#!/usr/bin/env python3
"""Download the v2 model from its GitHub release and verify it (VPS, Docker build or any PC).

models/release.json (tracked in git) pins the release tag, asset name and sha256; every file of the
archive is checked again against the sha256 listed in its manifest.json.

    python scripts/v2/fetch_model.py            # -> models/harmonia_v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RELEASE_FILE = BASE_DIR / "models" / "release.json"
REPOSITORY = "Harmonia-EIP/Harmonia-AI"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_dir(model_dir: Path) -> None:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if sha256(model_dir / name) != info["sha256"]:
            raise RuntimeError(f"{name}: sha256 mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="harmonia_v2")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    release = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))[args.model]
    out = args.out or BASE_DIR / "models" / args.model
    if (out / "manifest.json").exists() and not args.force:
        try:
            verify_model_dir(out)
            print(f"{out} is already up to date")
            return 0
        except (RuntimeError, OSError, KeyError):
            print(f"{out} is incomplete or modified, downloading again")

    url = f"https://github.com/{REPOSITORY}/releases/download/{release['tag']}/{release['asset']}"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / release["asset"]
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, archive)  # nosec B310 - fixed https URL
        if sha256(archive) != release["sha256"]:
            raise SystemExit(f"{release['asset']}: sha256 mismatch, refusing to install")
        staging = Path(tmp) / "staging"
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        verify_model_dir(staging)
        out.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(out / path.name)
    print(f"installed {args.model} ({release['tag']}) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
