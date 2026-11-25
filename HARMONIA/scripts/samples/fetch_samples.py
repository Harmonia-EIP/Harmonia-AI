#!/usr/bin/env python3
"""Download the free sample libraries the sample source is built from, at pinned versions (#41).

The recordings are never in git (GitHub refuses files over 100 MB, and the libraries weigh several GB):
they are downloaded here, outside the repository, checked against the sha256 pinned in
samples/sources.json, and unpacked safely (no absolute paths, no links out of the folder).

    python scripts/samples/fetch_samples.py                 # -> ~/Datasets/harmonia-samples/<library>/
    python scripts/samples/fetch_samples.py --only salamander
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCES = BASE_DIR / "samples" / "sources.json"
ROOT = Path.home() / "Datasets" / "harmonia-samples"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str) -> None:
    """Download once; a file already there with the right sha256 is kept."""
    if target.exists() and sha256(target) == expected:
        return
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "harmonia-fetch-samples"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as f:  # nosec B310 - pinned https URLs
        shutil.copyfileobj(response, f, length=1 << 20)
    found = sha256(part)
    if found != expected:
        part.unlink()
        raise SystemExit(f"{target.name}: sha256 {found}, expected {expected} (the source changed: check it and re-pin)")
    part.rename(target)


def unpack(archive: Path, out: Path) -> int:
    """Unpack into `out`, dropping the archive's top folder; refuses unsafe members. Returns the file count."""
    tmp = out.with_name(out.name + ".unpacking")
    if tmp.exists():
        shutil.rmtree(tmp)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")  # nosec B202 - the "data" filter refuses absolute paths and outside links
    tops = [p for p in tmp.iterdir() if not p.name.startswith(".")]
    root = tops[0] if len(tops) == 1 and tops[0].is_dir() else tmp
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(root), out)
    shutil.rmtree(tmp, ignore_errors=True)
    return sum(1 for p in out.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="where the libraries go (outside the repository)")
    parser.add_argument("--only", nargs="+", default=None, help="library names of samples/sources.json")
    args = parser.parse_args()
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    for name, source in sources.items():
        if args.only and name not in args.only:
            continue
        archive = args.root / "downloads" / f"{name}{source['archive_suffix']}"
        download(source["url"], archive, source["sha256"])
        files = unpack(archive, args.root / name)
        print(f"{name}: {files} files in {args.root / name} ({source['license']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
