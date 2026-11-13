#!/usr/bin/env python3
"""Download the third-party preset banks used by the v3 research pipeline, and the DX7 FM core sources.

The banks are not redistributed with Harmonia: this script fetches them from their source, pinned by
git commit or sha256, into a local directory (default ~/Datasets/harmonia-presets).

    python scripts/v3/fetch_presets.py                       # every bank
    python scripts/v3/fetch_presets.py --only surge obxf     # a subset
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess  # nosec B404 - fixed git commands, no shell
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_OUT = Path.home() / "Datasets" / "harmonia-presets"

# name -> source. "git": sparse checkout of `paths` at `commit`; "url": archive checked by sha256.
SOURCES = {
    "surge": {
        "git": "https://github.com/surge-synthesizer/surge.git",
        "commit": "348cfb3d0bd081797cfd6505d5f8d3ffd6f34c49",
        "paths": ["resources/data/patches_factory", "resources/data/patches_3rdparty"],
        "license": "GPL-3.0",
    },
    "obxf": {
        "git": "https://github.com/surge-synthesizer/OB-Xf.git",
        "commit": "b08ffb6ab6cfa0f66cb057e855149ab640de0f78",
        "paths": ["assets/installer/Surge Synth Team/OB-Xf/Patches"],
        "license": "GPL-3.0",
    },
    "dexed_src": {  # msfa FM core (Apache-2.0) for the DX7 renderer, see native/dx7
        "git": "https://github.com/asb2m10/dexed.git",
        "commit": "2e182b3db85c09083ab13c8b9b00565ce7d9ff85",
        "paths": ["Source/msfa"],
        "license": "Apache-2.0 (Source/msfa)",
    },
    "dexed_builtin": {
        "url": "https://raw.githubusercontent.com/asb2m10/dexed/2e182b3db85c09083ab13c8b9b00565ce7d9ff85/assets/builtin_pgm.zip",
        "sha256": "6ab2487b1e76f58ba748858ecf0867b7cc91708cfc46a0dcec6f2922bf9cbb8d",
        "license": "GPL-3.0 (Dexed repository)",
    },
    "dexed_carts": {
        "url": "http://hsjp.eu/downloads/Dexed/Dexed_cart_1.0.zip",
        "sha256": "fc595fdf26a1b7e4e77f724b62e5ed015a9e99790971e02de401663cef676904",
        "license": "unspecified (community DX7 cartridge collection)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)  # nosec B603 B607 - fixed git invocation


def fetch_git(name: str, src: dict, out: Path) -> None:
    dest = out / name
    if (dest / ".git").exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True,  # nosec B603 B607
                              check=True).stdout.strip()
        if head == src["commit"]:
            print(f"{name}: already at {head[:10]}")
            return
    else:
        git("clone", "--filter=blob:none", "--no-checkout", "--sparse", src["git"], str(dest))
    git("sparse-checkout", "set", "--no-cone", *[f"/{p}/" for p in src["paths"]], cwd=dest)
    git("fetch", "--filter=blob:none", "origin", src["commit"], cwd=dest)
    git("checkout", "--detach", src["commit"], cwd=dest)
    print(f"{name}: checked out {src['commit'][:10]}")


def fetch_url(name: str, src: dict, out: Path) -> None:
    dest = out / name
    archive = out / f"{name}.zip"
    if not archive.exists():
        print(f"{name}: downloading {src['url']}")
        urllib.request.urlretrieve(src["url"], archive)  # nosec B310 - pinned https/http source checked below
    digest = sha256(archive)
    if src["sha256"] and digest != src["sha256"]:
        archive.unlink()
        raise RuntimeError(f"{name}: sha256 mismatch ({digest})")
    if not src["sha256"]:
        print(f"{name}: sha256 {digest} (not pinned yet)")
    if not dest.exists():
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError(f"{name}: unsafe path in archive: {member}")
            zf.extractall(dest)
    print(f"{name}: ready in {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only or SOURCES:
        src = SOURCES[name]
        (fetch_git if "git" in src else fetch_url)(name, src, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
