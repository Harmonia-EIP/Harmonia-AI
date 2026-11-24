#!/usr/bin/env python3
"""Install a sample pack from the samples release (samples/release.json, #43).

- listen: what the AI server plays while it generates (models/samples, or $HARMONIA_SAMPLES_DIR at run time);
- full: what the plugin plays (its samples folder, e.g. ~/Library/Application Support/Harmonia/Samples).

Each archive is checked against the sha256 pinned in samples/release.json, then unpacked safely (no absolute
paths, no links out of the folder). The archives of the full pack (one per library) unpack into the same
folder: each carries the whole catalogue.

    python scripts/samples/fetch_pack.py --pack listen                  # -> models/samples
    python scripts/samples/fetch_pack.py --pack full --out "$HOME/Library/Application Support/Harmonia/Samples"
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.samples.fetch_samples import download  # noqa: E402

RELEASE = BASE_DIR / "samples" / "release.json"
REPOSITORY = "Harmonia-EIP/Harmonia-AI"


def installed_marker(out: Path, archive: dict) -> Path:
    return out / f".{archive['asset']}.sha256"


def install(pack: str, out: Path, base_url: str, release: dict) -> int:
    """Download, check and unpack every archive of `pack` into `out`; already installed ones are skipped.
    Returns the number of archives unpacked."""
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for archive in release["packs"][pack]:
        marker = installed_marker(out, archive)
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == archive["sha256"]:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / archive["asset"]
            download(f"{base_url}/{archive['asset']}", path, archive["sha256"])
            with tarfile.open(path) as tar:
                tar.extractall(out, filter="data")  # nosec B202 - the "data" filter refuses absolute paths and outside links
        marker.write_text(archive["sha256"] + "\n", encoding="utf-8")
        done += 1
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=("listen", "full"), default="listen")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "models" / "samples")
    args = parser.parse_args()
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    base_url = f"https://github.com/{REPOSITORY}/releases/download/{release['tag']}"
    count = install(args.pack, args.out, base_url, release)
    print(f"{args.pack} pack {release['tag']}: {count} archive(s) unpacked in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
