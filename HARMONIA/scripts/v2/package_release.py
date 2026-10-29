#!/usr/bin/env python3
"""Package an exported model directory as a release asset and pin it in models/release.json.

    python scripts/v2/package_release.py --model-dir models/harmonia_v2 --tag ai-v2.0.0
    gh release create ai-v2.0.0 dist/harmonia_v2-ai-v2.0.0.tar.gz   # then commit models/release.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.fetch_model import RELEASE_FILE, sha256, verify_model_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "models" / "harmonia_v2")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path, default=BASE_DIR / "dist")
    args = parser.parse_args()

    verify_model_dir(args.model_dir)
    manifest = json.loads((args.model_dir / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["model_version"]
    args.dist.mkdir(parents=True, exist_ok=True)
    asset = args.dist / f"{name}-{args.tag}.tar.gz"
    with tarfile.open(asset, "w:gz") as tar:
        for file in ["manifest.json", *manifest["files"]]:
            tar.add(args.model_dir / file, arcname=file)

    releases = json.loads(RELEASE_FILE.read_text(encoding="utf-8")) if RELEASE_FILE.exists() else {}
    releases[name] = {"tag": args.tag, "asset": asset.name, "sha256": sha256(asset), "bytes": asset.stat().st_size}
    RELEASE_FILE.write_text(json.dumps(releases, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(releases[name], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
