#!/usr/bin/env python3
"""Build the catalogue of sampled instruments from the downloaded libraries (#41).

Writes:
- <root>/catalog.json: every instrument with its regions (file, keys, velocities, round robin), which the
  sample packs carry for the plugin and the AI's run time;
- samples/instruments.json (in git, a few kB per instrument): the list the charter's instrument choice and the
  plugin's menu are built from, the credits of each library, and the folders left out with the reason.

    python scripts/samples/fetch_samples.py
    python scripts/samples/build_catalog.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.samples.fetch_samples import ROOT, SOURCES  # noqa: E402
from src.samples import catalog  # noqa: E402

SUMMARY = BASE_DIR / "samples" / "instruments.json"
# Libraries read from an SFZ map instead of their file names: (file, instrument name, family)
SFZ_MAPS = {"salamander": [("SalamanderGrandPiano-V3+20200602.sfz", "Salamander Grand Piano", "keys")]}


def build(root: Path):
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    instruments, left_out = [], []
    for library in sources:
        folder = root / library
        if not folder.exists():
            raise SystemExit(f"{folder} is missing: run scripts/samples/fetch_samples.py first")
        if library in SFZ_MAPS:
            for name, title, family in SFZ_MAPS[library]:
                instruments.append(catalog.sfz_instrument(library, folder, folder / name, title, family))
            continue
        groups = defaultdict(list)
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in catalog.AUDIO and path.is_file():
                groups[path.parent].append(path)
        for group, files in sorted(groups.items()):
            instrument, why = catalog.folder_instrument(library, folder, group, files)
            if instrument is None:
                left_out.append({"folder": f"{library}/{group.relative_to(folder).as_posix()}", "files": len(files),
                                 "why": why})
            else:
                instruments.append(instrument)
            print(f"  {library}/{group.relative_to(folder).as_posix()}: {'kept' if instrument else why}", flush=True)
    ids = [i["id"] for i in instruments]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(f"duplicate instrument ids: {sorted(duplicates)}")
    return sources, instruments, left_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    sources, instruments, left_out = build(args.root)
    (args.root / "catalog.json").write_text(json.dumps({"version": 1, "instruments": instruments}, indent=0) + "\n",
                                            encoding="utf-8")
    credits = {name: {k: s[k] for k in ("name", "author", "license", "license_url", "homepage")}
               for name, s in sources.items()}
    summary = {"version": 1, "libraries": credits, "instruments": [catalog.summary(i) for i in instruments],
               "left_out": left_out}
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    regions = sum(len(i["regions"]) for i in instruments)
    print(json.dumps({"instruments": len(instruments), "regions": regions, "left_out": len(left_out),
                      "by_family": {f: sum(1 for i in instruments if i["family"] == f)
                                    for f in sorted({i["family"] for i in instruments})}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
