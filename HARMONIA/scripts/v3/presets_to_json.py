#!/usr/bin/env python3
"""Convert the downloaded preset banks (scripts/v3/fetch_presets.py) to JSON lines, one preset per line.

    python scripts/v3/presets_to_json.py            # -> data/v3/presets/{dx7,obxf,surge}.jsonl

Each line: id, source, name, aliases (other names of the same sound), category, author, license, comment,
paths, params (the synth's own parameters, untouched) and source-specific extras. Identical sounds are merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.fetch_presets import DEFAULT_OUT  # noqa: E402
from src.presets import dx7, fxp  # noqa: E402

OUT_DIR = BASE_DIR / "data" / "v3" / "presets"


def digest(payload: object) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:16]


class Merger:
    """Merge presets whose sound parameters are identical, keeping every name and path."""

    def __init__(self, source: str):
        self.source = source
        self.items: dict = {}

    def add(self, sound: dict, record: dict) -> None:
        key = digest(sound)
        if key not in self.items:
            self.items[key] = {"id": f"{self.source}:{key}", "source": self.source, **record, "aliases": [],
                               "paths": []}
        item = self.items[key]
        if record["name"] and record["name"] != item["name"] and record["name"] not in item["aliases"]:
            item["aliases"].append(record["name"])
        item["paths"].append(record.pop("path"))
        item.pop("path", None)

    def write(self, path: Path) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(self.items)


def convert_dx7(root: Path, merger: Merger, stats: Counter) -> None:
    files = sorted(p for d in ("dexed_builtin", "dexed_carts") for p in (root / d).rglob("*") if p.suffix.lower() == ".syx")
    for path in files:
        rel = path.relative_to(root)
        collection = rel.parts[1] if rel.parts[0] == "dexed_carts" and len(rel.parts) > 2 else rel.parts[0]
        try:
            voices = list(dx7.read_file(path))
        except OSError:
            stats["dx7_unreadable_files"] += 1
            continue
        stats["dx7_files"] += 1
        for voice in voices:
            name = voice.pop("name")
            slot = voice.pop("slot")
            stats["dx7_voices"] += 1
            if name.upper().startswith("INIT VOICE") or not name:
                stats["dx7_init_or_unnamed"] += 1
                continue
            merger.add(voice, {"name": name, "category": "", "author": collection, "license": "",
                               "comment": "", "cartridge": path.stem, "params": voice,
                               "path": f"{rel}#{slot + 1}"})


def convert_obxf(root: Path, merger: Merger, stats: Counter) -> None:
    for path in sorted(root.rglob("*.fxp")):
        rel = path.relative_to(root)
        try:
            patch = fxp.read_obxf(path)
        except (ValueError, OSError) as exc:
            stats["obxf_errors"] += 1
            print(f"skip {rel}: {exc}")
            continue
        meta = patch["meta"]
        merger.add(patch["params"], {"name": meta.get("programName") or path.stem,
                                     "category": meta.get("category") or path.parent.name,
                                     "author": meta.get("author", ""), "license": meta.get("license", ""),
                                     "comment": "", "params": patch["params"], "path": str(rel)})


def convert_surge(root: Path, merger: Merger, stats: Counter) -> None:
    for path in sorted(root.rglob("*.fxp")):
        rel = path.relative_to(root)
        try:
            patch = fxp.read_surge(path)
        except (ValueError, OSError, SyntaxError) as exc:
            stats["surge_errors"] += 1
            print(f"skip {rel}: {exc}")
            continue
        meta = patch["meta"]
        parts = rel.parts
        third_party = "patches_3rdparty" in parts
        folder_author = parts[parts.index("patches_3rdparty") + 1] if third_party else ""
        merger.add({"p": patch["params"], "m": patch["modulation"]}, {
            "name": meta.get("name") or path.stem,
            "category": meta.get("category") or path.parent.name,
            "author": meta.get("author") or folder_author, "license": meta.get("license", ""),
            "comment": meta.get("comment", ""), "bank": "3rdparty" if third_party else "factory",
            "params": patch["params"], "modulation": patch["modulation"], "sections": patch["sections"],
            "wavetable_bytes": patch["wavetable_bytes"], "path": str(rel)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    jobs = (("dx7", convert_dx7, args.root), ("obxf", convert_obxf, args.root / "obxf"),
            ("surge", convert_surge, args.root / "surge"))
    for name, convert, root in jobs:
        merger = Merger(name)
        convert(root, merger, stats)
        count = merger.write(args.out / f"{name}.jsonl")
        stats[f"{name}_unique"] = count
        print(f"{name}: {count} unique presets -> {args.out / (name + '.jsonl')}")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
