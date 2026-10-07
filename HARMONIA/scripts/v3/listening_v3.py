#!/usr/bin/env python3
"""Build the v3 listening page: real presets next to their Harmonia version, plus the DX7 mode.

For each matched preset that kept its audio (scripts/v3/match_presets.py), three clips: the original synth,
the direct parameter conversion and the matched Harmonia preset. For the DX7 mode, voices whose names
refer to famous sounds, played by the msfa core. Each clip chains C3, C4 and C5 (2.4 s each) at one
loudness. Writes data/v3/listening/{index.html, items.json, audio/*.mp3}.

    python scripts/v3/listening_v3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets import dx7_render  # noqa: E402
from src.presets import matching as M  # noqa: E402

MATCHED_DIR = BASE_DIR / "data" / "v3" / "matched"
PRESET_DIR = BASE_DIR / "data" / "v3" / "presets"
OUT_DIR = BASE_DIR / "data" / "v3" / "listening"
TEMPLATE = Path(__file__).with_name("listening_v3_template.html")
NOTE_SECONDS = 2.4
FADE_SECONDS = 0.05
TARGET_DBFS = -20.0
SOURCE_LABELS = {"obxf": "OB-Xf", "surge": "Surge XT"}
DX7_FAMOUS = ["SUPERTRAMP", "WURLITZER", "E.PIANO 1", "TOTO HMND1", "HAMMOND2", "AFRICA  1", "JUMP", "Axel F",
              "BILLIEJEAN", "BLADERUNNR", "VANGELIS 2", "Zawinul.1", "STEVIE COL", "QueenBel1", "BRASS   1",
              "STRINGS 1", "MINIMOOG", "TUB BELLS", "MARIMBA", "HARPSICH 1", "FLUTE   1", "MELLOTRON"]


def clip(notes_audio) -> np.ndarray:
    n = int(NOTE_SECONDS * M.SR)
    fade = np.linspace(1.0, 0.0, int(FADE_SECONDS * M.SR))
    parts = []
    for audio in notes_audio:
        part = np.asarray(audio[:n], dtype=np.float64).copy()
        part[-fade.size:] *= fade
        parts.append(part)
    out = np.nan_to_num(np.concatenate(parts))
    rms = np.sqrt(np.mean(out ** 2))
    if rms > 1e-7:
        out *= 10 ** (TARGET_DBFS / 20) / rms
    return np.clip(out, -0.99, 0.99).astype(np.float32)


def write_mp3(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio, M.SR, format="MP3")


def matched_items(source: str, audio_dir: Path, limit: int):
    rows = [json.loads(line) for line in open(MATCHED_DIR / f"{source}.jsonl", encoding="utf-8")]
    meta = {json.loads(line)["id"]: json.loads(line) for line in open(PRESET_DIR / f"{source}.jsonl", encoding="utf-8")}
    rows = [r for r in rows if r.get("status") == "ok" and r.get("audio")]
    by_category = {}
    for r in sorted(rows, key=lambda r: r["name"].lower()):
        by_category.setdefault(r["category"], []).append(r)
    picked = []  # round-robin over categories so every kind of sound is heard
    while len(picked) < min(limit, len(rows)):
        for group in by_category.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    picked.sort(key=lambda r: (r["category"].lower(), r["name"].lower()))
    for r in picked:
        stem = r["id"].replace(":", "_")
        data = np.load(MATCHED_DIR / f"{source}_audio" / f"{stem}.npz")
        clips = {}
        for version in ("original", "converted", "harmonia"):
            name = f"{stem}_{version}.mp3"
            write_mp3(audio_dir / name, clip(data[version]))
            clips[version] = f"audio/{name}"
        rec = meta[r["id"]]
        yield {"id": r["id"].replace(":", "-"), "source": SOURCE_LABELS[source], "name": r["name"],
               "category": r["category"], "author": rec.get("author", ""), "license": rec.get("license", ""),
               "start_distance": r["start_distance"], "distance": r["distance"], "clips": clips}


def dx7_origin(paths) -> str:
    """Where a DX7 voice comes from: Yamaha's own cartridges first, else the first collection."""
    for path in paths:
        if "Original Yamaha" in path:
            return "Yamaha, cartouche d'origine (" + Path(path.split("#")[0]).stem + ")"
    parts = Path(paths[0].split("#")[0]).parts
    if parts[0] == "dexed_builtin":
        return "Dexed, programmes intégrés"
    return "collection " + (parts[2] if len(parts) > 3 else parts[-1]).lstrip("!")


def dx7_items(audio_dir: Path):
    records = [json.loads(line) for line in open(PRESET_DIR / "dx7.jsonl", encoding="utf-8")]
    by_name = {}
    for rec in records:
        for name in [rec["name"], *rec["aliases"]]:
            by_name.setdefault(name.strip(), []).append(rec)
    for wanted in DX7_FAMOUS:
        candidates = by_name.get(wanted.strip(), [])
        if not candidates:
            continue
        # prefer Yamaha's own cartridges, then the most copied version
        rec = max(candidates, key=lambda r: (any("Original Yamaha" in p for p in r["paths"]), len(r["paths"])))
        notes = [dx7_render.render(rec["params"], note=n, velocity=M.VELOCITY, hold_seconds=M.HOLD_SECONDS,
                                   total_seconds=M.TOTAL_SECONDS) for n in M.NOTES]
        stem = rec["id"].replace(":", "_")
        write_mp3(audio_dir / f"{stem}.mp3", clip(notes))
        yield {"id": rec["id"].replace(":", "-"), "source": "DX7", "name": wanted.strip(), "category": "",
               "author": dx7_origin(rec["paths"]), "license": "", "copies": len(rec["paths"]),
               "clips": {"dx7": f"audio/{stem}.mp3"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=["obxf", "surge"])
    parser.add_argument("--per-source", type=int, default=30)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for source in args.sources:
        if (MATCHED_DIR / f"{source}.jsonl").exists():
            items += list(matched_items(source, audio_dir, args.per_source))
    items += list(dx7_items(audio_dir))
    (args.out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} items -> {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
