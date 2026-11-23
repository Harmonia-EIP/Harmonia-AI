#!/usr/bin/env python3
"""Sample packs for the plugin and for the AI's server, from the catalogue (#41).

- full: every recording the catalogue uses, losslessly in FLAC (same rate and depth as recorded), for the
  plugin. Split in archives under 1.9 GB (GitHub release assets stop at 2 GB).
- listen: what the server needs to hear a preset while it generates (src/v3/inference.py plays every
  candidate): per instrument and recorded note, one middle-velocity take, mono, LISTEN_RATE, the first
  LISTEN_SECONDS. Small enough for a small VPS; the plugin never uses it.

Each archive carries catalog.json (which file answers which key and velocity) and CREDITS.md. The archives,
their sha256 and size are written to samples/release.json, like the models (models/release.json); nothing is
uploaded by this script.

    python scripts/samples/build_packs.py --tag samples-1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.samples.fetch_samples import ROOT, SOURCES  # noqa: E402

RELEASE = BASE_DIR / "samples" / "release.json"
LISTEN_RATE, LISTEN_SECONDS, FADE_SECONDS = 32000, 4.0, 0.05
MAX_ARCHIVE = 1_900_000_000
DEPTHS = {"PCM_16": "PCM_16", "PCM_S8": "PCM_16", "PCM_U8": "PCM_16"}  # anything else becomes 24-bit

CREDITS = """# Harmonia sample packs

The recordings in these packs come from free libraries, unchanged apart from the file format (FLAC) and, for
the "listen" pack, a shorter mono copy. Harmonia does not sell these samples.

{libraries}
"""


def flac_name(path: str) -> str:
    return str(Path(path).with_suffix(".flac").as_posix())


def _convert(task) -> int:
    import soundfile as sf

    source, target, listen = task
    target = Path(target)
    if target.exists():
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(source, always_2d=True, dtype="float32")
    subtype = DEPTHS.get(sf.info(source).subtype, "PCM_24")
    if listen:
        import soxr

        mono = audio.mean(axis=1)
        mono = mono[:int(LISTEN_SECONDS * rate)]
        mono = soxr.resample(mono, rate, LISTEN_RATE).astype(np.float32) if rate != LISTEN_RATE else mono
        fade = min(len(mono), int(FADE_SECONDS * LISTEN_RATE))
        if fade:
            mono[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        audio, rate, subtype = mono[:, None], LISTEN_RATE, "PCM_16"
    tmp = target.with_name(target.name + ".part")
    sf.write(str(tmp), audio, rate, format="FLAC", subtype=subtype)
    os.replace(tmp, target)
    return target.stat().st_size


def listen_regions(instrument: dict) -> list:
    """One take per recorded note: the middle velocity layer, first round robin, answering every velocity."""
    attack = [r for r in instrument["regions"] if r["trigger"] == "attack"]
    chosen = []
    for key in sorted({r["key"] for r in attack}):
        takes = sorted((r for r in attack if r["key"] == key), key=lambda r: (r["lovel"], r["round_robin"]))
        layers = sorted({r["lovel"] for r in takes})
        middle = layers[len(layers) // 2]
        take = next(r for r in takes if r["lovel"] == middle)
        chosen.append({**take, "lovel": 1, "hivel": 127, "round_robin": 1})
    return chosen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archives(name: str, tag: str, folder: Path, files: list, dist: Path, extra: dict) -> list:
    """Uncompressed tar archives (FLAC is already compressed) of at most MAX_ARCHIVE bytes each."""
    parts, current, size = [], [], 0
    for rel in files:
        n = (folder / rel).stat().st_size
        if current and size + n > MAX_ARCHIVE:
            parts.append(current)
            current, size = [], 0
        current.append(rel)
        size += n
    if current:
        parts.append(current)
    written = []
    for i, part in enumerate(parts, 1):
        suffix = f"-part{i}" if len(parts) > 1 else ""
        path = dist / f"harmonia-samples-{name}-{tag}{suffix}.tar"
        with tarfile.open(path, "w") as tar:
            for meta_name, meta_path in extra.items():
                tar.add(meta_path, arcname=meta_name)
            for rel in part:
                tar.add(folder / rel, arcname=rel)
        written.append({"asset": path.name, "sha256": sha256(path), "bytes": path.stat().st_size, "files": len(part)})
        print(f"  {path.name}: {len(part)} files, {path.stat().st_size / 1e9:.2f} GB", flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", required=True, help="release tag, e.g. samples-1.0.0")
    parser.add_argument("--dist", type=Path, default=BASE_DIR / "dist" / "samples")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()

    catalog = json.loads((args.root / "catalog.json").read_text(encoding="utf-8"))
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    out = args.root / "packs"
    args.dist.mkdir(parents=True, exist_ok=True)
    credits = "\n".join(f"- **{s['name']}** by {s['author']}, {s['license']} ({s['license_url']}), {s['homepage']}"
                        for s in sources.values())
    (out / "CREDITS.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "CREDITS.md").write_text(CREDITS.format(libraries=credits), encoding="utf-8")

    release = {"tag": args.tag, "packs": {}}
    for kind in ("full", "listen"):
        listen = kind == "listen"
        instruments, tasks, files = [], [], []
        for instrument in catalog["instruments"]:
            regions = listen_regions(instrument) if listen else instrument["regions"]
            moved = []
            for region in regions:
                rel = f"{instrument['library']}/{flac_name(region['file'])}"
                source = args.root / instrument["library"] / region["file"]
                tasks.append((str(source), str(out / kind / rel), listen))
                files.append(rel)
                moved.append({**region, "file": rel, **({"sample_rate": LISTEN_RATE, "channels": 1} if listen else {})})
            instruments.append({**instrument, "regions": moved})
        files = list(dict.fromkeys(files))
        tasks = list({t[1]: t for t in tasks}.values())
        with ProcessPoolExecutor(args.workers) as pool:
            total = sum(pool.map(_convert, tasks, chunksize=16))
        if listen:  # frames of the shortened copies
            import soundfile as sf

            for instrument in instruments:
                for region in instrument["regions"]:
                    region["frames"] = int(sf.info(str(out / kind / region["file"])).frames)
        catalog_path = out / f"catalog-{kind}.json"
        catalog_path.write_text(json.dumps({"version": catalog["version"], "pack": kind, "instruments": instruments},
                                           indent=0) + "\n", encoding="utf-8")
        print(f"{kind}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
        extra = {"catalog.json": catalog_path, "CREDITS.md": out / "CREDITS.md"}
        if listen:
            release["packs"][kind] = archives(kind, args.tag, out / kind, files, args.dist, extra)
        else:
            by_library = {}
            for rel in files:
                by_library.setdefault(rel.split("/", 1)[0], []).append(rel)
            release["packs"][kind] = [a for library, rels in by_library.items()
                                      for a in archives(f"{kind}-{library}", args.tag, out / kind, rels, args.dist, extra)]
    RELEASE.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({kind: sum(a["bytes"] for a in packs) for kind, packs in release["packs"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
