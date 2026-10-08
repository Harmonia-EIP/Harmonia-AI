#!/usr/bin/env python3
"""Gather the trained v3 models into models/harmonia_v3/ with a manifest (sha256 of every file), ready for
scripts/v2/package_release.py and scripts/v2/fetch_model.py --model harmonia_v3.

Only trained weights go in: no preset of the banks (DX7 cartridges, Surge, OB-Xf) and no audio. The bank
and the curated library are rebuilt locally from their sources (scripts/v3/fetch_presets.py ...).

    python scripts/v3/export_model.py
    python scripts/v2/package_release.py --model-dir models/harmonia_v3 --tag ai-v3.0.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.fetch_model import sha256  # noqa: E402

DATA = BASE_DIR / "data" / "v3"
FILES = {
    # multilingual text encoder (FR/EN) distilled into CLAP's text space
    "text_student/encoder/config.json": "text_student/encoder/config.json",
    "text_student/encoder/model.safetensors": "text_student/encoder/model.safetensors",
    "text_student/encoder/tokenizer.json": "text_student/encoder/tokenizer.json",
    "text_student/encoder/tokenizer_config.json": "text_student/encoder/tokenizer_config.json",
    "text_student/proj.pt": "text_student/proj.pt",
    "text_student/meta.json": "text_student/meta.json",
    # v3.0: text -> CLAP sound prior and generators conditioned on CLAP sound
    "generator/prior.pt": "generator/prior.pt",
    "generator/dx7.pt": "generator/dx7.pt",
    "generator/analog.pt": "generator/analog.pt",
    "generator/meta.json": "generator/meta.json",
    # v3.1: synth space (types, text <-> sound) and generators conditioned on it
    "synth_space/synth_space.pt": "synth_space/synth_space.pt",
    "synth_space/report.json": "synth_space/report.json",
    "generator_synth/dx7.pt": "generator_synth/dx7.pt",
    "generator_synth/analog.pt": "generator_synth/analog.pt",
    "generator_synth/meta.json": "generator_synth/meta.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "models" / "harmonia_v3")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    files = {}
    for target, source in FILES.items():
        dest = args.out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA / source, dest)
        files[target] = {"sha256": sha256(dest), "bytes": dest.stat().st_size}
    manifest = {"model_version": "harmonia_v3", "engine": "src/synth/engine_v3.py (46 parameters) + DX7 mode (msfa)",
                "text_encoder": json.loads((DATA / "text_student" / "meta.json").read_text())["base_model"],
                "files": files}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in files.values())
    print(f"{len(files)} files, {total / 1e6:.0f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
