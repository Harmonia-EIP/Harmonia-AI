#!/usr/bin/env python3
"""Gather the trained v3 models into models/harmonia_v3/ with a manifest (sha256 of every file), ready for
scripts/v2/package_release.py and scripts/v2/fetch_model.py --model harmonia_v3.

Only trained weights go in: no preset of the banks (DX7 cartridges, Surge, OB-Xf) and no audio. The bank
and the curated library are rebuilt locally from their sources (scripts/v3/fetch_presets.py ...).

The archive also carries CLAP's audio tower (laion/larger_clap_general, Apache-2.0, half precision), the ear
of the run time (src/v3/listener.py): the server then starts without reaching the Hugging Face hub.

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


def export_listener(out: Path) -> dict:
    """CLAP's audio tower and its feature-extractor settings, in half precision (the run time reads it back
    in single precision; cosine to the full-precision embeddings above 0.9999)."""
    from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor

    from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION
    from src.v3.listener import AUDIO_DIR

    target = out / AUDIO_DIR
    model = ClapAudioModelWithProjection.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)  # nosec B615
    model.half().save_pretrained(target)
    ClapFeatureExtractor.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION).save_pretrained(target)  # nosec B615
    files = {f"{AUDIO_DIR}/{f.name}": {"sha256": sha256(f), "bytes": f.stat().st_size}
             for f in sorted(target.iterdir()) if f.is_file()}
    return {"id": TEACHER_ID, "revision": TEACHER_REVISION, "license": "Apache-2.0", "files": files}


NOTICE = """# Harmonia model {version}

Trained weights of the Harmonia v3 text-to-preset model. Copyright (c) 2026 Harmonia-EIP
(https://github.com/Harmonia-EIP), PolyForm Noncommercial License 1.0.0 (see LICENSE).

Third-party parts in this archive:
- text_student/encoder: fine-tuned from sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (Apache-2.0).
- clap_audio: the audio tower of laion/larger_clap_general (Apache-2.0), unchanged, in half precision.

No preset of the banks the generators learnt from (DX7 cartridges, Surge XT, OB-Xf) and no audio are in
this archive. {recordings}
"""
RECORDINGS_NOTE = ("The analog generator also learnt from synthesizer settings that imitate {count} FSD50K recordings "
                   "(Freesound, CC0 and CC BY only); the recordings themselves are not included. Their titles, "
                   "authors and licences are listed in ATTRIBUTION.json.")


def export_notices(out: Path, version: str, recordings: Path | None, min_cos: float, max_distance: float) -> dict:
    """LICENSE, NOTICE.md and, when imitations of recordings were used, the credit of each recording."""
    note = ""
    if recordings is not None:
        rows = [json.loads(line) for line in open(recordings, encoding="utf-8")]
        rows = [r for r in rows if r["cos"] >= min_cos and r["distance"] <= max_distance]
        credits = [{"freesound_id": r["fname"], "title": r["title"], "author": r["uploader"], "license": r["license"]}
                   for r in sorted(rows, key=lambda r: int(r["fname"]))]
        (out / "ATTRIBUTION.json").write_text(json.dumps(credits, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        note = RECORDINGS_NOTE.format(count=len(credits))
    (out / "NOTICE.md").write_text(NOTICE.format(version=version, recordings=note), encoding="utf-8")
    shutil.copy2(BASE_DIR.parent / "LICENSE", out / "LICENSE")
    return {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
            for name in ("LICENSE", "NOTICE.md", "ATTRIBUTION.json") if (out / name).exists()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "models" / "harmonia_v3")
    parser.add_argument("--version", default="harmonia_v3", help="model_version written in the manifest")
    parser.add_argument("--display-name", default="Harmonia V3", help="name shown in the plugin's model menu")
    parser.add_argument("--generator", type=Path, default=DATA / "generator",
                        help="folder of the generators to ship (prior.pt, dx7.pt, analog.pt, meta.json)")
    parser.add_argument("--no-listener", action="store_true", help="leave CLAP's audio tower out (read from the hub)")
    parser.add_argument("--runtime-only", action="store_true",
                        help="only what the server loads: no v3.1 synth space and generators (research scripts)")
    parser.add_argument("--recordings", type=Path, default=None,
                        help="the imitations the analog generator was trained with (match_recordings.py): credits them")
    parser.add_argument("--min-cos", type=float, default=0.45)
    parser.add_argument("--max-distance", type=float, default=30.0)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    files = {}
    for target, source in FILES.items():
        if args.runtime_only and target.split("/")[0] in ("synth_space", "generator_synth"):
            continue
        dest = args.out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        origin = args.generator / Path(source).name if source.startswith("generator/") else DATA / source
        shutil.copy2(origin, dest)
        files[target] = {"sha256": sha256(dest), "bytes": dest.stat().st_size}
    manifest = {"model_version": args.version, "display_name": args.display_name, "engine": "src/synth/engine_v3.py (46 parameters) + DX7 mode (msfa)",
                "text_encoder": json.loads((DATA / "text_student" / "meta.json").read_text())["base_model"],
                "files": files}
    files.update(export_notices(args.out, args.version, args.recordings, args.min_cos, args.max_distance))
    if not args.no_listener:
        listener = export_listener(args.out)
        files.update(listener.pop("files"))
        manifest["listener"] = listener
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in files.values())
    print(f"{len(files)} files, {total / 1e6:.0f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
