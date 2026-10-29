#!/usr/bin/env python3
"""English sound-description corpus for distilling the multilingual text encoder.

Sources: FSD50K titles, tags and labels (dev split), the concept prompts and modifier words of
synthesize_dataset.py, and compositional templates ("<adjective> <source> <space>").

    python scripts/v2/build_text_corpus.py --fsd50k ~/Datasets/FSD50K --out data/v2/corpus_en.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.synthesize_dataset import CONCEPTS, MODIFIERS  # noqa: E402

SPACES = [
    "in a cathedral", "in a church", "in a small room", "in a bathroom", "in a large hall", "in a cave",
    "in a tunnel", "in a stairwell", "in a stadium", "outdoors", "in a forest", "underwater",
    "in an empty warehouse", "in a parking garage", "far away", "up close", "through a wall",
    "on an old radio", "in a dry studio", "in a concert hall", "in a metal tank", "in a canyon",
]
SFX_SOURCES = [
    "a door slamming", "a door creaking", "a heavy door closing", "footsteps", "footsteps on gravel",
    "glass breaking", "a glass shattering", "a metal pipe being hit", "a wooden knock", "knocking on a door",
    "a gunshot", "an explosion", "thunder", "rain", "wind blowing", "a heartbeat", "a bell ringing",
    "a church bell", "a clock ticking", "a car horn", "a siren", "a laser zap", "a spaceship engine",
    "a robot voice", "a whoosh", "a riser", "a bass drop", "a kick drum", "a snare drum", "a hand clap",
    "a cymbal crash", "a gong", "water dripping", "a stone falling", "a chain rattling", "a sword clash",
    "an alarm", "a phone ringing", "a typewriter", "a camera shutter", "a punch", "a whip crack",
    "a balloon popping", "sparks", "electric buzz", "a drone hum", "a steam hiss", "a coin dropping",
    "a dog barking", "a bird chirping", "ocean waves", "a crackling fire", "a train passing",
]
ADJECTIVES = sorted(set(MODIFIERS) | {
    "distant", "loud", "quiet", "muffled", "wooden", "huge", "tiny", "short", "long", "sharp",
    "smooth", "harsh", "airy", "breathy", "pulsing", "wobbly", "glitchy", "eerie", "creepy", "happy",
    "sad", "massive", "gritty", "crisp", "hollow", "resonant", "noisy", "pure", "buzzy", "plucky",
})
TEMPLATES = [
    "{src}", "{src} {space}", "{adj} {src}", "{adj} {src} {space}", "the sound of {src}",
    "the sound of {src} {space}", "{src} with a long reverb tail", "{src}, {adj} and {adj2}",
]
TAG_STOPWORDS = {
    "field-recording", "fieldrecording", "stereo", "mono", "wav", "mp3", "flac", "aif", "aiff", "zoom",
    "h4n", "h2n", "h1", "recording", "sound", "sounds", "sfx", "effect", "effects", "fx", "audio",
    "multisample", "single-note", "freesound", "binaural", "ambisonic", "48khz", "44khz",
}


def clean_title(title: str) -> str:
    title = re.sub(r"\.(wav|mp3|flac|aiff?|ogg|m4a)\b", " ", title, flags=re.I)
    title = re.sub(r"[_\-+()\[\]{}#~=]+", " ", title)
    title = re.sub(r"\b\w*\d\w*\b", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" .,'\"")
    return title


def label_phrase(label: str) -> str:
    return label.replace("_and_", " and ").replace("_", " ").replace("(", "").replace(")", "").lower()


def tag_phrase(tags, rng: random.Random) -> str:
    useful = [t for t in tags if not re.search(r"\d", t) and t.lower() not in TAG_STOPWORDS and len(t) > 2]
    if len(useful) < 2:
        return ""
    picked = rng.sample(useful, k=min(len(useful), rng.randint(2, 4)))
    return " ".join(t.replace("-", " ") for t in picked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "corpus_en.jsonl")
    parser.add_argument("--templates", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)  # nosec B311 - seeded RNG for reproducible data, not security-sensitive
    rows = {}

    def add(text: str, source: str) -> None:
        text = text.strip()
        n_words = len(text.split())
        if 1 <= n_words <= 24 and len(text) <= 160 and text.lower() not in rows:
            rows[text.lower()] = {"en": text, "source": source}

    info = json.loads((args.fsd50k / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    for meta in info.values():
        title = clean_title(meta.get("title", ""))
        if len(title.split()) >= 2:
            add(title, "fsd50k_title")
        phrase = tag_phrase(meta.get("tags", []), rng)
        if phrase:
            add(phrase, "fsd50k_tags")
    with open(args.fsd50k / "FSD50K.ground_truth" / "dev.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels = row["labels"].split(",")
            add(label_phrase(labels[0]), "fsd50k_label")
            if len(labels) > 1:
                add(f"{label_phrase(labels[0])}, {label_phrase(labels[1])}", "fsd50k_label")

    for concept in CONCEPTS:
        for prompt in concept["prompts"]:
            add(prompt, "concept")
            for _ in range(3):
                add(f"{rng.choice(ADJECTIVES)} {prompt.lower()}", "concept_modifier")

    instruments = [p.lower() for c in CONCEPTS for p in c["prompts"]]
    sources = SFX_SOURCES + instruments
    for _ in range(args.templates):
        template = rng.choice(TEMPLATES)
        adj, adj2 = rng.sample(ADJECTIVES, 2)
        src = rng.choice(sources)
        if "{adj} {src}" in template:
            src = re.sub(r"^(a|an) ", "", src)
        add(template.format(src=src, space=rng.choice(SPACES), adj=adj, adj2=adj2), "template")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows.values():
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(json.dumps({"total": len(rows), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
