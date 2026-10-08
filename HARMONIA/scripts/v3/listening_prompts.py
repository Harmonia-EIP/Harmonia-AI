#!/usr/bin/env python3
"""Blind listening page for text -> preset: for each prompt, one sound per system (v3.0: generation and
selection from the bank; v3.1: generation, variation of real presets and selection), in a random order,
each rated 1-5 for "matches the description".

The page only carries opaque ids; which sound is which stays in <out>/key_<round>.json.

    python scripts/v3/generate_v3.py --prompts prompts.json --out data/v3/generations
    python scripts/v3/listening_prompts.py --generations data/v3/generations
    python scripts/v3/generate_v31.py --prompts prompts.json --out data/v3/generations_v31
    python scripts/v3/listening_prompts.py --generations data/v3/generations_v31 --round p2
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.listening_v3 import clip, write_mp3  # noqa: E402
from src.presets import dx7_render  # noqa: E402
from src.presets import matching as M  # noqa: E402
from src.synth.engine_v3 import render_physical  # noqa: E402

BANK = BASE_DIR / "data" / "v3" / "bank"
TEMPLATE = Path(__file__).with_name("listening_v3_template.html")
NOTE = ("Écoute à l'aveugle : pour chaque description, {count} sons joués sur trois notes (do2, do3, do4) : {what}, "
        "dans un ordre caché. Note chacun de 1 à 5 selon qu'il correspond à la description.")
WHAT = {2: "l'un créé par l'IA (génération), l'autre un vrai preset choisi dans la banque (sélection)",
        "v32": "de vrais presets choisis par l'IA et une légère variation créée par l'IA",
        3: "un créé par l'IA de zéro (génération), un créé par l'IA à partir de vrais presets (variation) et un vrai "
           "preset choisi dans la banque (sélection)"}


def notes_audio(kind: str, preset) -> list:
    if kind == "dx7":
        return [dx7_render.render(preset, note=n, velocity=M.VELOCITY, hold_seconds=M.HOLD_SECONDS,
                                  total_seconds=M.TOTAL_SECONDS) for n in M.NOTES]
    return [render_physical(np.asarray(preset, dtype=np.float64), n, M.VELOCITY / 127, M.HOLD_SECONDS,
                            M.TOTAL_SECONDS, M.SR, 7) for n in M.NOTES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=BASE_DIR / "data" / "v3" / "generations")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v3" / "listening_prompts")
    parser.add_argument("--round", default="p1")
    args = parser.parse_args()

    from src.v3 import codecs

    rows = json.loads((args.generations / "generations.json").read_text(encoding="utf-8"))
    z = np.load(BANK / "bank.npz")
    ids = [json.loads(line)["id"] for line in open(BANK / "entries.jsonl", encoding="utf-8")]
    index = {pid: i for i, pid in enumerate(ids)}
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"harmonia-{args.round}")  # nosec B311 - shuffles A/B order, not security-sensitive
    items, key = [], {}
    def decoded(kind, vector):
        vector = np.asarray(vector)
        return kind, codecs.dx7_decode(vector) if kind == "dx7" else codecs.analog_decode(vector)

    def bank_preset(pid):
        i = index[pid]
        if z["kind"][i] == "dx7":
            return "dx7", dx7_render.from_patch(bytes(z["dx7"][i]))
        return "analog", z["analog"][i].astype(np.float64)

    count = 2
    for n, row in enumerate(rows):
        if "generated_vector" in row:  # v3.0
            systems = [("generation", decoded(row["generated_kind"], row["generated_vector"])),
                       ("selection", bank_preset(row["selected"]["id"]))]
        elif "curated" in row:  # v3.2: one sound when the curated choice is v3.0's
            systems = [("curated", bank_preset(row["curated"]["id"])),
                       ("varied", decoded(row["varied"]["kind"], row["varied"]["vector"]))]
            if row["reference"]["id"] == row["curated"]["id"]:
                systems[0] = ("curated+reference", systems[0][1])
            else:
                systems.append(("reference", bank_preset(row["reference"]["id"])))
        else:  # v3.1
            systems = [("generation", decoded(row["generation"]["kind"], row["generation"]["vector"])),
                       ("variation", decoded(row["variation"]["kind"], row["variation"]["vector"])),
                       ("selection", bank_preset(row["selection"]["id"]))]
        count = max(count, len(systems))
        rng.shuffle(systems)
        for letter, (system, (kind, preset)) in zip("ABC", systems):
            opaque = hashlib.sha1(f"{args.round}-{n}-{letter}".encode(), usedforsecurity=False).hexdigest()[:12]
            path = f"audio/{opaque}.mp3"
            write_mp3(args.out / path, clip(notes_audio(kind, preset)))
            item_id = f"prompt-{args.round}-{n:03d}-{letter}"
            key[item_id] = {"system": system, "kind": kind, "prompt": row["prompt"]}
            items.append({"id": item_id, "source": "Prompts", "name": f"{letter} · {row['prompt']}", "category": "",
                          "author": "", "license": "", "clips": {"dx7": path}})
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
    page = page.replace("<title>Banc d'écoute Harmonia v3</title>", "<title>Prompts Harmonia v3</title>")
    page = page.replace("<h1>Banc d'écoute Harmonia v3</h1>", "<h1>Harmonia v3 · du texte au preset</h1>")
    if "curated" in rows[0]:
        note = NOTE.format(count="2 ou 3", what=WHAT["v32"])
    else:
        note = NOTE.format(count=count, what=WHAT[count])
    page = re.sub(r'<p class="lede">.*?</p>', f'<p class="lede">{html.escape(note)}</p>', page, count=1, flags=re.S)
    page = re.sub(r'\s*<div class="legend">.*?(?=\s*<div class="bar">)', "", page, count=1, flags=re.S)
    page = re.sub(r'\s*<button data-f="(OB-Xf|Surge XT|DX7)"[^\n]*', "", page)
    page = page.replace("Mode DX7 (Harmonia)", "Son").replace("Son réussi ?", "Correspond ?")
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / f"key_{args.round}.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} sounds for {len(rows)} prompts -> {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
