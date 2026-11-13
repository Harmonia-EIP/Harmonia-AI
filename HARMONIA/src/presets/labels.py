"""Readable text for a preset from its human-written metadata (name, category, cartridge, comment).

DX7 names are 10-character abbreviations ("E.PNO 13.2", "PRC SYNTH2", "TOTO HMND1"); this expands the usual
synth-programmer shorthand so a text encoder can read them. Only the preset's own words are used.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

ABBREVIATIONS: Dict[str, str] = {
    "epno": "electric piano", "epiano": "electric piano", "ep": "electric piano", "pno": "piano",
    "apno": "acoustic piano", "gpno": "grand piano", "rhds": "rhodes", "wurli": "wurlitzer", "wur": "wurlitzer",
    "brs": "brass", "brss": "brass", "hrn": "horn", "hrns": "horns", "tpt": "trumpet", "trp": "trumpet",
    "tbn": "trombone", "sax": "saxophone", "strg": "strings", "strgs": "strings", "str": "strings",
    "stg": "strings", "strngs": "strings", "vln": "violin", "vla": "viola", "cel": "cello", "orch": "orchestra",
    "hmnd": "hammond organ", "hamm": "hammond organ", "hamnd": "hammond organ", "b3": "hammond organ",
    "org": "organ", "orgn": "organ", "syn": "synth", "synt": "synth", "ld": "lead", "bs": "bass",
    "sbass": "synth bass", "gtr": "guitar", "guit": "guitar", "egtr": "electric guitar", "harpsich": "harpsichord",
    "hpsch": "harpsichord", "hpscd": "harpsichord", "cembalo": "harpsichord", "clav": "clavinet",
    "vib": "vibraphone", "vibes": "vibraphone", "mrmba": "marimba", "xylo": "xylophone", "tub": "tubular",
    "glock": "glockenspiel", "vox": "voice", "chr": "choir", "flt": "flute", "clar": "clarinet",
    "clari": "clarinet", "obo": "oboe", "bsn": "bassoon", "perc": "percussion", "prc": "percussion",
    "fx": "effect", "sfx": "sound effect", "anlg": "analog", "ana": "analog", "pd": "pad", "swp": "sweep",
    "dr": "drum", "drm": "drum", "snr": "snare", "bd": "bass drum", "hh": "hi-hat", "cym": "cymbal",
    "acc": "accordion", "harm": "harmonica", "kalimba": "kalimba", "dist": "distorted", "elec": "electric",
    "ens": "ensemble", "clp": "clap", "orgn": "organ", "pipeorg": "pipe organ", "bel": "bell", "bls": "bells",
    "eorgan": "electric organ", "eorg": "electric organ", "eguitar": "electric guitar", "ebass": "electric bass",
}
INSTRUMENT_WORDS = set(" ".join(ABBREVIATIONS.values()).split()) | {
    "piano", "keys", "keyboard", "bass", "brass", "string", "strings", "organ", "synth", "lead", "pad", "pads",
    "bell", "bells", "flute", "guitar", "choir", "voice", "voices", "drum", "drums", "percussion", "harp",
    "marimba", "horn", "horns", "trumpet", "orchestra", "effect", "effects", "sweep", "analog", "wind", "winds",
    "mallet", "mallets", "pluck", "plucks", "sequence", "atmosphere", "ambient", "soundscape", "clav", "organs",
    "brass", "reed", "reeds", "sitar", "koto", "harmonica", "accordion", "clarinet", "oboe", "saxophone",
}
_TOKEN = re.compile(r"[A-Za-z]+|\d+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def expand(text: str) -> str:
    """'TOTO HMND1' -> 'toto hammond organ'; numbers (program numbers) are dropped."""
    text = re.sub(r"\b([Ee])\.(?=[A-Za-z])", r"\1", text)  # E.PNO / E.ORGAN -> EPNO / EORGAN
    text = _CAMEL.sub(" ", text.replace(".", " "))
    words: List[str] = []
    for token in _TOKEN.findall(text):
        if token.isdigit():
            continue
        lower = token.lower()
        words.append(ABBREVIATIONS.get(lower, lower))
    return " ".join(words)


def preset_text(entry: Dict[str, object]) -> str:
    """One line describing a preset with its own metadata only."""
    parts: List[str] = []
    names: Iterable[str] = [str(entry.get("name", "")), *entry.get("aliases", [])[:4]]
    seen = set()
    for name in names:
        readable = expand(name)
        if readable and readable not in seen:
            seen.add(readable)
            parts.append(readable)
    for key in ("category", "cartridge"):
        value = expand(str(entry.get(key) or "").replace("\\", " ").replace("/", " ").replace("_", " "))
        if key == "cartridge" and not set(value.split()) & INSTRUMENT_WORDS:
            continue  # cartridge file names are mostly codes ("TX7-17C"); keep only descriptive ones
        if value and value not in seen:
            seen.add(value)
            parts.append(value)
    comment = str(entry.get("comment") or "").strip()
    if comment:
        parts.append(comment)
    return ". ".join(parts)
