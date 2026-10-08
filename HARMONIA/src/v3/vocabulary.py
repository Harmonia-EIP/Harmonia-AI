"""Synth vocabulary from human labels: the type of a sound (bass, pad, lead...) read from the words preset
designers and FSD50K annotators used (preset categories and names, FSD50K class labels).

The table below only groups synonyms of words that appear in those labels; it adds no description of its own.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np

from src.presets.labels import expand

TYPES: List[str] = ["bass", "pad", "lead", "pluck", "keys", "organ", "bell", "brass", "strings", "winds", "voice",
                    "guitar", "drums", "fx"]
# words (after src/presets/labels.expand) -> type; multi-word keys are matched first
TYPE_WORDS: Dict[str, str] = {
    "bass drum": "drums", "kick drum": "drums", "steel drum": "bell", "pan flute": "winds",
    "electric piano": "keys", "grand piano": "keys", "pipe organ": "organ", "hammond organ": "organ",
    "bass": "bass", "basses": "bass", "sub": "bass", "subbass": "bass",
    "pad": "pad", "pads": "pad", "ambient": "pad", "ambience": "pad", "atmosphere": "pad", "atmospheres": "pad",
    "soundscape": "pad", "soundscapes": "pad", "drone": "pad", "drones": "pad",
    "lead": "lead", "leads": "lead",
    "pluck": "pluck", "plucks": "pluck", "plucked": "pluck",
    "keys": "keys", "key": "keys", "piano": "keys", "pianos": "keys", "rhodes": "keys", "wurlitzer": "keys",
    "clavinet": "keys", "clav": "keys", "harpsichord": "keys", "keyboard": "keys", "keyboards": "keys",
    "organ": "organ", "organs": "organ",
    "bell": "bell", "bells": "bell", "mallet": "bell", "mallets": "bell", "vibraphone": "bell", "marimba": "bell",
    "xylophone": "bell", "glockenspiel": "bell", "kalimba": "bell", "celesta": "bell", "chime": "bell",
    "chimes": "bell", "tubular": "bell", "gong": "bell",
    "brass": "brass", "horn": "brass", "horns": "brass", "trumpet": "brass", "trumpets": "brass",
    "trombone": "brass", "tuba": "brass",
    "strings": "strings", "string": "strings", "violin": "strings", "viola": "strings", "cello": "strings",
    "orchestra": "strings", "orchestral": "strings", "pizzicato": "strings", "fiddle": "strings",
    "flute": "winds", "clarinet": "winds", "oboe": "winds", "bassoon": "winds", "saxophone": "winds",
    "reed": "winds", "reeds": "winds", "wind": "winds", "winds": "winds", "harmonica": "winds",
    "accordion": "winds", "whistle": "winds", "recorder": "winds",
    "choir": "voice", "voice": "voice", "voices": "voice", "vocal": "voice", "vocals": "voice", "singing": "voice",
    "guitar": "guitar", "guitars": "guitar", "sitar": "guitar", "koto": "guitar", "harp": "guitar",
    "banjo": "guitar", "mandolin": "guitar",
    "drum": "drums", "drums": "drums", "percussion": "drums", "snare": "drums", "kick": "drums", "hihat": "drums",
    "cymbal": "drums", "clap": "drums", "tom": "drums", "toms": "drums", "conga": "drums", "bongo": "drums",
    "timpani": "drums", "cowbell": "drums", "tabla": "drums",
    "effect": "fx", "effects": "fx", "fx": "fx", "sweep": "fx", "sweeps": "fx", "riser": "fx", "noise": "fx",
}
# FSD50K class names (first match wins, most specific first)
FSD_TYPES: Dict[str, str] = {
    "Bass_drum": "drums", "Drum": "drums", "Percussion": "drums", "Cymbal": "drums", "Snare_drum": "drums",
    "Hi-hat": "drums", "Tabla": "drums", "Tambourine": "drums", "Cowbell": "drums",
    "Bass_guitar": "bass", "Piano": "keys", "Keyboard_(musical)": "keys", "Electric_piano": "keys",
    "Harpsichord": "keys", "Organ": "organ", "Bell": "bell", "Glockenspiel": "bell", "Marimba_and_xylophone": "bell",
    "Mallet_percussion": "bell", "Chime": "bell", "Brass_instrument": "brass", "Trumpet": "brass",
    "Bowed_string_instrument": "strings", "Wind_instrument_and_woodwind_instrument": "winds", "Harmonica": "winds",
    "Accordion": "winds", "Singing": "voice", "Choir": "voice", "Guitar": "guitar", "Harp": "guitar",
    "Plucked_string_instrument": "guitar",
}
_MULTI = sorted((w for w in TYPE_WORDS if " " in w), key=len, reverse=True)


def words_to_types(text: str) -> List[str]:
    """Types named in a text (already expanded), in order of appearance."""
    text = " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "
    found: List[str] = []
    for phrase in _MULTI:
        if f" {phrase} " in text:
            found.append(TYPE_WORDS[phrase])
            text = text.replace(f" {phrase} ", " ")
    for word in text.split():
        if word in TYPE_WORDS and TYPE_WORDS[word] not in found:
            found.append(TYPE_WORDS[word])
    return found


def preset_types(entry: Dict[str, object]) -> List[str]:
    """Types of a bank preset: from its category path when it names one, otherwise from its names."""
    category = str(entry.get("category") or "").replace("\\", " ").replace("/", " ").replace("_", " ")
    found = words_to_types(expand(category))
    if found:
        return found
    names: Iterable[str] = [str(entry.get("name", "")), *entry.get("aliases", [])[:4]]
    for name in names:
        found = words_to_types(expand(name))
        if found:
            return found
    return []


def fsd_types(labels: str) -> List[str]:
    for label in labels.split(","):
        if label in FSD_TYPES:
            return [FSD_TYPES[label]]
    return []


def multi_hot(types: Iterable[str]) -> np.ndarray:
    out = np.zeros(len(TYPES), dtype=np.float32)
    for t in types:
        out[TYPES.index(t)] = 1.0
    return out
