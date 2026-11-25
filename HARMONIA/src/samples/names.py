"""Note, loudness and round robin read from a recording's file name (#41).

VSCO 2 and VCSL ship their recordings as plain files named after what they play, e.g.
"VlnEns_susVib_A2_v1.wav", "GrandPno_Main_Sus_C7_v4_rr1.wav", "KSHarp_D4_mf.wav", "Upright1_Sus_C0_vl1_rr1.wav".
The octave numbering differs between libraries: src/samples/pitch.py checks it by ear afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.samples.sfz import note_number

NOTE = re.compile(r"^([A-Ga-g][#b]?-?\d)$")
LEVEL = re.compile(r"^(?:v|vl|vel|dyn)(\d+)$", re.I)
DYNAMIC = re.compile(r"^(ppp|pp|p|mp|mf|f|ff|fff)(\d*)$")
DYNAMIC_ORDER = ("ppp", "pp", "p", "mp", "mf", "f", "ff", "fff")
ROUND_ROBIN = re.compile(r"^(?:rr|r)(\d+)$", re.I)
RELEASE = re.compile(r"^(?:rel|release)$", re.I)
RELEASE_FOLDERS = {"releases", "release", "rel"}


@dataclass(frozen=True)
class Named:
    note: Optional[int]  # MIDI number as named (octave not checked yet)
    level: float         # loudness rank: higher is louder (comparable within one instrument)
    round_robin: int
    release: bool


def read(path: Path) -> Named:
    stem = Path(path).stem
    tokens = [t for t in re.split(r"[_ ]+", stem) if t]
    note, level, rr, release = None, 0.0, 1, False
    for i, token in enumerate(tokens):
        if note is None and NOTE.match(token):
            note = note_number(token)
            continue
        match = LEVEL.match(token)
        if match:
            level = float(match.group(1))
            continue
        match = DYNAMIC.match(token)
        if match and note is not None:  # "f" alone before the note is part of a name, after it a dynamic
            level = float(DYNAMIC_ORDER.index(match.group(1)))
            if match.group(2):
                rr = int(match.group(2))
            continue
        match = ROUND_ROBIN.match(token)
        if match:
            rr = int(match.group(1))
            continue
        if RELEASE.match(token) or token.lower().endswith("-rel"):
            release = True
            continue
        if note is not None and re.fullmatch(r"\d{1,3}", token) and i == len(tokens) - 1:
            rr = int(token)  # "..._A2_v1_1.wav": a take number
    if any(part.lower() in RELEASE_FOLDERS for part in Path(path).parent.parts):
        release = True
    return Named(note=note, level=level, round_robin=rr, release=release)
