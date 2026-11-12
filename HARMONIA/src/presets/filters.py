"""Presets Harmonia cannot play: left out of matching, listening and the bank.

- teaching material rather than sounds (Surge tutorials, templates, init patches);
- arpeggios and step sequences: Harmonia has no arpeggiator and plays one held note;
- record-noise effects (vinyl crackle), heard as impossible to reproduce in listening round 2.
"""

from __future__ import annotations

import re
from typing import Dict

NOT_A_SOUND = re.compile(r"tutorial|template|\binit\b", re.I)
SEQUENCED = re.compile(r"\barp|arpegg|\bseq|sequence|vinyl", re.I)
SURGE_STEPSEQ_SHAPE = 7
SURGE_LFO_SOURCES = range(17, 29)  # voice LFO 1-6, scene LFO 1-6 (src/presets/analyze_tables.py)


def _surge_steps_sound(rec: Dict[str, object]) -> bool:
    """A Surge step sequencer driving pitch, level or cutoff makes the patch a sequence."""
    params = rec.get("params", {})
    for m in rec.get("modulation", []):
        if m["source"] not in SURGE_LFO_SOURCES or abs(m["depth"]) < 1e-3:
            continue
        lfo = m["source"] - 17
        scene = m["target"][:1]
        if params.get(f"{scene}_lfo{lfo}_shape") == SURGE_STEPSEQ_SHAPE and any(
                k in m["target"] for k in ("pitch", "level", "volume", "cutoff")):
            return True
    return False


def playable(rec: Dict[str, object]) -> bool:
    text = " ".join(str(rec.get(k) or "") for k in ("name", "category", "path"))
    text += " " + " ".join(str(p) for p in rec.get("paths", []))
    if NOT_A_SOUND.search(text) or SEQUENCED.search(text):
        return False
    return not (rec.get("source") == "surge" and _surge_steps_sound(rec))
