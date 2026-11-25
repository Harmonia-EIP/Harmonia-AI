"""The instruments the sample source can play, and which recording answers which note and velocity (#41).

An instrument is one articulation of one recorded instrument ("Violin Section · susVib"):
- Salamander Grand Piano: read from its SFZ map (src/samples/sfz.py);
- VSCO 2 and VCSL: one folder of recordings named after their notes (src/samples/names.py), the octave
  checked by ear on every recording (src/samples/pitch.py). A folder is kept when most of its recordings name
  a note, it plays at least MIN_NOTES different notes and three quarters of its recordings agree on the octave;
  unpitched folders (drums, effects) and bell-like sounds whose octave is in doubt are left for later.

Each region: the file (relative to the library folder), the keys and velocities it answers, its root key,
its round robin and whether it is a release sound. Notes between two recorded notes take the nearest one;
velocity layers split 1..127 evenly, softest first.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf

from src.samples import names, pitch, sfz

AUDIO = (".wav", ".flac")
MIN_NOTES = 4
NAMED_SHARE = 0.5  # below this share of recordings named after a note, the folder mixes several instruments
WRAPPERS = {"sustains", "sustain", "sus", "notes", "samples"}
FAMILIES = (  # first match on the folder path, lower case
    ("piano", "keys"), ("harpsichord", "keys"), ("keys", "keys"), ("clavi", "keys"), ("tx81z", "keys"),
    ("organ", "organ"),
    ("violin", "strings"), ("viola", "strings"), ("cello", "strings"), ("contrabass", "strings"),
    ("harp", "plucked"), ("chordophones", "plucked"), ("strumstick", "plucked"),
    ("trumpet", "brass"), ("trombone", "brass"), ("horn", "brass"), ("tuba", "brass"), ("lip aerophones", "brass"),
    ("flute", "woodwinds"), ("piccolo", "woodwinds"), ("recorder", "woodwinds"), ("ocarina", "woodwinds"),
    ("clarinet", "woodwinds"), ("oboe", "woodwinds"), ("bassoon", "woodwinds"), ("sax", "woodwinds"),
    ("harmonica", "woodwinds"), ("aerophones", "woodwinds"), ("woodwinds", "woodwinds"),
    ("marimba", "mallets"), ("xylo", "mallets"), ("glock", "mallets"), ("vibraphone", "mallets"),
    ("kalimba", "mallets"), ("idiophones", "mallets"), ("bell", "mallets"),
    ("timpani", "drums"), ("membranophones", "drums"), ("drum", "drums"), ("percussion", "drums"),
)


def family(path: str) -> str:
    lower = path.lower()
    return next((name for key, name in FAMILIES if key in lower), "other")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def display_name(parts: List[str]) -> str:
    kept = [p for p in parts if p.lower() not in WRAPPERS]
    if len(kept) >= 2:
        return f"{kept[-2]} · {kept[-1]}"
    return kept[-1] if kept else parts[-1]


def key_ranges(notes: List[int]) -> Dict[int, Tuple[int, int]]:
    """Each recorded note answers the keys nearer to it than to its neighbours (ties go up)."""
    notes = sorted(set(notes))
    ranges = {}
    for i, note in enumerate(notes):
        lo = 0 if i == 0 else (notes[i - 1] + note) // 2 + 1
        hi = 127 if i == len(notes) - 1 else (note + notes[i + 1]) // 2
        ranges[note] = (lo, hi)
    return ranges


def velocity_ranges(count: int) -> List[Tuple[int, int]]:
    """count layers splitting 1..127 evenly, softest first."""
    edges = [round(1 + i * 127 / count) for i in range(count + 1)]
    return [(edges[i], (edges[i + 1] - 1) if i < count - 1 else 127) for i in range(count)]


def _info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"sample_rate": int(info.samplerate), "channels": int(info.channels), "frames": int(info.frames)}


def folder_instrument(library: str, root: Path, folder: Path, files: List[Path]) -> Tuple[Optional[dict], str]:
    """(instrument, "") or (None, why it is left out)."""
    read = [(f, names.read(f)) for f in files]
    attack = [(f, n) for f, n in read if n.note is not None and not n.release]
    notes = sorted({n.note for _, n in attack})
    if len(notes) < MIN_NOTES:
        return None, f"{len(notes)} named notes"
    if len(attack) < NAMED_SHARE * sum(1 for _, n in read if not n.release):
        return None, f"only {len(attack)} of {len(read)} recordings name a note (several instruments)"
    by_note = defaultdict(list)
    for f, n in attack:
        by_note[n.note].append((n.level, f))
    shifts = []
    for f, n in attack:
        audio, rate = sf.read(str(f), always_2d=True)
        shifts.append(pitch.octave_shift(audio, rate, n.note))
    shift = pitch.agreed_shift(shifts)
    if shift is None:
        clear = [s for s in shifts if s is not None]
        return None, f"octave in doubt ({len(clear)} clear of {len(shifts)}: {sorted(set(clear))})"
    keys = key_ranges([n + shift for n in notes])
    regions = []
    for note in notes:
        levels = sorted({level for level, _ in by_note[note]})
        layer_velocities = dict(zip(levels, velocity_ranges(len(levels))))
        for f, n in attack:
            if n.note != note:
                continue
            lo, hi = keys[note + shift]
            lovel, hivel = layer_velocities[n.level]
            regions.append({"file": f.relative_to(root).as_posix(), "key": note + shift, "lokey": lo, "hikey": hi,
                            "lovel": lovel, "hivel": hivel, "round_robin": n.round_robin, "trigger": "attack",
                            **_info(f)})
    parts = list(folder.relative_to(root).parts)
    instrument = {"id": f"{library}/{'/'.join(slug(p) for p in parts if p.lower() not in WRAPPERS)}",
                  "name": display_name(parts), "library": library, "family": family(folder.relative_to(root).as_posix()),
                  "octave_shift": shift, "regions": sorted(regions, key=lambda r: (r["key"], r["lovel"], r["round_robin"]))}
    return instrument, ""


def sfz_instrument(library: str, root: Path, sfz_file: Path, name: str, family_name: str) -> dict:
    regions = []
    for region in sfz.parse(sfz_file):
        path = (sfz_file.parent / region.sample).resolve()
        regions.append({"file": path.relative_to(root.resolve()).as_posix(), "key": region.pitch_keycenter,
                        "lokey": region.lokey, "hikey": region.hikey, "lovel": region.lovel, "hivel": region.hivel,
                        "round_robin": region.seq_position, "trigger": region.trigger,
                        "volume_db": region.volume, "tune_cents": region.tune, **_info(path)})
    return {"id": f"{library}/{slug(name)}", "name": name, "library": library, "family": family_name,
            "octave_shift": 0, "regions": regions}


def summary(instrument: dict) -> dict:
    """What the charter and the plugin's menu need, without the regions."""
    attack = [r for r in instrument["regions"] if r["trigger"] == "attack"]
    keys = [r["key"] for r in attack]
    return {"id": instrument["id"], "name": instrument["name"], "library": instrument["library"],
            "family": instrument["family"], "range": [min(keys), max(keys)], "notes": len(set(keys)),
            "layers": max(sum(1 for r in attack if r["key"] == k and r["round_robin"] == 1) for k in set(keys)),
            "regions": len(instrument["regions"])}
