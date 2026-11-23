"""Read SFZ instrument files: which recording plays for which note and velocity (#41).

SFZ is the plain-text format the free libraries ship (Salamander Grand Piano, VSCO 2, VCSL): headers
(<control>, <global>, <master>, <group>, <region>) followed by opcodes. A region inherits the opcodes of the
headers above it. Only what Harmonia's sample source needs is kept: the file, the key and velocity ranges,
the root key, release triggers, loops, offset, volume, tuning and round robins.

    regions = parse(Path("Violin Section Sustain.sfz"))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

NOTE = re.compile(r"^([a-gA-G])([#b]?)(-?\d+)$")
OFFSETS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
LEVELS = ("global", "master", "group")


def note_number(value: str) -> int:
    """MIDI number of an SFZ key: a number, or a name such as "c4" (60, the SFZ convention), "f#3", "eb2"."""
    value = value.strip()
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    match = NOTE.match(value)
    if not match:
        raise ValueError(f"not a key: {value!r}")
    letter, accidental, octave = match.groups()
    semitone = OFFSETS[letter.lower()] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    return (int(octave) + 1) * 12 + semitone


@dataclass
class Region:
    sample: str                 # path relative to the SFZ file's folder, forward slashes
    lokey: int = 0
    hikey: int = 127
    pitch_keycenter: int = 60
    lovel: int = 1
    hivel: int = 127
    trigger: str = "attack"     # attack, release, first, legato
    loop_mode: str = "no_loop"  # no_loop, one_shot, loop_continuous, loop_sustain
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None
    offset: int = 0
    volume: float = 0.0         # dB
    tune: float = 0.0           # cents
    transpose: int = 0          # semitones
    pitch_keytrack: float = 100.0
    seq_length: int = 1
    seq_position: int = 1
    extra: Dict[str, str] = field(default_factory=dict)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _expand(text: str, folder: Path, defines: Dict[str, str], depth: int = 0) -> str:
    """#define $NAME value and #include "file.sfz" (relative to the instrument's folder)."""
    out = []
    for line in _strip_comments(text).splitlines():
        stripped = line.strip()
        match = re.match(r'#define\s+(\$\w+)\s+(.+)', stripped)
        if match:
            defines[match.group(1)] = match.group(2).strip()
            continue
        match = re.match(r'#include\s+"([^"]+)"', stripped)
        if match:
            if depth > 8:
                raise ValueError("#include nested too deep")
            path = folder / match.group(1)
            out.append(_expand(path.read_text(encoding="utf-8", errors="replace"), folder, defines, depth + 1))
            continue
        for name in sorted(defines, key=len, reverse=True):
            line = line.replace(name, defines[name])
        out.append(line)
    return "\n".join(out)


def _opcodes(body: str) -> Dict[str, str]:
    """opcode=value pairs; a value runs until the next opcode, so sample paths may hold spaces."""
    pairs = list(re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=", body))
    result = {}
    for i, match in enumerate(pairs):
        end = pairs[i + 1].start() if i + 1 < len(pairs) else len(body)
        result[match.group(1).lower()] = body[match.end():end].strip()
    return result


def _region(opcodes: Dict[str, str], default_path: str) -> Optional[Region]:
    if "sample" not in opcodes:
        return None
    sample = (default_path + opcodes["sample"]).replace("\\", "/")
    region = Region(sample=sample)
    if "key" in opcodes:
        key = note_number(opcodes["key"])
        region.lokey = region.hikey = region.pitch_keycenter = key
    for name in ("lokey", "hikey", "pitch_keycenter"):
        if name in opcodes:
            try:
                setattr(region, name, note_number(opcodes[name]))
            except ValueError:  # e.g. pitch_keycenter=sample (root read from the file): left at its default
                region.extra[name] = opcodes[name]
    for name, cast in (("lovel", int), ("hivel", int), ("offset", int), ("loop_start", int), ("loop_end", int),
                       ("loopstart", int), ("loopend", int), ("volume", float), ("tune", float), ("transpose", int),
                       ("pitch_keytrack", float), ("seq_length", int), ("seq_position", int)):
        if name in opcodes:
            try:
                value = cast(float(opcodes[name])) if cast is int else cast(opcodes[name])
            except ValueError:
                continue
            setattr(region, {"loopstart": "loop_start", "loopend": "loop_end"}.get(name, name), value)
    for name in ("trigger", "loop_mode"):
        if name in opcodes:
            setattr(region, name, opcodes[name].lower())
    known = set(Region.__dataclass_fields__) | {"key", "sample", "loopstart", "loopend"}
    region.extra.update({k: v for k, v in opcodes.items() if k not in known})
    return region


def parse(path: Path) -> List[Region]:
    """Every playable region of an SFZ file, with the opcodes inherited from its headers."""
    path = Path(path)
    text = _expand(path.read_text(encoding="utf-8", errors="replace"), path.parent, {})
    scopes: Dict[str, Dict[str, str]] = {level: {} for level in LEVELS}
    default_path, regions = "", []
    chunks = re.split(r"<(\w+)>", text)
    for i in range(1, len(chunks), 2):
        header, body = chunks[i].lower(), chunks[i + 1]
        opcodes = _opcodes(body)
        if header == "control":
            default_path = opcodes.get("default_path", default_path).replace("\\", "/")
            continue
        if header in LEVELS:
            # a header resets the scopes below it
            index = LEVELS.index(header)
            for level in LEVELS[index:]:
                scopes[level] = {}
            scopes[header] = opcodes
            continue
        if header == "region":
            merged: Dict[str, str] = {}
            for level in LEVELS:
                merged.update(scopes[level])
            merged.update(opcodes)
            region = _region(merged, default_path)
            if region is not None:
                regions.append(region)
    return regions
