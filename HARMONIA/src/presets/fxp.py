"""VST2 .fxp program files carrying an opaque chunk (Surge XT, OB-Xf), plus the two chunk payloads."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET  # nosec B405 - parses local preset files fetched from pinned sources
from pathlib import Path
from typing import Dict, List, Tuple

FXP_HEADER = 60  # 'CcnK' size 'FPCh' version fxID fxVersion numPrograms prgName[28] chunkSize


def read_chunk(path: Path) -> Tuple[bytes, bytes]:
    """(plugin id, chunk bytes) of an opaque-chunk .fxp file."""
    data = path.read_bytes()
    if data[0:4] != b"CcnK" or data[8:12] != b"FPCh":
        raise ValueError(f"{path}: not an opaque-chunk fxp")
    size = struct.unpack(">I", data[56:60])[0]
    return data[16:20], data[FXP_HEADER:FXP_HEADER + size]


def _xml_from(chunk: bytes, start: int) -> ET.Element:
    end = chunk.find(b"\x00", start)
    text = chunk[start:end if end != -1 else len(chunk)]
    return ET.fromstring(text)  # nosec B314 - local files from pinned sources


def read_obxf(path: Path) -> Dict[str, object]:
    """OB-Xf patch: every VST parameter (normalized 0..1) plus name/author/license/category."""
    plugin, chunk = read_chunk(path)
    if plugin != b"OBXf":
        raise ValueError(f"{path}: plugin id {plugin!r}")
    root = _xml_from(chunk, chunk.find(b"<?xml"))
    attrs = dict(root.attrib)
    meta = {key: attrs.pop(key, "") for key in ("programName", "author", "license", "category", "project",
                                                 "ob-xf_version", "voiceCount")}
    params = {}
    for key, value in attrs.items():
        try:
            params[key] = float(value)
        except ValueError:
            meta[key] = value
    return {"meta": meta, "params": params}


def read_surge(path: Path) -> Dict[str, object]:
    """Surge XT patch: parameters (physical values), modulation routings and metadata from its XML."""
    plugin, chunk = read_chunk(path)
    if plugin != b"cjs3" or chunk[0:4] != b"sub3":
        raise ValueError(f"{path}: not a Surge patch")
    xml_size = struct.unpack("<I", chunk[4:8])[0]
    root = ET.fromstring(chunk[32:32 + xml_size].split(b"\x00")[0])  # nosec B314 - local pinned files
    meta_el = root.find("meta")
    meta = dict(meta_el.attrib) if meta_el is not None else {}
    meta["revision"] = root.get("revision", "")
    params: Dict[str, object] = {}
    routings: List[Dict[str, object]] = []
    parameters = root.find("parameters")
    for el in parameters if parameters is not None else []:
        kind, value = el.get("type"), el.get("value")
        if value is None:
            continue
        params[el.tag] = int(value) if kind == "0" else float(value)
        for flag in ("deactivated", "extend_range", "absolute", "deform_type"):
            if el.get(flag) not in (None, "0"):
                params[f"{el.tag}@{flag}"] = int(el.get(flag))
        for mod in el.findall("modrouting"):
            if mod.get("muted") == "1":
                continue
            routings.append({"target": el.tag, "source": int(mod.get("source", -1)),
                             "index": int(mod.get("source_index", 0)), "depth": float(mod.get("depth", 0.0))})
    sections = sorted({child.tag for child in root if child.tag not in ("meta", "parameters")})
    return {"meta": meta, "params": params, "modulation": routings, "sections": sections,
            "wavetable_bytes": sum(struct.unpack("<6I", chunk[8:32]))}
