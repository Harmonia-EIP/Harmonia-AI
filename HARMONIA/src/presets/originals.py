"""Play presets in the synthesizer they were made for (OB-Xf, Surge XT) through pedalboard.

Used offline only, to compare a converted Harmonia preset with the original sound. The preset chunk of
the .fxp file replaces the plugin's own state; JUCE's private data that follows it is kept as is.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET  # nosec B405 - local preset files from pinned sources
from pathlib import Path
from typing import Optional

import numpy as np

from src.presets import fxp
from src.presets.juce_state import binary_xml, get_component_state, with_component_state, xml_from_binary

WARMUP_SECONDS = 0.5  # Surge applies a new patch on the audio thread: let it run before playing a note

PLUGIN_DIR = Path.home() / "Datasets" / "harmonia-presets" / "plugins"
PLUGINS = {
    "obxf": PLUGIN_DIR / "obxf-pkg" / "ob-xf_VST3.pkg" / "Payload" / "OB-Xf.vst3",
    "surge": PLUGIN_DIR / "surge-xt" / "Surge XT.vst3",
}


def _plugin_data_size(kind: str, component: bytes) -> int:
    """Length of the plugin's own state at the start of a JUCE VST3 component state."""
    if kind == "obxf":
        return 8 + struct.unpack("<I", component[4:8])[0]
    xml_size = struct.unpack("<I", component[4:8])[0]
    return 32 + xml_size + sum(struct.unpack("<6I", component[8:32]))


class OriginalSynth:
    def __init__(self, kind: str, plugin_path: Optional[Path] = None):
        from pedalboard import load_plugin  # training-only dependency

        self.kind = kind
        self.plugin = load_plugin(str(plugin_path or PLUGINS[kind]))
        self.default_state = bytes(self.plugin.raw_state)
        component = get_component_state(self.default_state)
        self.juce_tail = component[_plugin_data_size(kind, component):]

    def load(self, path: Path) -> None:
        _, chunk = fxp.read_chunk(path)
        chunk = chunk[:_plugin_data_size(self.kind, chunk)]
        if self.kind == "obxf":
            chunk = self._obxf_plugin_state(chunk)
        self.plugin.raw_state = with_component_state(self.default_state, chunk + self.juce_tail)
        if self.kind == "surge":
            self.plugin([], duration=WARMUP_SECONDS, sample_rate=48000, num_channels=2, reset=False)

    def _obxf_plugin_state(self, program_chunk: bytes) -> bytes:
        """An .fxp holds one program; the plugin state wraps it in <program> next to the DAW extra state."""
        program = ET.fromstring(xml_from_binary(program_chunk))  # nosec B314 - local pinned files
        state = ET.fromstring(xml_from_binary(get_component_state(self.default_state)))  # nosec B314
        node = state.find("program")
        node.attrib.clear()
        node.attrib.update({k: v for k, v in program.attrib.items() if k != "ob-xf_version"})
        state.set("ob-xf_version", program.get("ob-xf_version", state.get("ob-xf_version", "")))
        return binary_xml('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(state, encoding="unicode"))

    def render(self, note: int = 60, velocity: int = 100, hold_seconds: float = 1.5, total_seconds: float = 4.0,
               sample_rate: int = 48000) -> np.ndarray:
        from mido import Message

        messages = [Message("note_on", note=note, velocity=velocity, time=0.0),
                    Message("note_off", note=note, velocity=0, time=hold_seconds)]
        audio = self.plugin(messages, duration=total_seconds, sample_rate=sample_rate, num_channels=2, reset=True)
        return np.asarray(audio, dtype=np.float32)
