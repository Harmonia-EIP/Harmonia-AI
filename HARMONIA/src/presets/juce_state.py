"""JUCE plugin state helpers: MemoryBlock base64 ("<size>.<chars>") and pedalboard's VST3 state wrapper.

pedalboard exposes a VST3 plugin state as JUCE binary XML (b"VC2!" + uint32 size + XML) whose
<IComponent> text is the component state encoded with juce::MemoryBlock::toBase64Encoding.
"""

from __future__ import annotations

import re
import struct

ALPHABET = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def b64_decode(text: str) -> bytes:
    """juce::MemoryBlock::fromBase64Encoding: 6 bits per char, least significant bits first."""
    size_text, data = text.split(".", 1)
    size = int(size_text)
    out = bytearray(size)
    bit = 0
    for ch in data:
        value = _INDEX[ch]
        for i in range(6):
            if bit >= size * 8:
                break
            if value >> i & 1:
                out[bit >> 3] |= 1 << (bit & 7)
            bit += 1
    return bytes(out)


def b64_encode(data: bytes) -> str:
    chars = []
    nbits = len(data) * 8
    for start in range(0, nbits, 6):
        value = 0
        for i in range(6):
            bit = start + i
            if bit < nbits and data[bit >> 3] >> (bit & 7) & 1:
                value |= 1 << i
        chars.append(ALPHABET[value])
    return f"{len(data)}." + "".join(chars)


def binary_xml(xml: str) -> bytes:
    """juce::AudioProcessor::copyXmlToBinary."""
    payload = xml.encode("utf-8") + b"\x00"
    return b"VC2!" + struct.pack("<I", len(payload)) + payload


def xml_from_binary(blob: bytes) -> str:
    size = struct.unpack("<I", blob[4:8])[0]
    return blob[8:8 + size].rstrip(b"\x00").decode("utf-8")


def get_component_state(raw_state: bytes) -> bytes:
    xml = xml_from_binary(raw_state)
    match = re.search(r"<IComponent>([^<]*)</IComponent>", xml)
    if not match:
        raise ValueError("no IComponent state")
    return b64_decode(match.group(1))


def with_component_state(raw_state: bytes, component: bytes) -> bytes:
    xml = xml_from_binary(raw_state)
    xml = re.sub(r"<IComponent>[^<]*</IComponent>", lambda _: f"<IComponent>{b64_encode(component)}</IComponent>", xml)
    return binary_xml(xml)
