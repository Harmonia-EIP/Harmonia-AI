"""Yamaha DX7 sysex reader: 32-voice bulk dumps (VMEM, packed) and single voices (VCED).

Layouts follow the DX7 owner's manual / service manual MIDI data format tables, as implemented by Dexed
(asb2m10/dexed Source/PluginData.cpp). Operators are returned op1..op6 (the sysex stores op6 first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

VMEM_HEADER = bytes([0x43, 0x00, 0x09, 0x20, 0x00])  # after F0; byte 1 carries the MIDI channel
VCED_HEADER = bytes([0x43, 0x00, 0x00, 0x01, 0x1B])
VOICE_BYTES = 128
VCED_BYTES = 155

# Maximum legal value of every decoded field; a voice with a field above it is treated as corrupt.
OP_LIMITS = {
    "rates": 99, "levels": 99, "kbd_breakpoint": 99, "kbd_left_depth": 99, "kbd_right_depth": 99,
    "kbd_left_curve": 3, "kbd_right_curve": 3, "rate_scaling": 7, "amp_mod_sens": 3, "key_vel_sens": 7,
    "output_level": 99, "osc_mode": 1, "freq_coarse": 31, "freq_fine": 99, "detune": 14,
}
GLOBAL_LIMITS = {
    "pitch_eg_rates": 99, "pitch_eg_levels": 99, "algorithm": 31, "feedback": 7, "osc_key_sync": 1,
    "lfo_speed": 99, "lfo_delay": 99, "lfo_pitch_mod_depth": 99, "lfo_amp_mod_depth": 99, "lfo_key_sync": 1,
    "lfo_wave": 5, "pitch_mod_sens": 7, "transpose": 48,
}


def _name(raw: bytes) -> str:
    chars = []
    for b in raw:
        b &= 0x7F
        chars.append(chr(b) if 32 <= b < 127 else " ")
    return "".join(chars).strip()


def _within(values: Dict[str, object], limits: Dict[str, int]) -> bool:
    for key, limit in limits.items():
        v = values[key]
        if any(x > limit for x in v) if isinstance(v, list) else v > limit:
            return False
    return True


def decode_packed_voice(v: bytes) -> Optional[Dict[str, object]]:
    """One 128-byte VMEM voice -> dict, or None when a field is out of range."""
    operators: List[Dict[str, object]] = []
    for op in range(6):
        o = v[op * 17:(op + 1) * 17]
        operators.append({
            "rates": list(o[0:4]), "levels": list(o[4:8]),
            "kbd_breakpoint": o[8], "kbd_left_depth": o[9], "kbd_right_depth": o[10],
            "kbd_left_curve": o[11] & 0x03, "kbd_right_curve": (o[11] >> 2) & 0x03,
            "rate_scaling": o[12] & 0x07, "detune": (o[12] >> 3) & 0x0F,
            "amp_mod_sens": o[13] & 0x03, "key_vel_sens": (o[13] >> 2) & 0x07,
            "output_level": o[14], "osc_mode": o[15] & 0x01, "freq_coarse": (o[15] >> 1) & 0x1F,
            "freq_fine": o[16],
        })
    operators.reverse()  # op6..op1 -> op1..op6
    voice = {
        "operators": operators,
        "pitch_eg_rates": list(v[102:106]), "pitch_eg_levels": list(v[106:110]),
        "algorithm": v[110] & 0x1F, "feedback": v[111] & 0x07, "osc_key_sync": (v[111] >> 3) & 0x01,
        "lfo_speed": v[112], "lfo_delay": v[113], "lfo_pitch_mod_depth": v[114], "lfo_amp_mod_depth": v[115],
        "lfo_key_sync": v[116] & 0x01, "lfo_wave": (v[116] >> 1) & 0x07, "pitch_mod_sens": (v[116] >> 4) & 0x07,
        "transpose": v[117], "name": _name(v[118:128]),
    }
    return voice if _valid(voice) else None


def decode_vced(v: bytes) -> Optional[Dict[str, object]]:
    """One 155-byte single-voice VCED block -> same dict as decode_packed_voice."""
    operators: List[Dict[str, object]] = []
    for op in range(6):
        o = v[op * 21:(op + 1) * 21]
        operators.append({
            "rates": list(o[0:4]), "levels": list(o[4:8]),
            "kbd_breakpoint": o[8], "kbd_left_depth": o[9], "kbd_right_depth": o[10],
            "kbd_left_curve": o[11], "kbd_right_curve": o[12], "rate_scaling": o[13],
            "amp_mod_sens": o[14], "key_vel_sens": o[15], "output_level": o[16],
            "osc_mode": o[17], "freq_coarse": o[18], "freq_fine": o[19], "detune": o[20],
        })
    operators.reverse()
    g = v[126:155]
    voice = {
        "operators": operators,
        "pitch_eg_rates": list(g[0:4]), "pitch_eg_levels": list(g[4:8]),
        "algorithm": g[8], "feedback": g[9], "osc_key_sync": g[10],
        "lfo_speed": g[11], "lfo_delay": g[12], "lfo_pitch_mod_depth": g[13], "lfo_amp_mod_depth": g[14],
        "lfo_key_sync": g[15], "lfo_wave": g[16], "pitch_mod_sens": g[17], "transpose": g[18],
        "name": _name(g[19:29]),
    }
    return voice if _valid(voice) else None


def _valid(voice: Dict[str, object]) -> bool:
    return _within(voice, GLOBAL_LIMITS) and all(_within(op, OP_LIMITS) for op in voice["operators"])


def _messages(data: bytes) -> Iterator[bytes]:
    start = data.find(0xF0)
    while start != -1:
        end = data.find(0xF7, start + 1)
        if end == -1:
            return
        yield data[start + 1:end]
        start = data.find(0xF0, end + 1)


def read_file(path: Path) -> Iterator[Dict[str, object]]:
    """Every valid voice of a .syx file (bulk dumps, single voices, or a raw 4096-byte VMEM block)."""
    data = path.read_bytes()
    if len(data) == 32 * VOICE_BYTES and data[0] != 0xF0:
        blocks = [data]
    else:
        blocks = []
        for msg in _messages(data):
            if len(msg) >= 5 + 32 * VOICE_BYTES and msg[0] == 0x43 and msg[2:5] == VMEM_HEADER[2:5]:
                blocks.append(msg[5:5 + 32 * VOICE_BYTES])
            elif len(msg) >= 5 + VCED_BYTES and msg[0] == 0x43 and msg[2:5] == VCED_HEADER[2:5]:
                voice = decode_vced(msg[5:5 + VCED_BYTES])
                if voice:
                    voice["slot"] = 0
                    yield voice
    for block in blocks:
        for slot in range(32):
            voice = decode_packed_voice(block[slot * VOICE_BYTES:(slot + 1) * VOICE_BYTES])
            if voice:
                voice["slot"] = slot
                yield voice


# Operator routing of the 32 DX7 algorithms, from Dexed (Source/msfa/fm_core.cc, FmCore::algorithms).
# Each row lists op6..op1; bits 0-1 output bus (0 = audio out), bit 2 add, bits 4-5 input bus, 6-7 feedback.
ALGORITHM_FLAGS = (
    (0xC1, 0x11, 0x11, 0x14, 0x01, 0x14), (0x01, 0x11, 0x11, 0x14, 0xC1, 0x14),
    (0xC1, 0x11, 0x14, 0x01, 0x11, 0x14), (0xC1, 0x11, 0x94, 0x01, 0x11, 0x14),
    (0xC1, 0x14, 0x01, 0x14, 0x01, 0x14), (0xC1, 0x94, 0x01, 0x14, 0x01, 0x14),
    (0xC1, 0x11, 0x05, 0x14, 0x01, 0x14), (0x01, 0x11, 0xC5, 0x14, 0x01, 0x14),
    (0x01, 0x11, 0x05, 0x14, 0xC1, 0x14), (0x01, 0x05, 0x14, 0xC1, 0x11, 0x14),
    (0xC1, 0x05, 0x14, 0x01, 0x11, 0x14), (0x01, 0x05, 0x05, 0x14, 0xC1, 0x14),
    (0xC1, 0x05, 0x05, 0x14, 0x01, 0x14), (0xC1, 0x05, 0x11, 0x14, 0x01, 0x14),
    (0x01, 0x05, 0x11, 0x14, 0xC1, 0x14), (0xC1, 0x11, 0x02, 0x25, 0x05, 0x14),
    (0x01, 0x11, 0x02, 0x25, 0xC5, 0x14), (0x01, 0x11, 0x11, 0xC5, 0x05, 0x14),
    (0xC1, 0x14, 0x14, 0x01, 0x11, 0x14), (0x01, 0x05, 0x14, 0xC1, 0x14, 0x14),
    (0x01, 0x14, 0x14, 0xC1, 0x14, 0x14), (0xC1, 0x14, 0x14, 0x14, 0x01, 0x14),
    (0xC1, 0x14, 0x14, 0x01, 0x14, 0x04), (0xC1, 0x14, 0x14, 0x14, 0x04, 0x04),
    (0xC1, 0x14, 0x14, 0x04, 0x04, 0x04), (0xC1, 0x05, 0x14, 0x01, 0x14, 0x04),
    (0x01, 0x05, 0x14, 0xC1, 0x14, 0x04), (0x04, 0xC1, 0x11, 0x14, 0x01, 0x14),
    (0xC1, 0x14, 0x01, 0x14, 0x04, 0x04), (0x04, 0xC1, 0x11, 0x14, 0x04, 0x04),
    (0xC1, 0x14, 0x04, 0x04, 0x04, 0x04), (0xC4, 0x04, 0x04, 0x04, 0x04, 0x04),
)


def carriers(algorithm: int) -> List[int]:
    """Operator numbers (1..6) that reach the audio output for a 0-based algorithm index."""
    flags = ALGORITHM_FLAGS[algorithm]
    return sorted(6 - i for i, f in enumerate(flags) if f & 0x03 == 0)


def feedback_operator(algorithm: int) -> int:
    """Operator number (1..6) whose output feeds back into itself."""
    flags = ALGORITHM_FLAGS[algorithm]
    return next(6 - i for i, f in enumerate(flags) if f & 0x40)
