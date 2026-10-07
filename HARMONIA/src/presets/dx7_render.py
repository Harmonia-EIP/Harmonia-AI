"""Play a DX7 voice (src/presets/dx7.py dict) with the msfa FM core compiled by scripts/v3/build_dx7.py."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
LIBRARY = BASE_DIR / "data" / "v3" / "build" / "dx7" / (
    "libharmonia_dx7.dylib" if sys.platform == "darwin" else "libharmonia_dx7.so")
OP_FIELDS = ("kbd_breakpoint", "kbd_left_depth", "kbd_right_depth", "kbd_left_curve", "kbd_right_curve",
             "rate_scaling", "amp_mod_sens", "key_vel_sens", "output_level", "osc_mode", "freq_coarse",
             "freq_fine", "detune")
_lib: Optional[ctypes.CDLL] = None


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(LIBRARY))
        _lib.dx7_render.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        _lib.dx7_render.restype = ctypes.c_int
    return _lib


def to_patch(voice: Dict[str, object]) -> bytes:
    """Voice dict -> the 156-byte unpacked layout of the DX7 voice edit buffer (op6 first)."""
    data = bytearray()
    for op in reversed(voice["operators"]):
        data += bytes(op["rates"]) + bytes(op["levels"]) + bytes(op[field] for field in OP_FIELDS)
    data += bytes(voice["pitch_eg_rates"]) + bytes(voice["pitch_eg_levels"])
    data += bytes([voice["algorithm"], voice["feedback"], voice["osc_key_sync"], voice["lfo_speed"],
                   voice["lfo_delay"], voice["lfo_pitch_mod_depth"], voice["lfo_amp_mod_depth"],
                   voice["lfo_key_sync"], voice["lfo_wave"], voice["pitch_mod_sens"], voice["transpose"]])
    name = str(voice.get("name", "")).encode("ascii", "replace")[:10].ljust(10)
    data += name + b"\x00"
    if len(data) != 156:
        raise ValueError(f"DX7 patch must be 156 bytes, got {len(data)}")
    return bytes(data)


DX_VELOCITY = 0.7874015  # Dexed's "DX7 velocity" option: MIDI velocity scaled to the DX7 keyboard's range


def render(voice: Dict[str, object], note: int = 60, velocity: int = 100, hold_seconds: float = 1.5,
           total_seconds: float = 4.0, sample_rate: int = 48000, dx_velocity: bool = True) -> np.ndarray:
    """Mono float32 audio of one key press, as Dexed plays it (before its output volume).

    With dx_velocity the MIDI velocity is scaled like a real DX7 keyboard (Dexed's option): at full MIDI
    velocity the voices otherwise sound harsher than on the instrument (E.PIANO 1's bell attack)."""
    if dx_velocity:
        velocity = int(velocity * DX_VELOCITY)
    total = int(round(total_seconds * sample_rate))
    out = np.zeros(total, dtype=np.float32)
    status = _library().dx7_render(to_patch(voice), note, velocity, int(round(hold_seconds * sample_rate)), total,
                                   sample_rate, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if status != 0:
        raise RuntimeError(f"dx7_render failed ({status})")
    return out
