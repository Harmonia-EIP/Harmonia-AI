"""Presets <-> vectors in [-1, 1] for the v3 generators.

Analog presets use the 46 normalized v3 parameters. DX7 voices use ordinal fields scaled to [-1, 1]
and one-hot vectors for the categorical ones (algorithm, keyboard curves, LFO wave), so the generator
never has to "interpolate" between two unrelated algorithms. Decoding always yields a valid preset.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.synth import v3_params as P

# --- analog -----------------------------------------------------------------------------------------------

ANALOG_DIM = P.N_PARAMS


def analog_encode(physical: np.ndarray) -> np.ndarray:
    return P.normalize(physical) * 2.0 - 1.0


def analog_decode(vector: np.ndarray) -> np.ndarray:
    return P.denormalize((np.clip(vector, -1.0, 1.0) + 1.0) / 2.0)


# --- DX7 --------------------------------------------------------------------------------------------------

# (field, kind, size): kind "ord" = integer 0..size, "cat" = one of `size` classes, "list" = 4 ordinals 0..99
OP_FIELDS: List[Tuple[str, str, int]] = [
    ("rates", "list", 99), ("levels", "list", 99), ("kbd_breakpoint", "ord", 99), ("kbd_left_depth", "ord", 99),
    ("kbd_right_depth", "ord", 99), ("kbd_left_curve", "cat", 4), ("kbd_right_curve", "cat", 4),
    ("rate_scaling", "ord", 7), ("amp_mod_sens", "ord", 3), ("key_vel_sens", "ord", 7), ("output_level", "ord", 99),
    ("osc_mode", "ord", 1), ("freq_coarse", "ord", 31), ("freq_fine", "ord", 99), ("detune", "ord", 14),
]
GLOBAL_FIELDS: List[Tuple[str, str, int]] = [
    ("pitch_eg_rates", "list", 99), ("pitch_eg_levels", "list", 99), ("algorithm", "cat", 32), ("feedback", "ord", 7),
    ("osc_key_sync", "ord", 1), ("lfo_speed", "ord", 99), ("lfo_delay", "ord", 99), ("lfo_pitch_mod_depth", "ord", 99),
    ("lfo_amp_mod_depth", "ord", 99), ("lfo_key_sync", "ord", 1), ("lfo_wave", "cat", 6), ("pitch_mod_sens", "ord", 7),
    ("transpose", "ord", 48),
]


def _width(kind: str, size: int) -> int:
    return 4 if kind == "list" else (size if kind == "cat" else 1)


DX7_DIM = 6 * sum(_width(k, s) for _, k, s in OP_FIELDS) + sum(_width(k, s) for _, k, s in GLOBAL_FIELDS)


def _encode_fields(values: Dict[str, object], fields, out: List[float]) -> None:
    for name, kind, size in fields:
        v = values[name]
        if kind == "list":
            out.extend(2.0 * x / size - 1.0 for x in v)
        elif kind == "cat":
            one_hot = [-1.0] * size
            one_hot[int(v)] = 1.0
            out.extend(one_hot)
        else:
            out.append(2.0 * v / size - 1.0)


def _decode_fields(vector: np.ndarray, start: int, fields) -> Tuple[Dict[str, object], int]:
    values: Dict[str, object] = {}
    i = start
    for name, kind, size in fields:
        if kind == "list":
            values[name] = [int(round((x + 1.0) / 2.0 * size)) for x in np.clip(vector[i:i + 4], -1, 1)]
            i += 4
        elif kind == "cat":
            values[name] = int(np.argmax(vector[i:i + size]))
            i += size
        else:
            values[name] = int(round((float(np.clip(vector[i], -1, 1)) + 1.0) / 2.0 * size))
            i += 1
    return values, i


def dx7_encode(voice: Dict[str, object]) -> np.ndarray:
    out: List[float] = []
    for op in voice["operators"]:
        _encode_fields(op, OP_FIELDS, out)
    _encode_fields(voice, GLOBAL_FIELDS, out)
    return np.array(out, dtype=np.float32)


def dx7_decode(vector: np.ndarray, name: str = "HARMONIA") -> Dict[str, object]:
    vector = np.asarray(vector, dtype=np.float64)
    operators = []
    i = 0
    for _ in range(6):
        op, i = _decode_fields(vector, i, OP_FIELDS)
        operators.append(op)
    voice, i = _decode_fields(vector, i, GLOBAL_FIELDS)
    voice["operators"] = operators
    voice["name"] = name[:10]
    return voice
