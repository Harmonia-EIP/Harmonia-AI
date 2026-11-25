"""Presets <-> vectors in [-1, 1] for the v3 generators.

Analog presets use the normalized analog parameters, in charter order. Two sizes:
- the 46 parameters of charter 3.0 (the v3 and v4 generators): the sample source stays off;
- ANALOG_DIM values with the sample source of charter 3.1 (#46): the 46, then at `sample_instrument`'s place a
  switch (1: a recorded instrument plays, -1: none), level, tune and start, then the instrument's timbre
  coordinates (src/samples/descriptors.py), so that a generator writes "what it sounds like" rather than an
  instrument number. Decoding takes the instrument nearest to the coordinates.
A sample source that plays nothing (no instrument, or level 0) is always written the same way: switch off,
level 0, tune and start at their defaults, coordinates 0.

DX7 voices use ordinal fields scaled to [-1, 1] and one-hot vectors for the categorical ones (algorithm,
keyboard curves, LFO wave), so the generator never has to "interpolate" between two unrelated algorithms.
Decoding always yields a valid preset.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.samples import descriptors as timbres
from src.synth import v3_params as P

# --- analog -----------------------------------------------------------------------------------------------

ANALOG_DIM_V3 = P.CORE_PARAMS
INSTRUMENT, LEVEL, TUNE, START = (P.INDEX[name] for name in P.SAMPLE_PARAMS)
ANALOG_DIM = P.N_PARAMS + timbres.load().dims


def _table(table: Optional[timbres.Descriptors]) -> timbres.Descriptors:
    return table if table is not None else timbres.load()


def analog_encode(physical: np.ndarray, dim: int = ANALOG_DIM, table: Optional[timbres.Descriptors] = None
                  ) -> np.ndarray:
    """The codec vector of an analog preset (a 46-value physical vector is completed with defaults first);
    `dim` is ANALOG_DIM_V3 or the full size (P.N_PARAMS + the table's coordinates)."""
    physical = P.complete(physical)
    if dim == ANALOG_DIM_V3:
        return P.normalize(physical)[:dim] * 2.0 - 1.0
    table = _table(table)
    if dim != P.N_PARAMS + table.dims:
        raise ValueError(f"an analog vector has {ANALOG_DIM_V3} or {P.N_PARAMS + table.dims} values, not {dim}")
    choice = int(round(physical[INSTRUMENT]))
    plays = choice > 0 and physical[LEVEL] > 0.0
    if not plays:
        physical = physical.copy()
        physical[[INSTRUMENT, LEVEL, TUNE, START]] = P.DEFAULTS[[INSTRUMENT, LEVEL, TUNE, START]]
    head = P.normalize(physical) * 2.0 - 1.0
    head[INSTRUMENT] = 1.0 if plays else -1.0
    return np.concatenate([head, table.coordinates(choice) if plays else np.zeros(table.dims)])


def analog_decode(vector: np.ndarray, table: Optional[timbres.Descriptors] = None) -> np.ndarray:
    """Physical values of every analog parameter. A 46-value vector leaves the sample source off; a full one
    plays the instrument nearest to its coordinates when its switch is on."""
    v = np.clip(np.asarray(vector, dtype=np.float64), -1.0, 1.0)
    if len(v) == ANALOG_DIM_V3:
        return P.complete(np.array([P.from_normalized(spec, (x + 1.0) / 2.0) for spec, x in zip(P.SPECS, v)]))
    table = _table(table)
    if len(v) != P.N_PARAMS + table.dims:
        raise ValueError(f"an analog vector has {ANALOG_DIM_V3} or {P.N_PARAMS + table.dims} values, not {len(v)}")
    physical = P.denormalize((v[:P.N_PARAMS] + 1.0) / 2.0)
    if v[INSTRUMENT] > 0.0:
        physical[INSTRUMENT] = table.nearest(v[P.N_PARAMS:])
    else:
        physical[[INSTRUMENT, LEVEL, TUNE, START]] = P.DEFAULTS[[INSTRUMENT, LEVEL, TUNE, START]]
    return physical


def analog_positions(per_parameter: np.ndarray, dim: int) -> np.ndarray:
    """A per-parameter mask (P.NAMES order) over the codec vector: the timbre coordinates follow
    `sample_instrument` (locking the instrument locks them too)."""
    per_parameter = np.asarray(per_parameter, dtype=bool)
    if dim == ANALOG_DIM_V3:
        return per_parameter[:dim].copy()
    return np.concatenate([per_parameter, np.full(dim - P.N_PARAMS, per_parameter[INSTRUMENT])])


def analog_continuous(dim: int) -> np.ndarray:
    """Positions of the codec vector a local search may move by small steps: the continuous parameters (the
    switch, the tune and the timbre coordinates only change by choosing another instrument or octave)."""
    continuous = np.array([spec.interval == 0.0 for spec in P.SPECS])
    return analog_positions(continuous, dim) & (np.arange(dim) < P.N_PARAMS)


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
