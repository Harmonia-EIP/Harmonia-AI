"""Harmonia v3 analog engine: the 46 parameters, their physical ranges and the 0..1 mapping.

The first 20 keep their v2 names. New parameters were chosen from what real presets use
(scripts/v3/analyze_presets.py on OB-Xf and Surge XT banks). Every new parameter has a neutral
default, so a v2 preset (see from_v2_physical) keeps its sound apart from engine quality fixes:
band-limited oscillators and exponential decay/release (v2 ramps linearly, like juce::ADSR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class V3Param:
    name: str
    start: float
    end: float
    default: float
    interval: float = 0.0
    midpoint: Optional[float] = None  # log-like skew: this physical value sits at 0.5
    choices: Optional[int] = None
    unit: str = ""

    @property
    def skew(self) -> float:
        if self.midpoint is None:
            return 1.0
        return math.log(0.5) / math.log((self.midpoint - self.start) / (self.end - self.start))


SPECS = (
    # --- v2 parameters (same names; lfo_to_pitch, lfo_to_cutoff, filter_env_amount now in cents / octaves) ---
    V3Param("osc_1_waveform", 0, 3, 0, 1, choices=4, unit="sine|triangle|saw|pulse"),
    V3Param("osc_2_waveform", 0, 3, 0, 1, choices=4, unit="sine|triangle|saw|pulse"),
    V3Param("osc_mix", 0, 1, 0, unit="osc1..osc2"),
    V3Param("osc_2_detune", 0, 100, 0, unit="cents"),
    V3Param("noise_level", 0, 1, 0),
    V3Param("filter_cutoff", 20, 20000, 20000, midpoint=1000, unit="Hz"),
    V3Param("filter_resonance", 0, 0.95, 0),
    V3Param("filter_type", 0, 2, 0, 1, choices=3, unit="lowpass|bandpass|highpass"),
    V3Param("amp_attack", 1, 5000, 1, midpoint=100, unit="ms"),
    V3Param("amp_decay", 10, 20000, 300, midpoint=500, unit="ms"),
    V3Param("amp_sustain", 0, 1, 1),
    V3Param("amp_release", 10, 20000, 300, midpoint=500, unit="ms"),
    V3Param("filter_env_amount", -10, 10, 0, unit="octaves"),
    V3Param("filter_env_decay", 10, 20000, 300, midpoint=500, unit="ms"),
    V3Param("lfo_rate", 0.05, 40, 5, midpoint=2, unit="Hz"),
    V3Param("lfo_to_pitch", 0, 1200, 0, midpoint=50, unit="cents"),
    V3Param("lfo_to_cutoff", 0, 4, 0, midpoint=1, unit="octaves"),
    V3Param("velocity_to_filter", 0, 1, 0),
    V3Param("distortion_mix", 0, 1, 0),
    V3Param("reverb_mix", 0, 1, 0),
    # --- oscillators ---
    V3Param("osc_1_coarse", -24, 24, 0, 1, unit="semitones"),
    V3Param("osc_2_coarse", -24, 24, 0, 1, unit="semitones"),
    V3Param("pulse_width", 0.05, 0.95, 0.5),
    V3Param("osc_sync", 0, 1, 0, 1, choices=2, unit="off|osc2 synced to osc1"),
    V3Param("fm_amount", 0, 48, 0, midpoint=6, unit="semitones of osc2 pitch per unit of osc1 (cross-mod)"),
    V3Param("ring_mod", 0, 1, 0),
    V3Param("unison_voices", 1, 16, 1, 1, unit="voices"),
    V3Param("unison_detune", 0, 50, 10, midpoint=10, unit="cents"),
    # --- filter ---
    V3Param("filter_slope", 0, 1, 0, 1, choices=2, unit="12 dB|24 dB"),
    V3Param("filter_keytrack", 0, 1, 0),
    V3Param("filter_env_attack", 1, 5000, 1, midpoint=100, unit="ms"),
    V3Param("filter_env_sustain", 0, 1, 0),
    V3Param("filter_env_release", 10, 20000, 50, midpoint=500, unit="ms"),
    # --- amp / pitch ---
    V3Param("velocity_to_amp", 0, 1, 0.6),
    V3Param("pitch_env_amount", -24, 24, 0, unit="semitones"),
    V3Param("pitch_env_decay", 1, 5000, 100, midpoint=100, unit="ms"),
    # --- LFO ---
    V3Param("lfo_waveform", 0, 4, 0, 1, choices=5, unit="sine|triangle|saw|square|sample&hold"),
    V3Param("lfo_delay", 0, 5000, 0, midpoint=500, unit="ms"),
    V3Param("lfo_to_amp", 0, 1, 0),
    V3Param("lfo_to_pw", 0, 1, 0),
    # --- effects ---
    V3Param("chorus_mix", 0, 1, 0),
    V3Param("delay_time", 10, 2000, 300, midpoint=300, unit="ms"),
    V3Param("delay_feedback", 0, 0.9, 0.3),
    V3Param("delay_mix", 0, 1, 0),
    V3Param("reverb_size", 0, 1, 0.7),
    # --- added after listening: real presets spread their voices in stereo (66 % of OB-Xf presets) ---
    V3Param("stereo_width", 0, 1, 0, unit="unison voices and osc1/osc2 spread"),
)

NAMES = tuple(spec.name for spec in SPECS)
INDEX: Dict[str, int] = {name: i for i, name in enumerate(NAMES)}
BY_NAME: Dict[str, V3Param] = {spec.name: spec for spec in SPECS}
DEFAULTS = np.array([spec.default for spec in SPECS], dtype=np.float64)
N_PARAMS = len(SPECS)


def snap(spec: V3Param, value: float) -> float:
    value = min(spec.end, max(spec.start, float(value)))
    if spec.interval > 0:
        value = spec.start + spec.interval * math.floor((value - spec.start) / spec.interval + 0.5)
    return min(spec.end, max(spec.start, value))


def to_normalized(spec: V3Param, value: float) -> float:
    proportion = (snap(spec, value) - spec.start) / (spec.end - spec.start)
    return proportion ** spec.skew if spec.skew != 1.0 and proportion > 0 else proportion


def from_normalized(spec: V3Param, normalized: float) -> float:
    proportion = min(1.0, max(0.0, float(normalized)))
    if spec.skew != 1.0 and proportion > 0:
        proportion = math.exp(math.log(proportion) / spec.skew)
    return snap(spec, spec.start + (spec.end - spec.start) * proportion)


def physical_from_dict(values: Dict[str, float]) -> np.ndarray:
    """Physical vector from a partial {name: value} dict (missing parameters keep their default)."""
    out = DEFAULTS.copy()
    for name, value in values.items():
        out[INDEX[name]] = snap(BY_NAME[name], value)
    return out


def normalize(physical: Sequence[float]) -> np.ndarray:
    return np.array([to_normalized(spec, v) for spec, v in zip(SPECS, physical)], dtype=np.float64)


def denormalize(normalized: Sequence[float]) -> np.ndarray:
    return np.array([from_normalized(spec, v) for spec, v in zip(SPECS, normalized)], dtype=np.float64)


def from_v2_physical(v2: Sequence[float]) -> np.ndarray:
    """A v2 preset (20 physical values, src/synth/params.py order) in the v3 engine."""
    v2 = list(v2)
    values = dict(zip(NAMES[:20], v2))
    values["filter_env_amount"] = v2[12] * 4.0
    values["lfo_to_pitch"] = v2[15] * 50.0
    values["lfo_to_cutoff"] = v2[16] * 2.0
    values["reverb_size"] = 0.55 + 0.35 * v2[19]
    return physical_from_dict(values)
