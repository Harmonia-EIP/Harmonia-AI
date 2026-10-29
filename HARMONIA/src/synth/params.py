"""Charter (normalized 0..1) -> physical parameter values, exactly as the app maps them.

Mirrors `HarmoniaParams::createLayout()` (Harmonia-APP/src/parameters/HarmoniaParameters.h)
and JUCE's `RangedAudioParameter::convertFrom0to1` (NormalisableRange skew + snapToLegalValue).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.charter import PARAM_NAMES


@dataclass(frozen=True)
class AppParamSpec:
    name: str
    start: float
    end: float
    interval: float
    skew: float = 1.0
    choices: Optional[int] = None


def _log_skew(min_v: float, max_v: float, midpoint: float) -> float:
    # makeLogRange(): skew = log(0.5) / log((mid - min) / (max - min)), computed in float.
    return float(np.float32(math.log(0.5) / math.log((midpoint - min_v) / (max_v - min_v))))


APP_PARAM_SPECS: Dict[str, AppParamSpec] = {
    spec.name: spec
    for spec in (
        AppParamSpec("osc_1_waveform", 0.0, 3.0, 1.0, choices=4),
        AppParamSpec("osc_2_waveform", 0.0, 3.0, 1.0, choices=4),
        AppParamSpec("osc_mix", 0.0, 1.0, 0.0001),
        AppParamSpec("osc_2_detune", 0.0, 100.0, 0.01),
        AppParamSpec("noise_level", 0.0, 1.0, 0.0001),
        AppParamSpec("filter_cutoff", 20.0, 20000.0, 0.0001, _log_skew(20.0, 20000.0, 1000.0)),
        AppParamSpec("filter_resonance", 0.0, 0.95, 0.0001),
        AppParamSpec("filter_type", 0.0, 2.0, 1.0, choices=3),
        AppParamSpec("amp_attack", 1.0, 5000.0, 0.0001, _log_skew(1.0, 5000.0, 100.0)),
        AppParamSpec("amp_decay", 10.0, 5000.0, 0.0001, _log_skew(10.0, 5000.0, 300.0)),
        AppParamSpec("amp_sustain", 0.0, 1.0, 0.0001),
        AppParamSpec("amp_release", 10.0, 10000.0, 0.0001, _log_skew(10.0, 10000.0, 500.0)),
        AppParamSpec("filter_env_amount", -1.0, 1.0, 0.0001),
        AppParamSpec("filter_env_decay", 10.0, 5000.0, 0.0001, _log_skew(10.0, 5000.0, 300.0)),
        AppParamSpec("lfo_rate", 0.1, 20.0, 0.0001, _log_skew(0.1, 20.0, 2.0)),
        AppParamSpec("lfo_to_pitch", 0.0, 1.0, 0.0001),
        AppParamSpec("lfo_to_cutoff", 0.0, 1.0, 0.0001),
        AppParamSpec("velocity_to_filter", 0.0, 1.0, 0.0001),
        AppParamSpec("distortion_mix", 0.0, 1.0, 0.0001, 1.8),
        AppParamSpec("reverb_mix", 0.0, 1.0, 0.0001),
    )
}

if tuple(APP_PARAM_SPECS) != tuple(PARAM_NAMES):
    raise RuntimeError("APP_PARAM_SPECS must follow the charter order.")


def convert_from_0to1(spec: AppParamSpec, normalized: float) -> float:
    proportion = min(1.0, max(0.0, float(normalized)))
    if spec.skew != 1.0 and proportion > 0.0:
        proportion = math.exp(math.log(proportion) / spec.skew)
    value = spec.start + (spec.end - spec.start) * proportion
    if spec.interval > 0.0:
        value = spec.start + spec.interval * math.floor((value - spec.start) / spec.interval + 0.5)
    if value <= spec.start or spec.end <= spec.start:
        return spec.start
    return min(value, spec.end)


def to_physical(normalized: Sequence[float]) -> np.ndarray:
    """Charter vector (20 normalized values) -> physical values in charter order (float64)."""
    values = list(normalized)
    if len(values) != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} values, got {len(values)}")
    return np.array(
        [convert_from_0to1(APP_PARAM_SPECS[name], v) for name, v in zip(PARAM_NAMES, values)],
        dtype=np.float64,
    )


def to_physical_batch(normalized: np.ndarray) -> np.ndarray:
    return np.stack([to_physical(row) for row in np.asarray(normalized, dtype=np.float64)])


def describe(normalized: Sequence[float]) -> List[str]:
    physical = to_physical(normalized)
    return [f"{name}={value:g}" for name, value in zip(PARAM_NAMES, physical)]
