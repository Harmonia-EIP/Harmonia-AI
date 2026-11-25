"""Harmonia v3 analog engine: the parameters, their physical ranges and the 0..1 mapping.

The parameters come from the charter (charter/harmonia_v3.json via src/charter_v3.py), the single source shared
with the plugin; this module keeps the array-oriented API the engine and the AI use. The first 20 keep their v2
names. New parameters were chosen from what real presets use (scripts/v3/analyze_presets.py on OB-Xf and
Surge XT banks). Every new parameter has a neutral default, so a v2 preset (see from_v2_physical) keeps its
sound apart from engine quality fixes: band-limited oscillators and exponential decay/release (v2 ramps
linearly, like juce::ADSR).

Charter 3.1 appends the sample source (sample_instrument, sample_level, sample_tune, sample_start) after the
46 parameters of charter 3.0 (CORE_PARAMS). Their defaults play no sample, so a 46-value vector (a v3.0
preset, or what the v3 / v4 generators produce) completes to the same sound (`complete`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from src import charter_v3


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


SPECS = tuple(
    V3Param(p.id, p.min, p.max, p.default, p.interval, p.centre, len(p.choices) or None, p.unit or "|".join(p.choices))
    for p in charter_v3.load().engine_params("analog")
)

NAMES = tuple(spec.name for spec in SPECS)
INDEX: Dict[str, int] = {name: i for i, name in enumerate(NAMES)}
BY_NAME: Dict[str, V3Param] = {spec.name: spec for spec in SPECS}
DEFAULTS = np.array([spec.default for spec in SPECS], dtype=np.float64)
N_PARAMS = len(SPECS)
SAMPLE_PARAMS = tuple(p.id for p in charter_v3.load().engine_params("analog") if p.module == "sample")
CORE_PARAMS = N_PARAMS - len(SAMPLE_PARAMS)  # the 46 parameters of charter 3.0, first in every vector


_CHARTER = {p.id: p for p in charter_v3.load().engine_params("analog")}


def snap(spec: V3Param, value: float) -> float:
    return _CHARTER[spec.name].snap(value)


def to_normalized(spec: V3Param, value: float) -> float:
    return _CHARTER[spec.name].to_normalized(value)


def from_normalized(spec: V3Param, normalized: float) -> float:
    return _CHARTER[spec.name].from_normalized(normalized)


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
    """A v2 preset (20 physical values, src/synth/params.py order) in the v3 engine (the charter's v2 rules)."""
    values = charter_v3.upgrade_v2(dict(zip(NAMES[:20], list(v2))))
    return physical_from_dict(values)


def complete(physical: Sequence[float]) -> np.ndarray:
    """A physical vector with every parameter: missing trailing ones (a charter 3.0 vector) take their default."""
    p = np.asarray(physical, dtype=np.float64)
    if p.shape[-1] > N_PARAMS:
        raise ValueError(f"expected at most {N_PARAMS} physical values, got {p.shape[-1]}")
    if p.shape[-1] == N_PARAMS:
        return p
    pad = np.broadcast_to(DEFAULTS[p.shape[-1]:], p.shape[:-1] + (N_PARAMS - p.shape[-1],))
    return np.concatenate([p, pad], axis=-1)
