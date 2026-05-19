from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CONTINUOUS = "continuous"
DISCRETE = "discrete"
BIPOLAR = "bipolar"

LINEAR = "linear"
LOGARITHMIC = "logarithmic"
EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class CharterParam:
    pid: str
    name: str
    section: str
    kind: str
    mapping: str
    range_min: float
    range_max: float
    unit: str
    role: str
    steps: Optional[Tuple[str, ...]] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.pid,
            "name": self.name,
            "section": self.section,
            "kind": self.kind,
            "mapping": self.mapping,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "unit": self.unit,
            "role": self.role,
        }
        if self.steps:
            payload["steps"] = list(self.steps)
        return payload


WAVEFORM_STEPS: Tuple[str, ...] = ("Sine", "Triangle", "Saw", "Square")
FILTER_TYPE_STEPS: Tuple[str, ...] = ("Lowpass", "Bandpass", "Highpass")


CHARTER: Tuple[CharterParam, ...] = (
    CharterParam("P1", "osc_1_waveform", "Generators", DISCRETE, LINEAR, 0.0, 1.0, "step", "Primary oscillator waveform.", WAVEFORM_STEPS),
    CharterParam("P2", "osc_2_waveform", "Generators", DISCRETE, LINEAR, 0.0, 1.0, "step", "Secondary oscillator waveform.", WAVEFORM_STEPS),
    CharterParam("P3", "osc_mix", "Generators", CONTINUOUS, LINEAR, 0.0, 1.0, "ratio", "Volume balance Osc1 <-> Osc2."),
    CharterParam("P4", "osc_2_detune", "Generators", CONTINUOUS, LINEAR, 0.0, 100.0, "cents", "Detune Osc2 (chorus / beating)."),
    CharterParam("P5", "noise_level", "Generators", CONTINUOUS, LINEAR, 0.0, 1.0, "gain", "Noise generator level (breath / impact)."),
    CharterParam("P6", "filter_cutoff", "Filter", CONTINUOUS, LOGARITHMIC, 20.0, 20000.0, "Hz", "Brightness."),
    CharterParam("P7", "filter_resonance", "Filter", CONTINUOUS, LINEAR, 0.0, 0.95, "ratio", "Self-oscillation amount."),
    CharterParam("P8", "filter_type", "Filter", DISCRETE, LINEAR, 0.0, 1.0, "step", "Filter topology.", FILTER_TYPE_STEPS),
    CharterParam("P9", "amp_attack", "Envelopes", CONTINUOUS, LOGARITHMIC, 1.0, 5000.0, "ms", "Amp envelope attack time."),
    CharterParam("P10", "amp_decay", "Envelopes", CONTINUOUS, LOGARITHMIC, 10.0, 5000.0, "ms", "Amp envelope decay time."),
    CharterParam("P11", "amp_sustain", "Envelopes", CONTINUOUS, LINEAR, 0.0, 1.0, "gain", "Amp envelope sustain level."),
    CharterParam("P12", "amp_release", "Envelopes", CONTINUOUS, LOGARITHMIC, 10.0, 10000.0, "ms", "Amp envelope release time."),
    CharterParam("P13", "filter_env_amount", "Envelopes", BIPOLAR, LINEAR, -1.0, 1.0, "ratio", "Filter envelope intensity (bipolar; 0.5 = neutral)."),
    CharterParam("P14", "filter_env_decay", "Envelopes", CONTINUOUS, LOGARITHMIC, 10.0, 5000.0, "ms", "Filter envelope decay time."),
    CharterParam("P15", "lfo_rate", "Modulation", CONTINUOUS, LOGARITHMIC, 0.1, 20.0, "Hz", "LFO rate."),
    CharterParam("P16", "lfo_to_pitch", "Modulation", CONTINUOUS, LINEAR, 0.0, 1.0, "depth", "Vibrato depth."),
    CharterParam("P17", "lfo_to_cutoff", "Modulation", CONTINUOUS, LINEAR, 0.0, 1.0, "depth", "Cutoff modulation depth."),
    CharterParam("P18", "velocity_to_filter", "Modulation", CONTINUOUS, LINEAR, 0.0, 1.0, "depth", "Velocity sensitivity on filter (essential for piano)."),
    CharterParam("P19", "distortion_mix", "Effects", CONTINUOUS, EXPONENTIAL, 0.0, 1.0, "mix", "Distortion / drive mix."),
    CharterParam("P20", "reverb_mix", "Effects", CONTINUOUS, LINEAR, 0.0, 1.0, "mix", "Reverb dry/wet."),
)

if len(CHARTER) != 20:
    raise RuntimeError("Charter must contain exactly 20 parameters.")

PARAM_NAMES: Tuple[str, ...] = tuple(p.name for p in CHARTER)
PARAM_INDEX: Dict[str, int] = {p.name: idx for idx, p in enumerate(CHARTER)}
PARAM_BY_ID: Dict[str, CharterParam] = {p.pid: p for p in CHARTER}

CONTINUOUS_INDICES: Tuple[int, ...] = tuple(idx for idx, p in enumerate(CHARTER) if p.kind == CONTINUOUS)
BIPOLAR_INDICES: Tuple[int, ...] = tuple(idx for idx, p in enumerate(CHARTER) if p.kind == BIPOLAR)
DISCRETE_INDICES: Tuple[int, ...] = tuple(idx for idx, p in enumerate(CHARTER) if p.kind == DISCRETE)
DISCRETE_STEPS: Dict[int, Tuple[str, ...]] = {
    idx: CHARTER[idx].steps  # type: ignore[assignment]
    for idx in DISCRETE_INDICES
    if CHARTER[idx].steps is not None
}


def charter_metadata() -> List[Dict[str, object]]:
    return [param.to_dict() for param in CHARTER]


def discrete_step_count(idx: int) -> int:
    steps = DISCRETE_STEPS.get(idx)
    return len(steps) if steps else 0


def snap_discrete(value: float, idx: int) -> float:
    n = discrete_step_count(idx)
    if n <= 1:
        return max(0.0, min(1.0, float(value)))
    clamped = max(0.0, min(1.0, float(value)))
    bucket = round(clamped * (n - 1))
    return bucket / (n - 1)


def clamp_unit(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalise_vector(values: List[float]) -> List[float]:
    out = [clamp_unit(v) for v in values[:20]]
    if len(out) < 20:
        out.extend([0.0] * (20 - len(out)))
    for idx in DISCRETE_INDICES:
        out[idx] = snap_discrete(out[idx], idx)
    return out
