"""OB-Xf preset (normalized VST parameters) -> Harmonia v3 physical parameters.

Scalings follow OB-Xf v1.0.3 (src/engine/SynthEngine.h, Voice.h, OscillatorBlock.h): cutoff is the pitch
index 120 * v - 45 (+ keyboard tracking around note 53), envelope times use logsc() in ms, oscillator
pitches are 48 * v semitones around 0.5. Features Harmonia does not have (second LFO, oscillator
brightness, slop, xpander modes) are approximated or dropped; the sound matching step corrects the rest.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from src.synth import v3_params as P

MAX_VOICES = 32


def logsc(param: float, lo: float, hi: float, rolloff: float = 19.0) -> float:
    return ((math.exp(param * math.log(rolloff + 1.0)) - 1.0) / rolloff) * (hi - lo) + lo


def _tri(value: float) -> float:
    """OB-Xf LFO target switches: 0 off, 0.5 on, 1 inverted -> 0, 1, -1."""
    if value < 0.25:
        return 0.0
    return 1.0 if value < 0.75 else -1.0


def _wave(saw: float, pulse: float) -> int:
    if saw >= 0.5:
        return 2
    if pulse >= 0.5:
        return 3
    return 1


def obxf_to_v3(p: Dict[str, float]) -> np.ndarray:
    g = lambda key, default=0.0: float(p.get(key, default))  # noqa: E731
    v: Dict[str, float] = {}

    # oscillators and mixer: out = o1 * m1 + o2 * m2 + noise * mn + o1 * o2 * mr
    v["osc_1_waveform"] = _wave(g("Osc1SawWave"), g("Osc1PulseWave"))
    v["osc_2_waveform"] = _wave(g("Osc2SawWave"), g("Osc2PulseWave"))
    m1, m2, mn, mr = g("Osc1Mix", 1.0), g("Osc2Mix", 1.0), g("NoiseMix"), g("RingModMix")
    osc = m1 + m2
    v["osc_mix"] = m2 / osc if osc > 0 else 0.0
    v["ring_mod"] = mr / (osc + mr) if osc + mr > 0 else 0.0
    v["noise_level"] = min(1.0, mn / (osc + mr + mn)) if osc + mr + mn > 0 else 0.0
    v["osc_1_coarse"] = round((g("Osc1Pitch", 0.5) - 0.5) * 48.0)
    v["osc_2_coarse"] = round((g("Osc2Pitch", 0.5) - 0.5) * 48.0)
    v["osc_2_detune"] = 100.0 * logsc(g("Osc2Detune"), 0.001, 0.6) if g("Osc2Detune") > 0 else 0.0
    v["pulse_width"] = 0.5 + 0.5 * 0.95 * g("OscPW")
    v["osc_sync"] = 1.0 if g("OscSync") >= 0.5 else 0.0
    # OB-Xf: osc1 bends osc2's pitch by 48 * v semitones per unit; Harmonia: osc2 phase-modulates osc1
    v["fm_amount"] = min(10.0, g("OscCrossmod") * 6.0)
    if g("Unison") >= 0.5:
        v["unison_voices"] = min(7, 1 + int(g("UnisonVoices") * MAX_VOICES))
        v["unison_detune"] = min(50.0, 50.0 * logsc(g("UnisonDetune", 0.25), 0.001, 1.0))

    # filter (cutoff pitch index -> Hz at middle C; keyboard tracking pivots on note 53)
    keytrack = g("FilterKeyFollow")
    index = 120.0 * g("FilterCutoff", 1.0) - 45.0 + keytrack * (60 - 53)
    v["filter_cutoff"] = 440.0 * 2.0 ** (index / 12.0)
    v["filter_keytrack"] = keytrack
    v["filter_resonance"] = 0.95 * (0.991 - logsc(1.0 - g("FilterResonance"), 0.0, 0.991, 40.0)) / 0.991
    v["filter_slope"] = 1.0 if g("Filter4PoleMode") >= 0.5 else 0.0
    if v["filter_slope"] < 0.5 and g("Filter2PoleBPBlend") >= 0.5:
        v["filter_type"] = 1
    elif v["filter_slope"] < 0.5 and g("FilterMode") > 0.66:
        v["filter_type"] = 2
    invert = -1.0 if g("FilterEnvInvert") >= 0.5 else 1.0
    vel_filter = 1.0 - (1.0 - 100 / 127) * g("VelToFilterEnv")
    v["filter_env_amount"] = invert * 140.0 * g("FilterEnvAmount") / 12.0 * vel_filter
    v["filter_env_attack"] = logsc(g("FilterEnvAttack"), 1.0, 60000.0, 900.0)
    v["filter_env_decay"] = logsc(g("FilterEnvDecay"), 1.0, 60000.0, 900.0)
    v["filter_env_sustain"] = g("FilterEnvSustain")
    v["filter_env_release"] = logsc(g("FilterEnvRelease"), 1.0, 60000.0, 900.0)

    # amplifier
    v["amp_attack"] = logsc(g("AmpEnvAttack"), 4.0, 60000.0, 900.0)
    v["amp_decay"] = logsc(g("AmpEnvDecay"), 4.0, 60000.0, 900.0)
    v["amp_sustain"] = g("AmpEnvSustain", 1.0)
    v["amp_release"] = logsc(g("AmpEnvRelease"), 8.0, 60000.0, 900.0)
    v["velocity_to_amp"] = g("VelToAmpEnv")

    # pitch envelope: OB-Xf sends the filter envelope to the oscillators' pitch (40 * v semitones)
    if g("EnvToPitchAmount") > 0:
        sign = -1.0 if g("EnvToPitchInvert") >= 0.5 else 1.0
        v["pitch_env_amount"] = sign * min(24.0, 40.0 * g("EnvToPitchAmount"))
        v["pitch_env_decay"] = min(5000.0, v["filter_env_attack"] + v["filter_env_decay"])

    # LFO: the first one routed anywhere (Harmonia has one LFO)
    for i in (1, 2):
        amt1 = logsc(logsc(g(f"LFO{i}ModAmount1"), 0.0, 1.0, 60.0), 0.0, 60.0, 10.0)
        amt2 = 0.7 * g(f"LFO{i}ModAmount2")
        pitch = max(abs(_tri(g(f"LFO{i}ToOsc1Pitch"))), abs(_tri(g(f"LFO{i}ToOsc2Pitch")))) * amt1
        cutoff = abs(_tri(g(f"LFO{i}ToFilterCutoff"))) * amt1
        pw = max(abs(_tri(g(f"LFO{i}ToOsc1PW"))), abs(_tri(g(f"LFO{i}ToOsc2PW")))) * amt2
        volume = abs(_tri(g(f"LFO{i}ToVolume"))) * amt2 * 1.4285714285714286
        if pitch + cutoff + pw + volume <= 1e-4:
            continue
        v["lfo_rate"] = logsc(g(f"LFO{i}Rate", 0.5), 0.0, 250.0, 3775.0)
        v["lfo_to_pitch"] = pitch * 100.0
        v["lfo_to_cutoff"] = cutoff / 12.0
        v["lfo_to_pw"] = min(1.0, pw * 0.5 / 0.45)
        v["lfo_to_amp"] = min(1.0, volume)
        waves = [g(f"LFO{i}Wave1"), g(f"LFO{i}Wave2", 0.5), g(f"LFO{i}Wave3", 0.5)]  # 0.5 = off (blend 0)
        strongest = int(np.argmax([abs(w - 0.5) for w in waves]))
        v["lfo_waveform"] = (0, 3, 4)[strongest] if abs(waves[strongest] - 0.5) > 0.05 else 0
        break
    return P.physical_from_dict(v)
