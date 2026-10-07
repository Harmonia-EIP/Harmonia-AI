"""Surge XT patch (physical parameter values from its XML) -> Harmonia v3 physical parameters.

Surge is far richer than Harmonia (3 oscillators x 12 types, 2 filters, 2 scenes, wavetables, a modulation
matrix, 16 effect slots), so this only gives the sound matching step a sensible start: scene A, its two
loudest oscillators, filter 1 (or 2 when 1 is off), the amp/filter envelopes (env1 = amp, env2 = filter),
voice LFO 1 routings and which effects are present. Units at the pinned Surge commit: cutoff in semitones
around 440 Hz, envelope times as log2(seconds), LFO rate as log2(Hz), modulation depths in the target's units.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from src.presets.analyze_tables import SURGE_FILTER_FAMILY, SURGE_FILTER_24DB, SURGE_FX_FAMILY
from src.synth import v3_params as P

FX_SLOTS = 16


def _seconds(log2_s: float) -> float:
    return 2.0 ** log2_s


def _osc_wave(osc_type: int, p: Dict[str, float], prefix: str) -> int:
    if osc_type == 1:  # sine
        return 0
    if osc_type == 0:  # classic: shape -1 (triangle-like) .. 0 (saw) .. 1 (square)
        shape = p.get(f"{prefix}_param0", 0.0)
        return 3 if shape >= 0.5 else (1 if shape <= -0.5 else 2)
    if osc_type == 8:  # modern: saw / pulse / triangle levels
        levels = [abs(p.get(f"{prefix}_param{i}", 0.0)) for i in range(3)]
        return (2, 3, 1)[int(np.argmax(levels))]
    if osc_type in (5, 6):  # FM3 / FM2: a sine carrier
        return 0
    return 2  # wavetable, window, string, twist, alias, S&H noise: brightest stand-in


def surge_to_v3(p: Dict[str, float], modulation: List[dict]) -> np.ndarray:
    s = "a"
    v: Dict[str, float] = {}

    # oscillators: the two loudest unmuted ones
    active = []
    for i in (1, 2, 3):
        level = p.get(f"{s}_level_o{i}", 0.0)
        if p.get(f"{s}_mute_o{i}", 0) == 0 and level > 1e-3:
            active.append((level, i))
    active.sort(reverse=True)
    picked = [i for _, i in active[:2]] or [1]
    fm_on = p.get(f"{s}_fm_switch", 0) and len(picked) == 2
    if fm_on and picked[0] < picked[1]:
        # Surge FM 2>1: the higher oscillator modulates the lower one; Harmonia's osc1 is the modulator
        picked = [picked[1], picked[0]]
    scene_semis = 12.0 * p.get(f"{s}_octave", 0) + p.get(f"{s}_pitch", 0.0)  # scene octave, used by 31 %
    levels = []
    for slot, i in enumerate(picked):
        prefix = f"{s}_osc{i}"
        osc_type = int(p.get(f"{prefix}_type", 0))
        semis = scene_semis + 12.0 * p.get(f"{prefix}_octave", 0) + p.get(f"{prefix}_pitch", 0.0)
        v[f"osc_{slot + 1}_waveform"] = _osc_wave(osc_type, p, prefix)
        v[f"osc_{slot + 1}_coarse"] = round(min(24.0, max(-24.0, semis)))
        levels.append(p.get(f"{s}_level_o{i}", 1.0))
        if slot == 0:
            if osc_type == 0:
                v["pulse_width"] = min(0.95, max(0.05, p.get(f"{prefix}_param1", 0.5)))
                v["osc_sync"] = 1.0 if p.get(f"{prefix}_param4", 0.0) > 0.5 else 0.0
            if osc_type in (0, 1, 2, 7, 8, 11) and p.get(f"{prefix}_param6", 1) > 1:
                v["unison_voices"] = min(16, int(p.get(f"{prefix}_param6", 1)))
                v["unison_detune"] = min(50.0, abs(p.get(f"{prefix}_param5", 0.1)) * 100.0)
                v["stereo_width"] = 0.8  # Surge spreads unison voices across the stereo field
    if len(picked) == 2:
        v["osc_mix"] = levels[1] / max(levels[0] + levels[1], 1e-6)
        v["osc_2_detune"] = 0.0
    if fm_on:
        v["fm_amount"] = min(48.0, 12.0 * 2.0 ** (p.get(f"{s}_fm_depth", -24.0) / 12.0))
        v["osc_mix"] = 0.8  # mostly the carrier
    if p.get(f"{s}_mute_noise", 1) == 0:
        v["noise_level"] = min(1.0, 0.5 * p.get(f"{s}_level_noise", 0.0))
    if p.get(f"{s}_mute_ring12", 1) == 0:
        v["ring_mod"] = min(1.0, 0.5 * p.get(f"{s}_level_ring12", 0.0))

    # filter 1, or filter 2 when filter 1 is off
    f = 1 if p.get(f"{s}_filter1_type", 0) else (2 if p.get(f"{s}_filter2_type", 0) else 0)
    if f:
        ftype = int(p.get(f"{s}_filter{f}_type", 1))
        family = SURGE_FILTER_FAMILY.get(ftype, "lp")
        v["filter_type"] = {"lp": 0, "bp": 1, "hp": 2}.get(family, 0)
        v["filter_slope"] = 1.0 if ftype in SURGE_FILTER_24DB else 0.0
        v["filter_cutoff"] = 440.0 * 2.0 ** (p.get(f"{s}_filter{f}_cutoff", 70.0) / 12.0)
        v["filter_resonance"] = 0.95 * min(1.0, max(0.0, p.get(f"{s}_filter{f}_resonance", 0.0)))
        v["filter_env_amount"] = p.get(f"{s}_filter{f}_envmod", 0.0) / 12.0
        v["filter_keytrack"] = min(1.0, max(0.0, p.get(f"{s}_filter{f}_keytrack", 0.0)))

    # envelopes: env1 = amp, env2 = filter
    v["amp_attack"] = 1000.0 * _seconds(p.get(f"{s}_env1_attack", -8.0))
    v["amp_decay"] = 1000.0 * _seconds(p.get(f"{s}_env1_decay", -2.0))
    v["amp_sustain"] = p.get(f"{s}_env1_sustain", 1.0)
    v["amp_release"] = 1000.0 * _seconds(p.get(f"{s}_env1_release", -2.0))
    v["filter_env_attack"] = 1000.0 * _seconds(p.get(f"{s}_env2_attack", -8.0))
    v["filter_env_decay"] = 1000.0 * _seconds(p.get(f"{s}_env2_decay", -2.0))
    v["filter_env_sustain"] = p.get(f"{s}_env2_sustain", 0.0)
    v["filter_env_release"] = 1000.0 * _seconds(p.get(f"{s}_env2_release", -2.0))
    v["velocity_to_amp"] = min(1.0, abs(p.get(f"{s}_vca_velsense", 0.0)) / 36.0)

    # voice LFO 1 (modulation source 17) and EG -> pitch routings
    for m in modulation:
        target, depth = m["target"], abs(m["depth"])
        if not target.startswith(f"{s}_"):
            continue
        if m["source"] == 17:
            if target.endswith("pitch"):
                v["lfo_to_pitch"] = max(v.get("lfo_to_pitch", 0.0), min(1200.0, depth * 100.0))
            elif "cutoff" in target:
                v["lfo_to_cutoff"] = max(v.get("lfo_to_cutoff", 0.0), min(4.0, depth / 12.0))
            elif target.endswith(("volume", "vca_level")):
                v["lfo_to_amp"] = min(1.0, depth)
        elif m["source"] in (15, 16) and target.endswith("pitch"):
            v["pitch_env_amount"] = math.copysign(min(24.0, depth), m["depth"])
            v["pitch_env_decay"] = min(5000.0, 1000.0 * _seconds(p.get(f"{s}_env{m['source'] - 14}_decay", -2.0)))
    if any(k in v for k in ("lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp")):
        v["lfo_rate"] = _seconds(p.get(f"{s}_lfo0_rate", 0.0))
        shape = int(p.get(f"{s}_lfo0_shape", 0))
        v["lfo_waveform"] = {0: 0, 1: 1, 2: 3, 3: 2, 5: 4}.get(shape, 0)
        v["lfo_delay"] = 1000.0 * _seconds(p.get(f"{s}_lfo0_delay", -8.0)) if p.get(f"{s}_lfo0_delay", -8.0) > -7.9 else 0.0

    # effects: only whether a family is present; amounts are left to sound matching
    disabled = int(p.get("fx_disable", 0))
    families = {SURGE_FX_FAMILY.get(int(p.get(f"fx{k}_type", 0)), "other")
                for k in range(1, FX_SLOTS + 1) if p.get(f"fx{k}_type", 0) and not (disabled >> (k - 1)) & 1}
    if "reverb" in families:
        v["reverb_mix"], v["reverb_size"] = 0.3, 0.8
    if "delay" in families:
        v["delay_mix"] = 0.25
    if "chorus" in families:
        v["chorus_mix"] = 0.6
    if "distortion" in families or p.get(f"{s}_ws_type", 0):
        v["distortion_mix"] = 0.3
    return P.physical_from_dict(v)
