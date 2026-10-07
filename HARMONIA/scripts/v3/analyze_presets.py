#!/usr/bin/env python3
"""Which synthesis features do real presets use? Drives the choice of the v3 engine parameters.

Reads data/v3/presets/*.jsonl (scripts/v3/presets_to_json.py) and reports, per bank, the share of presets
using each feature (a second oscillator, unison, a pitch envelope, chorus...).

    python scripts/v3/analyze_presets.py     # -> data/v3/presets/analysis.json + printed table
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets.dx7 import carriers, feedback_operator  # noqa: E402

PRESET_DIR = BASE_DIR / "data" / "v3" / "presets"
EPS = 1e-3

# Surge XT enums at the pinned commit (src/common/SurgeStorage.h, ModulationSource.h, sst-filters).
SURGE_OSC = ["classic", "sine", "wavetable", "shnoise", "audioinput", "fm3", "fm2", "window", "modern",
             "string", "twist", "alias"]
SURGE_UNISON_OSC = {0, 1, 2, 7, 8, 11}  # oscillator types whose param6 is the unison voice count
SURGE_LFO_SOURCES = set(range(17, 29))  # voice LFO 1-6, scene LFO 1-6
SURGE_EG_SOURCES = {15, 16}  # amp EG, filter EG
SURGE_FILTER_FAMILY = {
    1: "lp", 2: "lp", 3: "lp", 10: "lp", 11: "lp", 12: "lp", 13: "lp", 15: "lp", 16: "lp", 28: "lp",
    4: "hp", 5: "hp", 14: "hp", 17: "hp", 20: "hp", 29: "hp",
    6: "bp", 19: "bp", 22: "bp", 23: "bp", 31: "bp",
    7: "notch", 18: "notch", 21: "notch", 24: "notch", 30: "notch",
    8: "comb", 25: "comb",
}
SURGE_FILTER_24DB = {2, 3, 5, 10, 12, 15, 23, 24}
SURGE_FX_FAMILY = {
    1: "delay", 30: "delay", 2: "reverb", 11: "reverb", 27: "reverb", 31: "reverb",
    9: "chorus", 20: "chorus", 3: "phaser", 12: "flanger", 4: "rotary",
    5: "distortion", 18: "distortion", 25: "distortion", 28: "distortion", 24: "distortion",
    6: "eq", 16: "eq", 8: "eq", 13: "ringmod", 7: "freqshift", 17: "resonator", 21: "comb",
}


def load(name: str) -> List[dict]:
    path = PRESET_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def share(records: Iterable[dict], features: Dict[str, Callable[[dict], bool]]) -> Dict[str, float]:
    records = list(records)
    counts = Counter()
    for rec in records:
        for name, test in features.items():
            if test(rec):
                counts[name] += 1
    return {name: round(counts[name] / max(len(records), 1), 4) for name in features}


# --- OB-Xf (normalized VST parameters, semantics from src/engine/SynthEngine.h and Voice.h) ---

def _obxf_lfo(p: dict, target_keys: List[str], amount: str, lfos=(1, 2)) -> bool:
    return any(p.get(f"LFO{i}{amount}", 0) > EPS and any(p.get(f"LFO{i}{t}", 0) >= 0.25 for t in target_keys)
               for i in lfos)


def _obxf_lfo_used(p: dict, i: int) -> bool:
    return (_obxf_lfo(p, ["ToOsc1Pitch", "ToOsc2Pitch", "ToFilterCutoff"], "ModAmount1", (i,))
            or _obxf_lfo(p, ["ToOsc1PW", "ToOsc2PW", "ToVolume"], "ModAmount2", (i,)))


OBXF_FEATURES: Dict[str, Callable[[dict], bool]] = {
    "two_oscillators": lambda r: r["params"].get("Osc1Mix", 0) > EPS and r["params"].get("Osc2Mix", 0) > EPS,
    "osc_interval": lambda r: abs(r["params"].get("Osc2Pitch", 0.5) - r["params"].get("Osc1Pitch", 0.5)) > 0.02,
    "osc_detune": lambda r: r["params"].get("Osc2Detune", 0) > EPS,
    "pulse_wave": lambda r: r["params"].get("Osc1PulseWave", 0) >= 0.5 or r["params"].get("Osc2PulseWave", 0) >= 0.5,
    "pwm": lambda r: r["params"].get("EnvToPWAmount", 0) > EPS or _obxf_lfo(r["params"], ["ToOsc1PW", "ToOsc2PW"], "ModAmount2"),
    "hard_sync": lambda r: r["params"].get("OscSync", 0) >= 0.5,
    "fm_crossmod": lambda r: r["params"].get("OscCrossmod", 0) > EPS,
    "ring_mod": lambda r: r["params"].get("RingModMix", 0) > EPS,
    "noise": lambda r: r["params"].get("NoiseMix", 0) > EPS,
    "unison": lambda r: r["params"].get("Unison", 0) >= 0.5,
    "filter_24db": lambda r: r["params"].get("Filter4PoleMode", 0) >= 0.5,
    "filter_not_lowpass": lambda r: r["params"].get("Filter4PoleMode", 0) < 0.5 and (r["params"].get("FilterMode", 0) > EPS or r["params"].get("Filter2PoleBPBlend", 0) > EPS),
    "filter_keytrack": lambda r: r["params"].get("FilterKeyFollow", 0) > EPS,
    "filter_env": lambda r: r["params"].get("FilterEnvAmount", 0) > EPS,
    "filter_env_attack": lambda r: r["params"].get("FilterEnvAmount", 0) > EPS and r["params"].get("FilterEnvAttack", 0) > 0.02,
    "filter_env_sustain": lambda r: r["params"].get("FilterEnvAmount", 0) > EPS and r["params"].get("FilterEnvSustain", 0) > 0.02,
    "pitch_env": lambda r: r["params"].get("EnvToPitchAmount", 0) > EPS,
    "lfo_pitch": lambda r: _obxf_lfo(r["params"], ["ToOsc1Pitch", "ToOsc2Pitch"], "ModAmount1"),
    "lfo_filter": lambda r: _obxf_lfo(r["params"], ["ToFilterCutoff"], "ModAmount1"),
    "lfo_amp": lambda r: _obxf_lfo(r["params"], ["ToVolume"], "ModAmount2"),
    "two_lfos": lambda r: _obxf_lfo_used(r["params"], 1) and _obxf_lfo_used(r["params"], 2),
    "glide": lambda r: r["params"].get("Portamento", 0) > EPS,
    "velocity": lambda r: r["params"].get("VelToAmpEnv", 0) > EPS or r["params"].get("VelToFilterEnv", 0) > EPS,
    "stereo_spread": lambda r: any(abs(r["params"].get(f"PanVoice{i}", 0.5) - 0.5) > 0.02 for i in range(1, 9)),
}


# --- Surge XT (physical parameter values from the patch XML) ---

def _surge_scenes(p: dict) -> List[str]:
    return ["a", "b"] if p.get("scenemode", 0) != 0 else ["a"]


def _surge_active_oscs(p: dict, s: str) -> List[int]:
    return [i for i in (1, 2, 3) if p.get(f"{s}_mute_o{i}", 0) == 0 and p.get(f"{s}_level_o{i}", 0) > EPS]


def _surge_any(fn: Callable[[dict, str], bool]) -> Callable[[dict], bool]:
    return lambda r: any(fn(r["params"], s) for s in _surge_scenes(r["params"]))


def _surge_mod(r: dict, sources: set, target_test: Callable[[str], bool]) -> bool:
    scenes = _surge_scenes(r["params"])
    return any(m["source"] in sources and abs(m["depth"]) > EPS and target_test(m["target"])
               and m["target"][:1] in scenes for m in r.get("modulation", []))


def _surge_fx(r: dict) -> set:
    p = r["params"]
    disabled = p.get("fx_disable", 0)
    families = set()
    for key, value in p.items():
        match = re.fullmatch(r"fx(\d+)_type", key)
        if match and value:
            slot = int(match.group(1)) - 1
            if not (disabled >> slot) & 1:
                families.add(SURGE_FX_FAMILY.get(value, "other"))
    return families


def _surge_unison(p: dict, s: str) -> bool:
    return any(p.get(f"{s}_osc{i}_type") in SURGE_UNISON_OSC and p.get(f"{s}_osc{i}_param6", 1) > 1
               for i in _surge_active_oscs(p, s))


def _surge_osc_types(p: dict, s: str) -> set:
    return {SURGE_OSC[p.get(f"{s}_osc{i}_type", 0)] for i in _surge_active_oscs(p, s)}


SURGE_FEATURES: Dict[str, Callable[[dict], bool]] = {
    "two_scenes": lambda r: r["params"].get("scenemode", 0) != 0,
    "two_oscillators": _surge_any(lambda p, s: len(_surge_active_oscs(p, s)) >= 2),
    "three_oscillators": _surge_any(lambda p, s: len(_surge_active_oscs(p, s)) >= 3),
    "osc_classic_or_modern": _surge_any(lambda p, s: bool(_surge_osc_types(p, s) & {"classic", "modern"})),
    "osc_sine": _surge_any(lambda p, s: "sine" in _surge_osc_types(p, s)),
    "osc_wavetable_or_window": _surge_any(lambda p, s: bool(_surge_osc_types(p, s) & {"wavetable", "window"})),
    "osc_fm2_fm3": _surge_any(lambda p, s: bool(_surge_osc_types(p, s) & {"fm2", "fm3"})),
    "osc_string_twist_alias_snh": _surge_any(lambda p, s: bool(_surge_osc_types(p, s) & {"string", "twist", "alias", "shnoise"})),
    "fm_between_oscs": _surge_any(lambda p, s: p.get(f"{s}_fm_switch", 0) != 0),
    "hard_sync": _surge_any(lambda p, s: any(p.get(f"{s}_osc{i}_type") in (0, 8) and p.get(f"{s}_osc{i}_param4", 0) > EPS for i in _surge_active_oscs(p, s))),
    "sub_osc": _surge_any(lambda p, s: any(p.get(f"{s}_osc{i}_type") == 0 and p.get(f"{s}_osc{i}_param3", 0) > EPS for i in _surge_active_oscs(p, s))),
    "ring_mod": _surge_any(lambda p, s: any(p.get(f"{s}_mute_{k}", 1) == 0 and p.get(f"{s}_level_{k}", 0) > EPS for k in ("ring12", "ring23"))),
    "noise": _surge_any(lambda p, s: p.get(f"{s}_mute_noise", 1) == 0 and p.get(f"{s}_level_noise", 0) > EPS),
    "unison": _surge_any(_surge_unison),
    "two_filters": _surge_any(lambda p, s: p.get(f"{s}_filter1_type", 0) != 0 and p.get(f"{s}_filter2_type", 0) != 0),
    "filter_24db": _surge_any(lambda p, s: p.get(f"{s}_filter1_type", 0) in SURGE_FILTER_24DB or p.get(f"{s}_filter2_type", 0) in SURGE_FILTER_24DB),
    "filter_not_lowpass": _surge_any(lambda p, s: any(SURGE_FILTER_FAMILY.get(p.get(f"{s}_filter{i}_type", 0), "lp") != "lp" for i in (1, 2) if p.get(f"{s}_filter{i}_type", 0))),
    "filter_keytrack": _surge_any(lambda p, s: abs(p.get(f"{s}_filter1_keytrack", 0)) > EPS),
    "filter_env": _surge_any(lambda p, s: abs(p.get(f"{s}_filter1_envmod", 0)) > EPS or abs(p.get(f"{s}_filter2_envmod", 0)) > EPS),
    "filter_env_attack": _surge_any(lambda p, s: (abs(p.get(f"{s}_filter1_envmod", 0)) > EPS) and p.get(f"{s}_env2_attack", -8) > -7.9),
    "filter_env_sustain": _surge_any(lambda p, s: (abs(p.get(f"{s}_filter1_envmod", 0)) > EPS) and p.get(f"{s}_env2_sustain", 0) > 0.02),
    "waveshaper": _surge_any(lambda p, s: p.get(f"{s}_ws_type", 0) != 0),
    "pitch_env": lambda r: _surge_mod(r, SURGE_EG_SOURCES, lambda t: t.endswith("pitch")),
    "lfo_pitch": lambda r: _surge_mod(r, SURGE_LFO_SOURCES, lambda t: t.endswith("pitch")),
    "lfo_filter": lambda r: _surge_mod(r, SURGE_LFO_SOURCES, lambda t: "cutoff" in t),
    "lfo_amp": lambda r: _surge_mod(r, SURGE_LFO_SOURCES, lambda t: t.endswith(("volume", "vca_level")) or "_level_" in t),
    "lfo_pan": lambda r: _surge_mod(r, SURGE_LFO_SOURCES, lambda t: "pan" in t),
    "lfo_osc_shape": lambda r: _surge_mod(r, SURGE_LFO_SOURCES, lambda t: "_osc" in t and "_param" in t),
    "velocity": lambda r: _surge_mod(r, {1}, lambda t: True) or any(abs(r["params"].get(f"{s}_vca_velsense", 0)) > EPS for s in _surge_scenes(r["params"])),
    "glide": _surge_any(lambda p, s: p.get(f"{s}_portamento", -8) > -7.9),
    "mono": _surge_any(lambda p, s: p.get(f"{s}_polymode", 0) in (1, 2, 3, 4)),
    "fx_reverb": lambda r: "reverb" in _surge_fx(r),
    "fx_delay": lambda r: "delay" in _surge_fx(r),
    "fx_chorus": lambda r: "chorus" in _surge_fx(r),
    "fx_phaser_flanger": lambda r: bool({"phaser", "flanger"} & _surge_fx(r)),
    "fx_distortion": lambda r: "distortion" in _surge_fx(r),
    "fx_eq": lambda r: "eq" in _surge_fx(r),
    "fx_rotary": lambda r: "rotary" in _surge_fx(r),
    "fx_other": lambda r: bool(_surge_fx(r) - {"reverb", "delay", "chorus", "phaser", "flanger", "distortion", "eq", "rotary"}),
}


# --- DX7 ---

def _dx7_active(op: dict, threshold: int) -> bool:
    return op["output_level"] >= threshold


def dx7_features(threshold: int) -> Dict[str, Callable[[dict], bool]]:
    def active_ops(r):
        return [i + 1 for i, op in enumerate(r["params"]["operators"]) if _dx7_active(op, threshold)]

    def modulators(r):
        car = set(carriers(r["params"]["algorithm"]))
        return [i for i in active_ops(r) if i not in car]

    return {
        "active_ops_le_2": lambda r: len(active_ops(r)) <= 2,
        "active_ops_le_4": lambda r: len(active_ops(r)) <= 4,
        "modulators_le_1": lambda r: len(modulators(r)) <= 1,
        "modulators_le_2": lambda r: len(modulators(r)) <= 2,
        "feedback": lambda r: r["params"]["feedback"] > 0 and feedback_operator(r["params"]["algorithm"]) in active_ops(r),
        "fixed_frequency_op": lambda r: any(r["params"]["operators"][i - 1]["osc_mode"] == 1 for i in active_ops(r)),
        "non_integer_ratio": lambda r: any(r["params"]["operators"][i - 1]["freq_fine"] > 0 or r["params"]["operators"][i - 1]["freq_coarse"] == 0 for i in active_ops(r)),
        "pitch_env": lambda r: any(level != 50 for level in r["params"]["pitch_eg_levels"]),
        "lfo_pitch": lambda r: r["params"]["lfo_pitch_mod_depth"] > 0 and r["params"]["pitch_mod_sens"] > 0,
        "lfo_amp": lambda r: r["params"]["lfo_amp_mod_depth"] > 0 and any(r["params"]["operators"][i - 1]["amp_mod_sens"] > 0 for i in active_ops(r)),
        "velocity": lambda r: any(r["params"]["operators"][i - 1]["key_vel_sens"] > 0 for i in active_ops(r)),
        "op_detune": lambda r: any(r["params"]["operators"][i - 1]["detune"] != 7 for i in active_ops(r)),
    }


def main() -> int:
    report: Dict[str, Dict[str, object]] = {}
    obxf, surge, dx7 = load("obxf"), load("surge"), load("dx7")
    report["obxf"] = {"count": len(obxf), "features": share(obxf, OBXF_FEATURES)}
    report["surge"] = {"count": len(surge), "features": share(surge, SURGE_FEATURES)}
    report["dx7"] = {"count": len(dx7), "features_level_gt_0": share(dx7, dx7_features(1)),
                     "features_level_ge_50": share(dx7, dx7_features(50)),
                     "algorithms": Counter(r["params"]["algorithm"] + 1 for r in dx7).most_common(10)}
    (PRESET_DIR / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for bank, data in report.items():
        print(f"\n== {bank} ({data['count']} presets)")
        for key in [k for k in data if k.startswith("features")]:
            print(f"  [{key}]")
            for name, value in data[key].items():
                print(f"    {name:32s} {value:6.1%}")
        if "algorithms" in data:
            print("  algorithms (top 10):", data["algorithms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
