#!/usr/bin/env python3
"""Inspect an unknown dataset and scaffold a HARMONIA profile.

Given a new dataset (NPY or JSON) whose schema you don't know, this script:

1. Flattens the parameter dicts and lists every native key it sees.
2. Suggests, for each of the 20 charter parameters, the most likely source
   key via fuzzy/keyword matching against an internal vocabulary.
3. Writes a profile JSON skeleton to ``src/profiles/<name>.json`` that is
   80-90% filled in. You fix the rest by hand, then run::

       python scripts/charterize_dataset.py --input <dataset> \
           --profile <name> --output data/processed/<name>.charter.npy

Heuristics use case-insensitive substring matching on a curated keyword set
per charter param, so it works regardless of whether the source dataset
follows Sylenth1, Serum, Vital, Massive, Diva, or a custom convention.

Usage examples::

    python scripts/inspect_dataset.py --input data/new.npy
    python scripts/inspect_dataset.py --input data/new.npy --suggest-profile myserum
    python scripts/inspect_dataset.py --input data/new.npy --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.charterize_dataset import _flatten_sylenth_parameters
from src.charter import (
    BIPOLAR_INDICES,
    DISCRETE_INDICES,
    PARAM_NAMES,
)
from src.dataset_profiles import PROFILES_DIR


# Keyword hints per charter parameter. The scoring function rewards a candidate
# source key when it contains one (or more) of these substrings.
CHARTER_HINTS: Dict[str, List[str]] = {
    "osc_1_waveform":     ["osc 1 wave", "osc1 wave", "osc a1 wave", "wave 1", "wt1", "shape 1"],
    "osc_2_waveform":     ["osc 2 wave", "osc2 wave", "osc a2 wave", "wave 2", "wt2", "shape 2"],
    "osc_mix":            ["osc mix", "mix", "osc 2 vol", "osc2 volume", "osc balance"],
    "osc_2_detune":       ["osc 2 detune", "osc2 detune", "detune 2", "fine 2"],
    "noise_level":        ["noise", "noise level", "noise amp"],
    "filter_cutoff":      ["cutoff", "filter freq", "filter cutoff", "fc"],
    "filter_resonance":   ["resonance", "reso", "q"],
    "filter_type":        ["filter type", "filter mode", "filter shape"],
    "amp_attack":         ["amp attack", "ampenv attack", "ampenv a attack", "env amp attack", "vca attack"],
    "amp_decay":          ["amp decay", "ampenv decay", "ampenv a decay", "vca decay"],
    "amp_sustain":        ["amp sustain", "ampenv sustain", "ampenv a sustain"],
    "amp_release":        ["amp release", "ampenv release", "ampenv a release", "vca release"],
    "filter_env_amount":  ["filter env amt", "fenv amount", "modenv amount", "env to filter", "env > filter", "modenv dest", "xmodenv"],
    "filter_env_decay":   ["filter env decay", "modenv decay", "fenv decay", "modenv 1 decay"],
    "lfo_rate":           ["lfo rate", "lfo 1 rate", "lfo speed"],
    "lfo_to_pitch":       ["lfo to pitch", "vibrato", "lfo pitch", "lfo > pitch"],
    "lfo_to_cutoff":      ["lfo to cutoff", "lfo to filter", "lfo cutoff", "lfo gain"],
    "velocity_to_filter": ["velocity to filter", "vel filter", "key track", "keytrk", "key trk"],
    "distortion_mix":     ["distort mix", "distort drywet", "distortion mix", "drive mix", "saturation"],
    "reverb_mix":         ["reverb mix", "reverb dry/wet", "reverb wet", "reverb drywet"],
}

# Optional companion fields the scaffold should propose alongside the main key.
COMPANION_HINTS: Dict[str, Dict[str, List[str]]] = {
    "distortion_mix": {"gate": ["distort on", "sw dist", "distortion on", "dist enable"]},
    "reverb_mix":     {"gate": ["reverb on", "sw reverb", "reverb enable"]},
    "noise_level":    {"alt": ["osc 3 vol", "osc b1 volume", "osc b2 volume", "sub osc"]},
}


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        payload = loaded.tolist() if isinstance(loaded, np.ndarray) else loaded
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        yield payload
    else:
        raise ValueError(f"Unsupported content in {path}")


def collect_keys(path: Path, sample_limit: int = 5000) -> Counter:
    counts: Counter = Counter()
    for idx, item in enumerate(_iter_records(path)):
        if idx >= sample_limit:
            break
        if not isinstance(item, dict):
            continue
        flat = _flatten_sylenth_parameters(item.get("parameters", item))
        counts.update(flat.keys())
    return counts


def _tokens(text: str) -> List[str]:
    """Lower-case word tokens, ignoring punctuation."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return [t for t in cleaned.split() if t]


def _score(candidate: str, hint: str) -> float:
    """Combine substring + word-overlap + SequenceMatcher similarity."""
    lc = candidate.lower()
    lh = hint.lower()
    substring = 1.0 if lh in lc else 0.0

    candidate_tokens = set(_tokens(candidate))
    hint_tokens = set(_tokens(hint))
    overlap = len(candidate_tokens & hint_tokens) / max(1, len(hint_tokens))

    ratio = SequenceMatcher(None, lc, lh).ratio()
    return substring + 0.6 * overlap + 0.3 * ratio


def suggest_for_param(charter_name: str, available_keys: Iterable[str], top: int) -> List[Tuple[str, float]]:
    hints = CHARTER_HINTS.get(charter_name, [charter_name.replace("_", " ")])
    scores: Dict[str, float] = {}
    for key in available_keys:
        best = 0.0
        for hint in hints:
            best = max(best, _score(key, hint))
        if best > 0.55:
            scores[key] = best
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top]


def suggest_companion(charter_name: str, role: str, available_keys: Iterable[str], top: int) -> List[Tuple[str, float]]:
    pool = COMPANION_HINTS.get(charter_name, {})
    hints = pool.get(role) or []
    if not hints:
        return []
    scores: Dict[str, float] = {}
    for key in available_keys:
        best = 0.0
        for hint in hints:
            best = max(best, _score(key, hint))
        if best > 0.55:
            scores[key] = best
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top]


def build_profile_skeleton(name: str, key_counts: Counter, top: int = 3) -> Dict[str, Any]:
    available = list(key_counts.keys())
    params: Dict[str, Any] = {}
    suggestions: Dict[str, List[Tuple[str, float]]] = {}

    for idx, charter_name in enumerate(PARAM_NAMES):
        ranked = suggest_for_param(charter_name, available, top=top)
        suggestions[charter_name] = ranked
        first = ranked[0][0] if ranked else None

        if charter_name == "distortion_mix":
            gate_ranked = suggest_companion("distortion_mix", "gate", available, top=1)
            if first and gate_ranked:
                params[charter_name] = {
                    "strategy": "gated",
                    "value": first,
                    "gate": gate_ranked[0][0],
                    "default": 0.0,
                }
                continue
        if charter_name == "reverb_mix":
            gate_ranked = suggest_companion("reverb_mix", "gate", available, top=1)
            if first and gate_ranked:
                params[charter_name] = {
                    "strategy": "gated",
                    "value": first,
                    "gate": gate_ranked[0][0],
                    "default": 0.0,
                }
                continue
        if charter_name == "noise_level":
            alt = suggest_companion("noise_level", "alt", available, top=3)
            keys = list(dict.fromkeys([k for k, _ in alt] + ([first] if first else [])))
            if keys:
                params[charter_name] = {"strategy": "max", "keys": keys, "default": 0.0}
                continue
        if charter_name == "filter_env_amount":
            if first:
                params[charter_name] = {"strategy": "bipolar_amount", "key": first, "default": 0.5}
                continue
            params[charter_name] = {"strategy": "constant", "value": 0.5}
            continue
        if charter_name == "lfo_to_pitch":
            if first:
                params[charter_name] = {"strategy": "direct", "key": first, "default": 0.0}
                continue
            params[charter_name] = {"strategy": "constant", "value": 0.0}
            continue

        if first is None:
            default = 0.5 if idx in BIPOLAR_INDICES else 0.0
            params[charter_name] = {"strategy": "constant", "value": default, "_todo": "no candidate found"}
            continue

        params[charter_name] = {"strategy": "direct", "key": first, "default": 0.0}

    profile = {
        "name": name,
        "description": f"Auto-scaffolded profile from {len(key_counts)} unique keys (review TODOs before training).",
        "version": "0.1-draft",
        "detect": {"required_keys": []},
        "params": params,
        "_suggestions": {
            charter_name: [{"key": k, "score": round(score, 3)} for k, score in ranked]
            for charter_name, ranked in suggestions.items()
        },
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a dataset and scaffold a HARMONIA profile.")
    parser.add_argument("--input", type=Path, required=True, help="Source dataset (NPY or JSON).")
    parser.add_argument("--top", type=int, default=3, help="How many candidates to list per charter param.")
    parser.add_argument("--sample-limit", type=int, default=5000, help="How many input records to scan for key discovery.")
    parser.add_argument("--suggest-profile", default=None, help="If set, write src/profiles/<name>.json with the scaffold.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing profile file.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 2

    print(f"Scanning {args.input} ...")
    key_counts = collect_keys(args.input, sample_limit=args.sample_limit)
    if not key_counts:
        print("No parameter keys discovered. Are records structured correctly?", file=sys.stderr)
        return 1

    print(f"Discovered {len(key_counts)} unique parameter keys "
          f"across {min(sum(key_counts.values()), args.sample_limit)} samples.\n")

    print("=== Charter parameter suggestions ===")
    available = list(key_counts.keys())
    for charter_name in PARAM_NAMES:
        ranked = suggest_for_param(charter_name, available, top=args.top)
        print(f"  {charter_name}:")
        if not ranked:
            print("    (no candidate found — needs manual mapping)")
            continue
        for k, score in ranked:
            print(f"    {score:>5.2f}  {k}")

    if args.suggest_profile:
        target = PROFILES_DIR / f"{args.suggest_profile}.json"
        if target.exists() and not args.overwrite:
            print(f"\nRefusing to overwrite existing {target} (pass --overwrite).", file=sys.stderr)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        skeleton = build_profile_skeleton(args.suggest_profile, key_counts, top=args.top)
        target.write_text(json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nProfile scaffold written to {target}")
        print("Next steps:")
        print(f"  1. Open the file and review every entry under 'params' (drop _todo notes).")
        print(f"  2. Remove the '_suggestions' block once you're done.")
        print(f"  3. Run: python scripts/charterize_dataset.py --input {args.input} --profile {args.suggest_profile} --output data/processed/{args.suggest_profile}.charter.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
