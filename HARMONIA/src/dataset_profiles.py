"""Pluggable dataset → charter profiles.

A profile is a JSON document that describes how the keys of one source
dataset (Sylenth1, Serum, Vital, a custom dump, ...) map to the 20 HARMONIA
charter parameters.

Profile schema:
    {
        "name": "sylenth1",
        "description": "Sylenth1 .fxp dump",
        "version": "1.0",
        "detect": {
            "required_keys": ["AmpEnv A Attack", "Filter A Cutoff"],
            "any_keys":      []
        },
        "params": {
            "<charter_param_name>": { "strategy": "direct" | "gated" | "max"
                                                   | "bipolar_amount"
                                                   | "constant",
                                       ...strategy-specific fields... }
        }
    }

Strategies:
    direct        { "key": "<src_key>", "default": 0.0 }
    gated         { "value": "<src_key>", "gate": "<src_key>",
                    "default": 0.0, "gate_threshold": 0.5 }
    max           { "keys": ["<src_key>", ...], "default": 0.0 }
    bipolar_amount{ "key": "<src_key>", "default": 0.5 }      # 0.5 = neutral
    routed_amount { "amount_key": "<src_key>", "dest_key": "<src_key>",
                    "expected_dest": <float>, "tolerance": 0.05,
                    "default": 0.0 }
    constant      { "value": <float> }

Every value is clamped to ``[0, 1]``; discrete params are snapped to step
centres declared by the charter (see ``src.charter``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.charter import (
    BIPOLAR_INDICES,
    DISCRETE_INDICES,
    PARAM_NAMES,
    clamp_unit,
    snap_discrete,
)


PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def list_available_profiles() -> List[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(name_or_path: str) -> Dict[str, Any]:
    """Load a profile by name (``src/profiles/<name>.json``) or absolute path."""
    candidate = Path(name_or_path)
    if not candidate.exists():
        if PROFILES_DIR.exists():
            candidate = PROFILES_DIR / f"{name_or_path}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Profile not found: {name_or_path}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Profile {candidate} is not a JSON object")
    _validate_profile(payload, source=str(candidate))
    return payload


def _validate_profile(profile: Dict[str, Any], source: str = "<profile>") -> None:
    if "params" not in profile or not isinstance(profile["params"], dict):
        raise ValueError(f"{source}: profile must contain a 'params' object")
    missing = [p for p in PARAM_NAMES if p not in profile["params"]]
    if missing:
        raise ValueError(
            f"{source}: profile is missing strategies for {len(missing)} charter params "
            f"({', '.join(missing[:6])}{'...' if len(missing) > 6 else ''})"
        )


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_strategy(spec: Mapping[str, Any], record: Mapping[str, Any]) -> Optional[float]:
    strategy = spec.get("strategy", "direct")

    if strategy == "constant":
        return _coerce_float(spec.get("value"))

    if strategy == "direct":
        return _coerce_float(record.get(spec.get("key")))

    if strategy == "max":
        keys = spec.get("keys") or []
        values = [_coerce_float(record.get(k)) for k in keys]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    if strategy == "gated":
        gate = _coerce_float(record.get(spec.get("gate")))
        threshold = float(spec.get("gate_threshold", 0.5))
        if gate is not None and gate < threshold:
            return 0.0
        return _coerce_float(record.get(spec.get("value")))

    if strategy == "bipolar_amount":
        return _coerce_float(record.get(spec.get("key")))

    if strategy == "routed_amount":
        dest = _coerce_float(record.get(spec.get("dest_key")))
        expected = float(spec.get("expected_dest", 0.0))
        tolerance = float(spec.get("tolerance", 0.05))
        if dest is None or abs(dest - expected) > tolerance:
            return 0.0
        amount = _coerce_float(record.get(spec.get("amount_key"))) or 0.5
        return abs(amount * 2.0 - 1.0)

    raise ValueError(f"Unknown profile strategy: {strategy!r}")


def apply_profile(profile: Mapping[str, Any], flat_record: Mapping[str, Any]) -> List[float]:
    """Map a flat ``{src_key: float}`` record to the 20-element charter vector."""
    params_spec = profile["params"]
    vector: List[float] = []
    for idx, name in enumerate(PARAM_NAMES):
        spec = params_spec.get(name) or {"strategy": "constant", "value": 0.0}
        value = _apply_strategy(spec, flat_record)
        if value is None:
            value = spec.get("default")
            if value is None:
                value = 0.5 if idx in BIPOLAR_INDICES else 0.0
        value = clamp_unit(value)
        if idx in DISCRETE_INDICES:
            value = snap_discrete(value, idx)
        vector.append(value)
    return vector


def profile_matches(profile: Mapping[str, Any], record_keys: Iterable[str]) -> Tuple[bool, str]:
    """Cheap heuristic: returns ``(matches, reason)`` for autodetect."""
    detect = profile.get("detect") or {}
    keys = set(record_keys)
    required = detect.get("required_keys") or []
    if required:
        missing = [k for k in required if k not in keys]
        if missing:
            return False, f"missing required keys: {missing[:3]}"
    any_keys = detect.get("any_keys") or []
    if any_keys and not (set(any_keys) & keys):
        return False, "no any_keys present"
    return True, "match"


def autodetect_profile(record_keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    keys = list(record_keys)
    for name in list_available_profiles():
        profile = load_profile(name)
        ok, _ = profile_matches(profile, keys)
        if ok:
            return profile
    return None
