"""Build a charter-compliant dataset from raw Sylenth1 dumps.

Output schema (per record):
    {
        "raw_text": str,
        "description": str,
        "parameters": {
            "continuous": {<charter_name>: float in [0,1], ...},
            "binary": {},
            "categorical": {<charter_name>: float in [0,1], ...}
        }
    }

The 20 parameters follow src.charter.CHARTER (P1..P20).
The C++ side handles the curve mapping; the AI only sees normalised [0,1] values.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import (  # noqa: E402  (needs sys.path)
    BIPOLAR_INDICES,
    CHARTER,
    CONTINUOUS_INDICES,
    DISCRETE_INDICES,
    DISCRETE_STEPS,
    PARAM_NAMES,
    clamp_unit,
    snap_discrete,
)
from src.dashboard_events import publish_command  # noqa: E402


DEFAULT_INPUT_PATH = BASE_DIR / "data" / "raw" / "my_raw_dump.txt"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "presets.json"
ANCHOR_PRESETS_PATH = BASE_DIR / "data" / "raw" / "anchor_presets.json"


# Legacy alias kept so existing tests/imports do not break. New code should
# import PARAM_NAMES from src.charter instead.
TARGET_PARAMS: Tuple[str, ...] = PARAM_NAMES


# Mapping from Sylenth1 dump keys to charter parameter id.
# The C++ engine only ever consumes the charter; this mapping is the
# bridge for legacy data.
SYLENTH_KEY_MAP: Dict[str, str] = {
    "Osc A1 Waveform": "osc_1_waveform",
    "Osc A2 Waveform": "osc_2_waveform",
    "Osc A2 Volume": "osc_mix",
    "Osc A2 Detune": "osc_2_detune",
    "Filter A Cutoff": "filter_cutoff",
    "Filter A Reso": "filter_resonance",
    "Filter A Type": "filter_type",
    "AmpEnv A Attack": "amp_attack",
    "AmpEnv A Decay": "amp_decay",
    "AmpEnv A Sustain": "amp_sustain",
    "AmpEnv A Release": "amp_release",
    "ModEnv 1 Decay": "filter_env_decay",
    "LFO 1 Rate": "lfo_rate",
    "LFO 1 Gain": "lfo_to_cutoff",
    "FilterCtl KeyTrk": "velocity_to_filter",
    "Distort DryWet": "distortion_mix",
    "Reverb Dry/Wet": "reverb_mix",
}


def clean_content(content: str) -> str:
    """Light fixup so Python's ast can read the dump's outer tuple."""
    content = content.strip()

    if content.endswith(','):
        content = content[:-1]

    has_start = content.startswith('(')
    has_end = content.endswith(')')

    if has_end and not has_start:
        content = '(' + content
    elif has_start and not has_end:
        content = content + ')'
    elif not has_start and not has_end:
        content = f"({content})"

    return content


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _amount_for_destination(params_dict: Dict[str, Any], amount_key: str, dest_key: str, target_dest: float) -> float:
    """Return amount when the modulation routing matches the target destination, else 0."""
    dest = _coerce_float(params_dict.get(dest_key))
    if dest is None:
        return 0.0
    if abs(dest - target_dest) > 0.05:
        return 0.0
    amount = _coerce_float(params_dict.get(amount_key)) or 0.5
    return abs(amount * 2.0 - 1.0)


def _filter_env_amount(params_dict: Dict[str, Any]) -> float:
    """Map the bipolar filter envelope amount to [0,1] (0.5 = neutral)."""
    raw = _coerce_float(params_dict.get("xModEnv1 Dest1Am"))
    if raw is None:
        raw = _coerce_float(params_dict.get("xModEnv1 Dest2Am"))
    if raw is None:
        return 0.5
    return clamp_unit(raw)


def _noise_level(params_dict: Dict[str, Any]) -> float:
    """Sylenth1 has no noise oscillator; estimate from B-side oscillator volume."""
    candidates = [
        _coerce_float(params_dict.get("Osc B1 Volume")),
        _coerce_float(params_dict.get("Osc B2 Volume")),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0.0
    return clamp_unit(max(candidates))


def _distortion_mix(params_dict: Dict[str, Any]) -> float:
    base = _coerce_float(params_dict.get("Distort DryWet")) or 0.0
    enabled = _coerce_float(params_dict.get("Sw DistOnOff"))
    if enabled is not None and enabled < 0.5:
        return 0.0
    return clamp_unit(base)


def _reverb_mix(params_dict: Dict[str, Any]) -> float:
    base = _coerce_float(params_dict.get("Reverb Dry/Wet")) or 0.0
    enabled = _coerce_float(params_dict.get("Sw ReverbOnOff"))
    if enabled is not None and enabled < 0.5:
        return 0.0
    return clamp_unit(base)


def _lfo_to_pitch(params_dict: Dict[str, Any]) -> float:
    return _amount_for_destination(params_dict, "xModLFO1 Dest1Am", "yModLFO1 Dest1", target_dest=0.0)


def _lfo_to_cutoff(params_dict: Dict[str, Any]) -> float:
    return clamp_unit(_coerce_float(params_dict.get("LFO 1 Gain")) or 0.0)


def map_sylenth_to_charter(params_dict: Dict[str, Any]) -> List[float]:
    """Sylenth1 parameter dict -> 20-length charter vector in [0,1]."""
    values: Dict[str, float] = {}

    # Direct mappings.
    for sylenth_key, charter_name in SYLENTH_KEY_MAP.items():
        raw = _coerce_float(params_dict.get(sylenth_key))
        if raw is not None:
            values[charter_name] = clamp_unit(raw)

    # Synthesised mappings.
    values.setdefault("noise_level", _noise_level(params_dict))
    values.setdefault("filter_env_amount", _filter_env_amount(params_dict))
    values.setdefault("distortion_mix", _distortion_mix(params_dict))
    values.setdefault("reverb_mix", _reverb_mix(params_dict))
    values.setdefault("lfo_to_pitch", _lfo_to_pitch(params_dict))
    values.setdefault("lfo_to_cutoff", _lfo_to_cutoff(params_dict))

    vector: List[float] = []
    for idx, charter_name in enumerate(PARAM_NAMES):
        raw = values.get(charter_name)
        if raw is None:
            raw = 0.5 if idx in BIPOLAR_INDICES else 0.0
        if idx in DISCRETE_INDICES:
            raw = snap_discrete(raw, idx)
        vector.append(clamp_unit(raw))
    return vector


def _vector_to_record(description: str, vector: List[float], raw_text: Optional[str] = None) -> Dict[str, Any]:
    continuous: Dict[str, float] = {}
    categorical: Dict[str, float] = {}
    for idx, name in enumerate(PARAM_NAMES):
        value = vector[idx]
        if idx in DISCRETE_INDICES:
            categorical[name] = value
        else:
            continuous[name] = value
    return {
        "raw_text": raw_text or description,
        "description": description,
        "parameters": {
            "continuous": continuous,
            "binary": {},
            "categorical": categorical,
        },
    }


def _iter_anchor_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    records: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        prompts = entry.get("prompts") or []
        if isinstance(prompts, str):
            prompts = [prompts]
        params_block = entry.get("parameters", {})
        if isinstance(params_block, dict) and "continuous" in params_block:
            vector = [0.5 if idx in BIPOLAR_INDICES else 0.0 for idx in range(len(PARAM_NAMES))]
            for idx, name in enumerate(PARAM_NAMES):
                if name in params_block.get("continuous", {}):
                    vector[idx] = clamp_unit(float(params_block["continuous"][name]))
                elif name in params_block.get("categorical", {}):
                    vector[idx] = snap_discrete(float(params_block["categorical"][name]), idx)
        elif isinstance(params_block, list):
            vector = [clamp_unit(float(v)) for v in params_block]
            while len(vector) < 20:
                vector.append(0.0)
        else:
            continue
        for prompt in prompts:
            text = str(prompt).strip()
            if not text:
                continue
            records.append(_vector_to_record(text, vector, raw_text=text))
    return records


def convert_fxp_dump_to_json(input_file, output_file, anchor_path: Optional[Path] = ANCHOR_PRESETS_PATH) -> int:
    """Convert a Sylenth1 dump to a charter-compliant JSON dataset.

    Returns the number of records written (0 if input is missing/invalid).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    dataset: List[Dict[str, Any]] = []

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 0

    raw_content = input_path.read_text(encoding="utf-8")
    fixed_content = clean_content(raw_content)

    try:
        parsed_data = ast.literal_eval(fixed_content)
    except Exception as exc:
        print(f"Error parsing file: {exc}")
        print("Tip: Check if your text file starts with '(' and ends with ')'.")
        return 0

    if isinstance(parsed_data, tuple):
        raw_data: List[Any] = [parsed_data]
    elif isinstance(parsed_data, list):
        raw_data = parsed_data
    else:
        print(f"Unexpected data format: {type(parsed_data)}")
        return 0

    for entry in raw_data:
        if isinstance(entry, tuple) and len(entry) == 2:
            name, params_dict = entry
        elif isinstance(entry, dict):
            params_dict = entry
            name = params_dict.get("description", "Unknown Preset")
        else:
            continue

        if not isinstance(params_dict, dict):
            continue

        description = params_dict.get("description", name)
        description = str(description).replace(".fxp", "").replace("_", " ").strip()

        vector = map_sylenth_to_charter(params_dict)
        dataset.append(_vector_to_record(description, vector))

    if anchor_path is not None:
        dataset.extend(_iter_anchor_records(Path(anchor_path)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Success! Converted {len(dataset)} preset(s) to {output_path}")

    try:
        publish_command(
            "prepare_dataset.py",
            status="ok",
            detail={
                "records": len(dataset),
                "output": str(output_path),
                "charter_version": "1.0",
            },
        )
    except Exception:  # pragma: no cover - best effort  # nosec B110
        pass  # publishing failures must never break the data pipeline

    return len(dataset)


def main() -> None:
    convert_fxp_dump_to_json(DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
