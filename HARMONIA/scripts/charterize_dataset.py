#!/usr/bin/env python3
"""Convert a Sylenth1 dataset (NPY or JSON) to the 20-parameter HARMONIA charter.

Input record formats supported:

1. ``{"name": str, "raw_text": str, "parameters": {"continuous": {...},
   "binary": {...}, "categorical": {...}}}`` — output of the original
   Sylenth1 cleaning pipeline (data/processed/cleaned_dataset.npy).
2. ``{"description": str, "parameters": {<sylenth_key>: value, ...}}`` —
   ASCII dump style.
3. Flat ``{<sylenth_key>: value, ...}`` records.

Output: charter-shaped records that ``PresetDataset`` recognises and trains
on with ``charter_mode = True``. Anchor presets from
``data/raw/anchor_presets.json`` are appended by default.

Usage:
    python scripts/charterize_dataset.py \\
        --input data/processed/cleaned_dataset.npy \\
        --output data/processed/cleaned_dataset.charter.npy

Run ``train.py`` with the new file:
    HARMONIA_DATASET_PATH=data/processed/cleaned_dataset.charter.npy \\
        python scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.prepare_dataset import (
    ANCHOR_PRESETS_PATH,
    _iter_anchor_records,
    _vector_to_record,
    map_sylenth_to_charter,
)
from src.charter import PARAM_NAMES
from src.dataset_profiles import (
    apply_profile,
    autodetect_profile,
    list_available_profiles,
    load_profile,
)


def _flatten_sylenth_parameters(params: Any) -> Dict[str, Any]:
    """Merge continuous/binary/categorical sub-dicts (or flat dict) into one."""
    if not isinstance(params, dict):
        return {}
    flat: Dict[str, Any] = {}
    if any(k in params for k in ("continuous", "binary", "categorical")):
        for group in ("continuous", "binary", "categorical"):
            sub = params.get(group)
            if isinstance(sub, dict):
                flat.update(sub)
    else:
        flat.update(params)
    return flat


def _extract_description(item: Dict[str, Any]) -> str:
    for key in ("raw_text", "description", "name"):
        value = item.get(key)
        if value:
            return str(value).replace(".fxp", "").replace("_", " ").strip()
    return "Unknown"


def _iter_input_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        payload = loaded.tolist() if isinstance(loaded, np.ndarray) else loaded
        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict):
            yield payload
        else:
            raise ValueError(f"Unsupported NPY content in {path}")
        return

    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict):
            yield payload
        else:
            raise ValueError(f"Unsupported JSON content in {path}")
        return

    raise ValueError(f"Unsupported input format: {suffix}")


def _resolve_profile(profile_arg: Optional[str], sample_keys: List[str]) -> Optional[Dict[str, Any]]:
    """Return a profile dict, or None when caller asked for the legacy mapping."""
    if profile_arg is None:
        return None
    if profile_arg == "auto":
        detected = autodetect_profile(sample_keys)
        if detected is None:
            raise RuntimeError(
                "Could not auto-detect a profile from the input dataset. "
                "Run `python scripts/inspect_dataset.py --input <file>` to scaffold one."
            )
        return detected
    return load_profile(profile_arg)


def charterize(
    input_path: Path,
    output_path: Path,
    anchor_path: Optional[Path] = ANCHOR_PRESETS_PATH,
    limit: Optional[int] = None,
    verbose: bool = True,
    profile: Optional[str] = "sylenth1",
) -> Dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records: List[Dict[str, Any]] = []
    skipped = 0
    converted = 0
    profile_dict: Optional[Dict[str, Any]] = None
    profile_resolved = False

    for idx, item in enumerate(_iter_input_records(input_path)):
        if not isinstance(item, dict):
            skipped += 1
            continue

        flat_params = _flatten_sylenth_parameters(item.get("parameters", item))
        if not flat_params:
            skipped += 1
            continue

        if not profile_resolved:
            profile_dict = _resolve_profile(profile, list(flat_params.keys()))
            profile_resolved = True
            if verbose and profile_dict is not None:
                print(f"Using profile     : {profile_dict.get('name', '<unnamed>')}")
            elif verbose:
                print("Using profile     : <legacy hardcoded mapping>")

        description = _extract_description(item)
        if profile_dict is not None:
            vector = apply_profile(profile_dict, flat_params)
        else:
            vector = map_sylenth_to_charter(flat_params)
        records.append(_vector_to_record(description, vector))
        converted += 1

        if verbose and converted and converted % 5000 == 0:
            print(f"  ... {converted} records charterised")

        if limit is not None and converted >= limit:
            break

    anchor_count = 0
    if anchor_path is not None:
        anchor_records = list(_iter_anchor_records(Path(anchor_path)))
        records.extend(anchor_records)
        anchor_count = len(anchor_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".npy":
        np.save(output_path, np.array(records, dtype=object), allow_pickle=True)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    stats = {
        "input_records": converted + skipped,
        "converted": converted,
        "skipped": skipped,
        "anchors_added": anchor_count,
        "output_records": len(records),
        "profile_name": profile_dict.get("name") if profile_dict else "legacy",
    }
    if verbose:
        print()
        print(f"Input file        : {input_path}")
        print(f"Output file       : {output_path}")
        print(f"Source converted  : {converted}")
        print(f"Skipped           : {skipped}")
        print(f"Anchors added     : {anchor_count}")
        print(f"Total written     : {len(records)}")
        print(f"Charter params    : {len(PARAM_NAMES)}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Sylenth1 dataset to the HARMONIA 20-param charter.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "processed" / "cleaned_dataset.npy", help="Input NPY or JSON file.")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "processed" / "cleaned_dataset.charter.npy", help="Output file (NPY recommended for large datasets).")
    parser.add_argument("--no-anchors", action="store_true", help="Skip injection of anchor presets.")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N input records (for tests).")
    parser.add_argument("--profile", default="sylenth1", help='Profile name in src/profiles/, "auto" to detect, "legacy" to use the hardcoded Sylenth1 mapping.')
    parser.add_argument("--list-profiles", action="store_true", help="Print available profiles and exit.")
    args = parser.parse_args()

    if args.list_profiles:
        names = list_available_profiles()
        if not names:
            print("(no profiles installed in src/profiles/)")
        else:
            for name in names:
                profile = load_profile(name)
                print(f"  {name:<14} {profile.get('description', '')}")
        return 0

    anchor_path = None if args.no_anchors else ANCHOR_PRESETS_PATH
    profile = None if args.profile == "legacy" else args.profile
    charterize(args.input, args.output, anchor_path=anchor_path, limit=args.limit, profile=profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
