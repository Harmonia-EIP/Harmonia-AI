#!/usr/bin/env python3
"""Compare a generated preset to similarly-named presets in the dataset.

Picks every entry whose ``raw_text`` or ``description`` matches ``--keyword``,
computes the centroid (mean) of their 20-param charter vectors, then reports:
  * the L1 and L2 distance from the centroid;
  * per-parameter delta vs. the centroid (with std across the cluster);
  * the closest individual real preset (nearest-neighbour).

Output is plain text so it lands cleanly in CI logs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import BIPOLAR_INDICES, DISCRETE_INDICES, PARAM_NAMES, snap_discrete


def _flatten_charter(parameters: dict) -> list[float] | None:
    if not isinstance(parameters, dict):
        return None
    continuous = {k: float(v) for k, v in (parameters.get("continuous") or {}).items()}
    categorical = {k: float(v) for k, v in (parameters.get("categorical") or {}).items()}
    cat_max = {k: max(1.0, categorical.get(k, 0.0)) for k in categorical}
    vector: list[float] = []
    for idx, key in enumerate(PARAM_NAMES):
        if key in continuous:
            value = continuous[key]
        elif key in categorical:
            raw = categorical[key]
            divisor = cat_max.get(key, 1.0) or 1.0
            raw = raw / divisor if divisor > 1.0 else raw
            value = snap_discrete(raw, idx) if idx in DISCRETE_INDICES else raw
        else:
            value = 0.5 if idx in BIPOLAR_INDICES else 0.0
        vector.append(max(0.0, min(1.0, value)))
    return vector


def _generated_vector(payload: dict) -> list[float]:
    if isinstance(payload.get("values"), list) and len(payload["values"]) == len(PARAM_NAMES):
        return [float(v) for v in payload["values"]]
    flat = _flatten_charter(payload.get("parameters") or {})
    if not flat:
        params = payload.get("parameters", {})
        return [float(params.get(k, 0.0)) for k in PARAM_NAMES]
    return flat


def _load_dataset(path: Path):
    raw = np.load(path, allow_pickle=True)
    return list(raw.tolist())


def _matches(item: dict, keyword: str) -> bool:
    bag = " ".join(str(item.get(k, "")) for k in ("raw_text", "description", "name")).lower()
    needles = [n.strip() for n in keyword.lower().split() if n.strip()]
    return all(n in bag for n in needles)


def _l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True, help="Generated preset JSON to evaluate.")
    parser.add_argument("--keyword", required=True, help="Word(s) to filter the dataset on (case-insensitive AND).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "cleaned_dataset.charter.npy",
    )
    parser.add_argument("--top", type=int, default=5, help="Show top-N nearest real presets.")
    args = parser.parse_args()

    payload = json.loads(args.preset.read_text(encoding="utf-8"))
    generated = _generated_vector(payload)
    if len(generated) != len(PARAM_NAMES):
        print(f"Generated vector has {len(generated)} dims, expected {len(PARAM_NAMES)}.")
        return 2

    dataset = _load_dataset(args.dataset)
    cluster = []
    for item in dataset:
        if not _matches(item, args.keyword):
            continue
        vec = _flatten_charter(item.get("parameters") or {})
        if vec is None:
            continue
        cluster.append({"label": str(item.get("raw_text") or item.get("name") or ""), "vector": vec})

    print(f"== Preset comparison ==")
    print(f"  generated file : {args.preset}")
    print(f"  prompt          : {payload.get('metadata', {}).get('name')}")
    print(f"  dataset         : {args.dataset}")
    print(f"  keyword filter  : '{args.keyword}' -> {len(cluster)} real preset(s) match")
    if not cluster:
        print("\nNo matching real presets — try a broader keyword.")
        return 0

    matrix = np.array([c["vector"] for c in cluster])
    centroid = matrix.mean(axis=0).tolist()
    stds = matrix.std(axis=0).tolist()

    l1_centroid = _l1(generated, centroid)
    l2_centroid = _l2(generated, centroid)
    mae = l1_centroid / len(PARAM_NAMES)
    mse = sum((g - c) ** 2 for g, c in zip(generated, centroid)) / len(PARAM_NAMES)

    print()
    print(f"  vs cluster centroid : L1={l1_centroid:.3f}  L2={l2_centroid:.3f}  MAE={mae:.3f}  MSE={mse:.4f}")
    print(f"  cluster spread (avg std per param): {statistics.mean(stds):.3f}")

    print("\n  per-parameter | generated | centroid | delta | cluster std | z-score")
    print("  " + "-" * 76)
    for idx, name in enumerate(PARAM_NAMES):
        delta = generated[idx] - centroid[idx]
        std = stds[idx] if stds[idx] > 1e-6 else 1e-6
        z = delta / std
        flag = "  ⚠" if abs(z) > 2.0 else ""
        print(f"  {name:<22} {generated[idx]:8.3f}   {centroid[idx]:7.3f}   {delta:+7.3f}   {stds[idx]:7.3f}   {z:+6.2f}{flag}")

    distances = [(_l2(generated, c["vector"]), c["label"], c["vector"]) for c in cluster]
    distances.sort(key=lambda t: t[0])
    print(f"\n  Top {args.top} nearest real presets (L2):")
    for dist, label, _ in distances[: args.top]:
        print(f"    {dist:6.3f}  {label[:80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
