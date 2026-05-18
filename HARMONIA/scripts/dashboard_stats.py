#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import PARAM_NAMES, charter_metadata
from src.dashboard_events import publish_system


SAVED_MODELS_DIR = BASE_DIR / "saved_models"
ARCHIVE_DIR = SAVED_MODELS_DIR / "archive"
BENCHMARK_HISTORY = BASE_DIR / "benchmarks" / "history.json"
REPORTS_DIR = BASE_DIR / "benchmarks" / "reports"
PRESETS_DIR = BASE_DIR / "data" / "processed"
LATEST_POINTER = SAVED_MODELS_DIR / "latest_model.json"
DEFAULT_OUTPUT = BASE_DIR / "metrics_dashboard" / "stats_snapshot.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _gather_models() -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    seen_versions: set = set()

    candidates: List[Path] = list(SAVED_MODELS_DIR.glob("*.meta.json"))
    if ARCHIVE_DIR.exists():
        for sub in ARCHIVE_DIR.iterdir():
            if sub.is_dir():
                candidates.extend(sub.glob("*.meta.json"))
            elif sub.suffix == ".json" and sub.name.endswith(".meta.json"):
                candidates.append(sub)

    for meta_path in sorted(candidates):
        payload = _load_json(meta_path, default={})
        if not isinstance(payload, dict):
            continue
        version = payload.get("model_version") or meta_path.stem.replace(".meta", "")
        if version in seen_versions:
            continue
        seen_versions.add(version)

        eval_path = Path(payload.get("evaluation_report_path", "")) if payload.get("evaluation_report_path") else None
        eval_payload = _load_json(eval_path, default={}) if eval_path and eval_path.exists() else {}
        metrics = eval_payload.get("metrics") if isinstance(eval_payload, dict) else None

        found.append(
            {
                "model_version": version,
                "model_hash": payload.get("model_hash"),
                "trained_at": payload.get("trained_at"),
                "dataset_size": payload.get("dataset_size"),
                "epochs": payload.get("epochs"),
                "batch_size": payload.get("batch_size"),
                "learning_rate": payload.get("learning_rate"),
                "plugin_param_count": payload.get("plugin_param_count"),
                "param_keys": payload.get("param_keys"),
                "tokenizer_model_id": payload.get("tokenizer_model_id"),
                "charter_mode": payload.get("charter_mode"),
                "charter_version": payload.get("charter_version"),
                "metrics": metrics,
                "is_archived": "archive" in str(meta_path),
                "metadata_path": str(meta_path),
                "evaluation_report_path": str(eval_path) if eval_path else None,
            }
        )

    return found


def _gather_presets(limit: int = 50) -> List[Dict[str, Any]]:
    if not PRESETS_DIR.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(PRESETS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _load_json(path, default=None)
        if not isinstance(payload, dict):
            continue
        params = payload.get("parameters")
        # accept charter-style ({"continuous": {...}}) OR flat dict
        if isinstance(params, dict) and ("continuous" in params or set(PARAM_NAMES).intersection(params.keys())):
            metadata = payload.get("metadata") or {}
            items.append(
                {
                    "file": str(path.relative_to(BASE_DIR)),
                    "name": metadata.get("name") or path.stem,
                    "model_version": metadata.get("model_version"),
                    "model_hash": metadata.get("model_hash"),
                    "charter_version": metadata.get("charter_version"),
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "parameters": payload.get("parameters"),
                    "values": payload.get("values"),
                }
            )
        if len(items) >= limit:
            break
    return items


def _gather_reports(limit: int = 20) -> List[Dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(REPORTS_DIR.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _load_json(path, default=None)
        if not isinstance(payload, dict):
            continue
        items.append(
            {
                "file": str(path.relative_to(BASE_DIR)),
                "timestamp": payload.get("timestamp"),
                "model_version": payload.get("model_version"),
                "model_hash": payload.get("model_hash"),
                "metrics": payload.get("metrics"),
                "loss_history": payload.get("loss_history"),
                "param_keys": payload.get("param_keys"),
            }
        )
        if len(items) >= limit:
            break
    return items


def build_snapshot() -> Dict[str, Any]:
    models = _gather_models()
    presets = _gather_presets()
    reports = _gather_reports()
    history = _load_json(BENCHMARK_HISTORY, default=[]) or []
    latest_pointer = _load_json(LATEST_POINTER, default={}) or {}
    return {
        "timestamp": _now_iso(),
        "model_version": latest_pointer.get("model_version", "snapshot"),
        "active_model": latest_pointer,
        "models": models,
        "presets": presets,
        "evaluation_reports": reports,
        "benchmark_history": history,
        "charter": charter_metadata(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and (optionally) push a Harmonia dashboard snapshot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the snapshot JSON locally.")
    parser.add_argument("--no-push", action="store_true", help="Skip remote push (still writes local snapshot).")
    parser.add_argument("--url", default=None, help="Override metrics receiver URL.")
    parser.add_argument("--token", default=None, help="Override security token.")
    args = parser.parse_args()

    snapshot = build_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Snapshot written to {args.output} ({len(snapshot['models'])} models, {len(snapshot['presets'])} presets).")

    if args.no_push:
        return 0

    result = publish_system(snapshot, url=args.url, token=args.token, base_dir=BASE_DIR)
    if result.get("ok"):
        print(f"[DASHBOARD] Snapshot pushed (HTTP {result.get('status')}).")
        return 0
    if result.get("skipped"):
        print(f"[DASHBOARD] Push skipped: {result.get('reason')}.")
        return 0
    print(f"[DASHBOARD] Push failed: {result.get('error') or result.get('status')}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
