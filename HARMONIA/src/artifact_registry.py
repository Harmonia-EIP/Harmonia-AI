import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

MODEL_FILE_NAME = "my_plugin_ai.pth"
METADATA_FILE_NAME = "my_plugin_ai.meta.json"
LATEST_POINTER_FILE_NAME = "latest_model.json"
MODEL_FILE_PREFIX = "model_"
MODEL_FILE_SUFFIX = ".pth"
METADATA_FILE_SUFFIX = ".meta.json"
EVAL_REPORT_SUFFIX = ".eval.json"


def _sanitize_version(model_version: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", model_version.strip())
    return sanitized.strip(".-") or "model-unknown"


def build_versioned_paths(saved_models_dir: Path, model_version: str) -> Dict[str, Path]:
    version = _sanitize_version(model_version)
    model_dir = saved_models_dir / version
    return {
        "model_version": version,
        "model_dir": model_dir,
        "model_path": model_dir / MODEL_FILE_NAME,
        "metadata_path": model_dir / METADATA_FILE_NAME,
    }


def build_flat_model_paths(saved_models_dir: Path, model_version: str) -> Dict[str, Path]:
    version = _sanitize_version(model_version)
    return {
        "model_version": version,
        "model_path": saved_models_dir / f"{version}{MODEL_FILE_SUFFIX}",
        "metadata_path": saved_models_dir / f"{version}{METADATA_FILE_SUFFIX}",
        "evaluation_report_path": saved_models_dir / f"{version}{EVAL_REPORT_SUFFIX}",
    }


def write_latest_pointer(saved_models_dir: Path, payload: Dict[str, str]) -> Path:
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = saved_models_dir / LATEST_POINTER_FILE_NAME
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return pointer_path


def _read_latest_pointer(saved_models_dir: Path) -> Dict[str, str]:
    pointer_path = saved_models_dir / LATEST_POINTER_FILE_NAME
    if not pointer_path.exists():
        return {}

    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def _find_latest_flat_artifact(saved_models_dir: Path) -> Dict[str, Path]:
    candidates = sorted(
        saved_models_dir.glob(f"{MODEL_FILE_PREFIX}*{MODEL_FILE_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for model_path in candidates:
        if not model_path.is_file():
            continue
        version = model_path.stem
        metadata_path = saved_models_dir / f"{version}{METADATA_FILE_SUFFIX}"
        report_path = saved_models_dir / f"{version}{EVAL_REPORT_SUFFIX}"
        resolved = {
            "model_version": version,
            "model_path": model_path,
            "metadata_path": metadata_path if metadata_path.exists() else None,
        }
        if report_path.exists():
            resolved["evaluation_report_path"] = report_path
        return resolved
    return {}


def archive_artifacts(archive_dir: Path, artifact: Dict[str, Path]) -> Dict[str, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: Dict[str, Path] = {}

    for key in ("model_path", "metadata_path", "evaluation_report_path"):
        source = artifact.get(key)
        if not source or not Path(source).exists():
            continue
        source_path = Path(source)
        destination = archive_dir / source_path.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if key == "evaluation_report_path":
            shutil.copy2(str(source_path), str(destination))
        else:
            shutil.move(str(source_path), str(destination))
        archived[key] = destination

    return archived


def resolve_latest_model(saved_models_dir: Path, legacy_model_path: Optional[Path] = None, legacy_metadata_path: Optional[Path] = None) -> Dict[str, Path]:
    payload = _read_latest_pointer(saved_models_dir)
    model_path_value = payload.get("model_path")
    metadata_path_value = payload.get("metadata_path")
    model_version = payload.get("model_version")

    if model_path_value:
        # Re-anchor to saved_models_dir using just the filename: the pointer
        # may have been written from a different machine/environment (e.g.
        # a local dev path), so the stored absolute path is not portable.
        model_path = saved_models_dir / Path(model_path_value).name
        metadata_path = saved_models_dir / Path(metadata_path_value).name if metadata_path_value else None
        if model_path.exists():
            resolved = {
                "model_version": model_version or model_path.parent.name,
                "model_path": model_path,
                "metadata_path": metadata_path,
            }
            for key in ("model_hash", "updated_at", "evaluation_report_path"):
                if key in payload:
                    resolved[key] = Path(payload[key]) if key == "evaluation_report_path" else payload[key]
            return resolved

    flat_artifact = _find_latest_flat_artifact(saved_models_dir)
    if flat_artifact:
        return flat_artifact

    for candidate_dir in sorted(saved_models_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not candidate_dir.is_dir():
            continue

        model_path = candidate_dir / MODEL_FILE_NAME
        if not model_path.exists():
            continue

        metadata_path = candidate_dir / METADATA_FILE_NAME
        return {
            "model_version": candidate_dir.name,
            "model_path": model_path,
            "metadata_path": metadata_path if metadata_path.exists() else None,
        }

    if legacy_model_path and legacy_model_path.exists():
        return {
            "model_version": "legacy",
            "model_path": legacy_model_path,
            "metadata_path": legacy_metadata_path if legacy_metadata_path and legacy_metadata_path.exists() else None,
        }

    return {}

