from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

DEFAULT_METRICS_URL = os.environ.get("HARMONIA_METRICS_URL", "https://harmonia.mcoet.com/receiver.php")
DEFAULT_TOKEN = os.environ.get("METRICS_TOKEN", os.environ.get("HARMONIA_METRICS_TOKEN", ""))


def _load_local_token(base_dir: Path) -> str:
    candidates = [
        base_dir / "metrics_dashboard" / ".env.local",
        base_dir / ".env.local",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "METRICS_TOKEN":
                return value.strip().strip('"').strip("'")
    return ""


def resolve_token(base_dir: Path) -> str:
    token = DEFAULT_TOKEN.strip()
    if token:
        return token
    return _load_local_token(base_dir)


def _guess_content_type(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "application/json"
    return "application/octet-stream"


def push_metrics_report(report_path: Path, url: str | None = None, token: str | None = None, base_dir: Path | None = None) -> dict:
    base_dir = base_dir or Path.cwd()
    url = (url or DEFAULT_METRICS_URL).strip()
    token = (token or resolve_token(base_dir)).strip()

    if not report_path.exists():
        return {"ok": False, "skipped": True, "reason": f"report not found: {report_path}"}
    if not token:
        return {"ok": False, "skipped": True, "reason": "missing token"}

    payload = report_path.read_bytes()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": _guess_content_type(report_path),
    }
    req = request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= getattr(response, "status", 200) < 300,
                "status": getattr(response, "status", 200),
                "response": body,
                "url": url,
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return {"ok": False, "status": exc.code, "response": body, "url": url}
    except Exception as exc:  # pragma: no cover - network failures
        return {"ok": False, "error": str(exc), "url": url}


def latest_report_from_history(history_path: Path) -> Path | None:
    if not history_path.exists():
        return None

    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(history, list) or not history:
        return None

    latest = history[-1]
    if not isinstance(latest, dict):
        return None

    report_path = latest.get("evaluation_report_path")
    if not report_path:
        return None

    candidate = Path(str(report_path))
    return candidate if candidate.exists() else None

