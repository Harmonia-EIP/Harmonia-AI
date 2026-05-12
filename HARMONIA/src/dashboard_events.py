"""Publish events (training, generation, command, system) to the Harmonia dashboard.

Each event is written to the local mirror under ``metrics_dashboard/events/``
and best-effort POSTed to the configured receiver (``HARMONIA_METRICS_URL``).
Calls never raise; failures are returned in the result dict so callers can
log without crashing the user's command.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode

from src.metrics_publisher import (
    DEFAULT_METRICS_URL,
    _validate_metrics_url,
    resolve_token,
)
import http.client
from urllib.parse import urlparse


VALID_KINDS = {"training", "generation", "command", "system", "dataset"}
DEFAULT_KIND = "command"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(value: str, max_len: int = 32) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)
    safe = safe.strip("-._")
    return (safe or "event").lower()[:max_len]


def _resolve_base_dir(base_dir: Optional[Path]) -> Path:
    return Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent


def _local_event_dir(base_dir: Path) -> Path:
    target = base_dir / "metrics_dashboard" / "events"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_local_mirror(base_dir: Path, kind: str, payload: Mapping[str, Any]) -> Path:
    events_dir = _local_event_dir(base_dir)
    name = f"{_now_compact()}_{_slugify(kind)}_{_slugify(str(payload.get('model_version', 'unknown')))}_{secrets.token_hex(2)}.json"
    path = events_dir / name
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _post_json(url: str, payload: Mapping[str, Any], token: str, kind: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    endpoint = parsed.path or "/"
    query = parsed.query
    extra_query = urlencode({"kind": kind})
    endpoint = f"{endpoint}?{extra_query}" if not query else f"{endpoint}?{query}&{extra_query}"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.netloc, timeout=15)
    try:
        conn.request("POST", endpoint, body=body, headers=headers)
        response = conn.getresponse()
        status = int(response.status)
        text = response.read().decode("utf-8", errors="replace")
        return {"ok": 200 <= status < 300, "status": status, "response": text}
    finally:
        conn.close()


def publish_event(
    kind: str,
    payload: Mapping[str, Any],
    *,
    url: Optional[str] = None,
    token: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Publish a dashboard event.

    Returns a result dict with ``ok``, ``status`` (HTTP code or None), ``local_path``,
    and either ``response`` or ``error``. Never raises.
    """
    kind = (kind or DEFAULT_KIND).lower()
    if kind not in VALID_KINDS:
        kind = DEFAULT_KIND

    base = _resolve_base_dir(base_dir)
    enriched: Dict[str, Any] = {
        "event_kind": kind,
        "timestamp": payload.get("timestamp") or _now_iso(),
    }
    enriched.update(payload)

    result: Dict[str, Any] = {"ok": False, "kind": kind}

    try:
        local_path = _write_local_mirror(base, kind, enriched)
        result["local_path"] = str(local_path)
    except OSError as exc:
        result["error"] = f"local-write-failed: {exc}"
        return result

    target_url = (url or os.environ.get("HARMONIA_METRICS_URL") or DEFAULT_METRICS_URL).strip()
    valid_url, reason = _validate_metrics_url(target_url)
    if not valid_url:
        result.update({"skipped": True, "reason": reason, "url": target_url})
        return result

    auth_token = (token or resolve_token(base)).strip()
    if not auth_token:
        result.update({"skipped": True, "reason": "missing-token", "url": target_url})
        return result

    if os.environ.get("HARMONIA_PUSH_METRICS", "1").strip().lower() in {"0", "false", "no", "off"}:
        result.update({"skipped": True, "reason": "push-disabled", "url": target_url})
        return result

    try:
        response = _post_json(target_url, enriched, auth_token, kind)
        result.update(response)
        result["url"] = target_url
    except Exception as exc:  # pragma: no cover - network failures
        result.update({"error": f"network: {exc}", "url": target_url})

    return result


def training_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": report.get("timestamp") or _now_iso(),
        "model_version": report.get("model_version"),
        "model_hash": report.get("model_hash"),
        "metrics": report.get("metrics", {}),
        "loss_history": report.get("loss_history"),
        "param_keys": report.get("param_keys"),
        "seed": report.get("seed"),
        "val_split": report.get("val_split"),
        "train_size": report.get("train_size"),
        "val_size": report.get("val_size"),
    }


def generation_payload(
    prompt: str,
    parameters: Mapping[str, float],
    values: Optional[list],
    *,
    model_version: str = "unknown",
    model_hash: str = "unknown",
    charter_version: Optional[str] = None,
    source: str = "cli",
) -> Dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "prompt": prompt,
        "preset_name": prompt,
        "parameters": dict(parameters),
        "values": list(values) if values is not None else None,
        "model_version": model_version,
        "model_hash": model_hash,
        "charter_version": charter_version,
        "source": source,
    }


def command_payload(
    command: str,
    *,
    status: str = "ok",
    duration_seconds: Optional[float] = None,
    detail: Optional[Mapping[str, Any]] = None,
    model_version: str = "current",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "command": command,
        "status": status,
        "model_version": model_version,
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = round(float(duration_seconds), 3)
    if detail:
        payload["detail"] = dict(detail)
    return payload


def system_payload(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {"timestamp": _now_iso(), "model_version": snapshot.get("model_version", "snapshot")}
    payload.update(snapshot)
    return payload


def publish_training(report: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return publish_event("training", training_payload(report), **kwargs)


def publish_generation(prompt: str, parameters: Mapping[str, float], values: Optional[list] = None, **kwargs: Any) -> Dict[str, Any]:
    pub_kwargs = {k: v for k, v in kwargs.items() if k in {"url", "token", "base_dir"}}
    payload_kwargs = {k: v for k, v in kwargs.items() if k not in pub_kwargs}
    payload = generation_payload(prompt, parameters, values, **payload_kwargs)
    return publish_event("generation", payload, **pub_kwargs)


def publish_command(command: str, **kwargs: Any) -> Dict[str, Any]:
    pub_kwargs = {k: v for k, v in kwargs.items() if k in {"url", "token", "base_dir"}}
    payload_kwargs = {k: v for k, v in kwargs.items() if k not in pub_kwargs}
    payload = command_payload(command, **payload_kwargs)
    return publish_event("command", payload, **pub_kwargs)


def publish_system(snapshot: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return publish_event("system", system_payload(snapshot), **kwargs)
