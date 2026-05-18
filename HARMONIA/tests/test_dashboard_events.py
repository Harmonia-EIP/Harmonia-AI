import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src import dashboard_events
from src.dashboard_events import (
    command_payload,
    generation_payload,
    publish_command,
    publish_event,
    publish_generation,
    publish_training,
    training_payload,
)


def _stub_push_disabled(monkeypatch):
    monkeypatch.setenv("HARMONIA_PUSH_METRICS", "0")


def test_publish_event_writes_local_mirror(tmp_path, monkeypatch):
    _stub_push_disabled(monkeypatch)

    result = publish_event(
        "command",
        {"command": "make check", "status": "ok", "model_version": "demo"},
        base_dir=tmp_path,
    )

    assert result["kind"] == "command"
    assert result.get("buffered") is True
    local_path = Path(result["local_path"])
    assert local_path.exists()
    payload = json.loads(local_path.read_text(encoding="utf-8"))
    assert payload["event_kind"] == "command"
    assert payload["command"] == "make check"
    assert payload["model_version"] == "demo"
    assert "system_metrics" in payload


def test_publish_event_normalises_unknown_kind(tmp_path, monkeypatch):
    _stub_push_disabled(monkeypatch)
    result = publish_event("???", {"foo": "bar"}, base_dir=tmp_path)
    assert result["kind"] == dashboard_events.DEFAULT_KIND


def test_training_payload_keeps_metrics():
    payload = training_payload({
        "timestamp": "2026-04-30T10:00:00Z",
        "model_version": "charter_v1",
        "model_hash": "deadbeef",
        "metrics": {"loss": 0.12, "mse": 0.13},
        "loss_history": [0.5, 0.3, 0.12],
        "param_keys": ["osc_1_waveform"],
    })
    assert payload["model_version"] == "charter_v1"
    assert payload["metrics"]["loss"] == 0.12
    assert payload["loss_history"][-1] == 0.12


def test_generation_payload_includes_values():
    payload = generation_payload(
        "Hard Electro Lead",
        {"osc_1_waveform": 0.66},
        [0.66, 0.33],
        model_version="charter_v1",
        model_hash="abc123",
        charter_version="1.0",
        source="cli",
    )
    assert payload["prompt"] == "Hard Electro Lead"
    assert payload["values"] == [0.66, 0.33]
    assert payload["parameters"]["osc_1_waveform"] == 0.66
    assert payload["charter_version"] == "1.0"


def test_command_payload_round_trips_metadata():
    payload = command_payload("make check", status="ok", duration_seconds=12.4, detail={"hash": "x"})
    assert payload["command"] == "make check"
    assert payload["status"] == "ok"
    assert payload["duration_seconds"] == 12.4
    assert payload["detail"] == {"hash": "x"}


def test_publish_command_uses_base_dir(tmp_path, monkeypatch):
    _stub_push_disabled(monkeypatch)
    publish_command("make dashboard-stats", status="ok", base_dir=tmp_path)
    events_dir = tmp_path / "metrics_dashboard" / "events"
    assert events_dir.exists()
    files = list(events_dir.glob("*.json"))
    assert files, "expected one mirrored event file"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["command"] == "make dashboard-stats"


def test_publish_generation_round_trips(tmp_path, monkeypatch):
    _stub_push_disabled(monkeypatch)
    publish_generation(
        "Acid Bass",
        {"filter_resonance": 0.9},
        [0.9],
        model_version="charter_v1",
        base_dir=tmp_path,
    )
    files = list((tmp_path / "metrics_dashboard" / "events").glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["event_kind"] == "generation"
    assert payload["prompt"] == "Acid Bass"


def test_publish_training_invokes_publish_event(tmp_path, monkeypatch):
    _stub_push_disabled(monkeypatch)
    publish_training(
        {
            "model_version": "charter_v1",
            "model_hash": "x",
            "metrics": {"loss": 0.1},
            "loss_history": [0.5, 0.1],
        },
        base_dir=tmp_path,
    )
    files = list((tmp_path / "metrics_dashboard" / "events").glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["event_kind"] == "training"
    assert payload["metrics"]["loss"] == 0.1
