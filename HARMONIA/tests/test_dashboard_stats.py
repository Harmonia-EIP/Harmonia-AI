import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts import dashboard_stats


def test_build_snapshot_has_expected_sections():
    snapshot = dashboard_stats.build_snapshot()
    assert "timestamp" in snapshot
    assert isinstance(snapshot["models"], list)
    assert isinstance(snapshot["presets"], list)
    assert isinstance(snapshot["benchmark_history"], list)
    assert isinstance(snapshot["evaluation_reports"], list)
    assert isinstance(snapshot["charter"], list)
    assert len(snapshot["charter"]) == 20


def test_main_writes_snapshot(tmp_path, monkeypatch):
    out = tmp_path / "snapshot.json"
    monkeypatch.setenv("HARMONIA_PUSH_METRICS", "0")
    # Force the dashboard_stats main to write to tmp_path and skip the push.
    monkeypatch.setattr(sys, "argv", ["dashboard_stats.py", "--output", str(out), "--no-push"])

    exit_code = dashboard_stats.main()

    assert exit_code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "charter" in payload
    assert isinstance(payload["models"], list)
