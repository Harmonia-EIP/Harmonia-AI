import torch
import json

from scripts import server as server_module


class FakeTokenizer:
    def __call__(self, prompt, return_tensors, padding, truncation, max_length):
        _ = (prompt, return_tensors, padding, truncation, max_length)
        return {
            "input_ids": torch.tensor([[101, 102, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        _ = (input_ids, attention_mask)
        return torch.tensor([[0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92]])


class FakeRuntime:
    def __init__(self, ready=True, error=""):
        self.model = FakeModel() if ready else None
        self.tokenizer = FakeTokenizer() if ready else None
        self.ready = ready
        self.error = error
        self.model_version = "test-v1"
        self.model_hash = "abc123"
        self.model_path = "/tmp/models/test-v1/my_plugin_ai.pth"
        self.model_metadata_path = "/tmp/models/test-v1/my_plugin_ai.meta.json"


def test_health_ok(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["model_path"].endswith("my_plugin_ai.pth")
    assert body["model_version"] == "test-v1"
    assert body["model_hash"] == "abc123"


def test_health_degraded(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_get_runtime",
        lambda: FakeRuntime(ready=False, error="Model file not found."),
    )
    client = server_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "degraded"
    assert body["model_ready"] is False


def test_generate_success(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "Warm synth pad"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["metadata"]["name"] == "Warm synth pad"
    assert body["metadata"]["model_version"] == "test-v1"
    assert body["metadata"]["model_hash"] == "abc123"
    assert len(body["parameters"]) == len(server_module.PARAM_KEYS)


def test_generate_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", data="{", content_type="application/json")

    assert response.status_code == 400
    assert "Invalid or missing JSON" in response.get_json()["error"]


def test_generate_rejects_non_string_prompt(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": 123})

    assert response.status_code == 400
    assert "must be a string" in response.get_json()["error"]


def test_generate_rejects_too_long_prompt(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    oversized_prompt = "x" * (server_module.MAX_PROMPT_LENGTH + 1)
    response = client.post("/generate", json={"prompt": oversized_prompt})

    assert response.status_code == 400
    assert "too long" in response.get_json()["error"]


def test_generate_returns_503_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=False))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "Deep bass"})

    assert response.status_code == 503


def test_metrics_latest_returns_latest_benchmark(tmp_path, monkeypatch):
    benchmark_file = tmp_path / "history.json"
    report_file = tmp_path / "eval.json"
    report_file.write_text(json.dumps({"metrics": {"mse": 0.12}}), encoding="utf-8")
    benchmark_file.write_text(
        json.dumps(
            [
                {"timestamp": "2026-03-01 10:00:00", "model_version": "old-v"},
                {
                    "timestamp": "2026-03-02 11:00:00",
                    "model_version": "new-v",
                    "evaluation_report_path": str(report_file),
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server_module, "BENCHMARK_FILE", benchmark_file)

    client = server_module.app.test_client()
    response = client.get("/metrics/latest")

    assert response.status_code == 200
    body = response.get_json()
    assert body["latest_benchmark"]["model_version"] == "new-v"
    assert body["latest_evaluation_report"]["metrics"]["mse"] == 0.12


def test_metrics_latest_returns_404_when_missing(monkeypatch, tmp_path):
    missing_history = tmp_path / "missing-history.json"
    monkeypatch.setattr(server_module, "BENCHMARK_FILE", missing_history)

    client = server_module.app.test_client()
    response = client.get("/metrics/latest")

    assert response.status_code == 404


