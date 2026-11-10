import torch
import json

import pytest

from scripts import server as server_module


class FakeTokenizer:
    def __call__(self, prompt, return_tensors, padding=False, truncation=False, max_length=None):
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
        self.param_keys = tuple(server_module.PARAM_KEYS)
        self.tokenizer_max_length = 32


class LongTokenizer:
    def __call__(self, prompt, return_tensors, padding=False, truncation=False, max_length=None):
        _ = (prompt, return_tensors, padding, truncation, max_length)
        return {
            "input_ids": torch.ones((1, 40), dtype=torch.long),
            "attention_mask": torch.ones((1, 40), dtype=torch.long),
        }


# /health should report status "ok" and model info when the runtime is ready.
def test_health_ok(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["model_path"].endswith("my_plugin_ai.pth")
    assert body["model_version"] == "test-v1"
    assert body["model_hash"] == "abc123"


# /health should report status "degraded" when the runtime failed to load the model.
def test_health_degraded(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_get_runtime",
        lambda model_key=None: FakeRuntime(ready=False, error="Model file not found."),
    )
    client = server_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "degraded"
    assert body["model_ready"] is False


# /generate should return 200 with parameters and metadata for a valid prompt.
def test_generate_success(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "Warm synth pad"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["metadata"]["name"] == "Warm synth pad"
    assert body["metadata"]["model_version"] == "test-v1"
    assert body["metadata"]["model_hash"] == "abc123"
    assert len(body["parameters"]) == len(server_module.PARAM_KEYS)


# /generate should return 400 for a body that is not valid JSON.
def test_generate_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", data="{", content_type="application/json")

    assert response.status_code == 400
    assert "Invalid or missing JSON" in response.get_json()["error"]


# /generate should return 400 when "prompt" is not a string.
def test_generate_rejects_non_string_prompt(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": 123})

    assert response.status_code == 400
    assert "must be a string" in response.get_json()["error"]


# /generate should return 400 when the prompt exceeds MAX_PROMPT_LENGTH characters.
def test_generate_rejects_too_long_prompt(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    oversized_prompt = "x" * (server_module.MAX_PROMPT_LENGTH + 1)
    response = client.post("/generate", json={"prompt": oversized_prompt})

    assert response.status_code == 400
    assert "too long" in response.get_json()["error"]


# /generate should return 503 when the runtime's model is not ready.
def test_generate_returns_503_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=False))
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "Deep bass"})

    assert response.status_code == 503


# /generate should return 400 when the tokenized prompt exceeds the model's token context.
def test_generate_rejects_prompt_that_exceeds_token_context(monkeypatch):
    runtime = FakeRuntime(ready=True)
    runtime.tokenizer = LongTokenizer()
    runtime.tokenizer_max_length = 32
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: runtime)
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "A" * 100})

    assert response.status_code == 400
    assert "token context" in response.get_json()["error"]


# /generate should label output values using the runtime's own param_keys, not a fixed list.
def test_generate_uses_runtime_parameter_keys(monkeypatch):
    runtime = FakeRuntime(ready=True)
    runtime.param_keys = ("cutoff", "attack")
    runtime.model = lambda input_ids, attention_mask: torch.tensor([[0.2, 0.9]])
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: runtime)
    client = server_module.app.test_client()

    response = client.post("/generate", json={"prompt": "short"})

    assert response.status_code == 200
    assert response.get_json()["parameters"] == {"cutoff": 0.2, "attack": 0.9}


CURRENT_PRESET = {name: 0.5 for name in server_module.PARAM_KEYS}


# /refine-preset should keep locked parameters at their current value and regenerate the others.
def test_refine_keeps_locked_parameters(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()
    generated = client.post("/generate", json={"prompt": "Bright lead"}).get_json()["parameters"]

    locked = ["filter_cutoff", "reverb_mix"]
    current = dict(CURRENT_PRESET, filter_cutoff=0.123, reverb_mix=0.987)
    response = client.post(
        "/refine-preset",
        json={"prompt": "Bright lead", "parameters": current, "locked_params": locked},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["parameters"]["filter_cutoff"] == 0.123
    assert body["parameters"]["reverb_mix"] == 0.987
    unlocked = [name for name in server_module.PARAM_KEYS if name not in locked]
    assert {name: body["parameters"][name] for name in unlocked} == {name: generated[name] for name in unlocked}
    assert body["values"] == [body["parameters"][name] for name in server_module.PARAM_KEYS]


# /ai/refine-preset should answer in the same charter format as /ai/generate-preset.
def test_refine_alias_returns_charter_format(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post(
        "/ai/refine-preset",
        json={"prompt": "Darker pad", "parameters": CURRENT_PRESET, "locked_params": []},
    )

    assert response.status_code == 200
    body = response.get_json()
    generated = client.post("/ai/generate-preset", json={"prompt": "Darker pad"}).get_json()
    assert body.keys() == generated.keys()
    assert body["metadata"] == generated["metadata"]
    assert body["parameters"] == generated["parameters"]


# /refine-preset should ignore locked ids the model does not know or the current preset does not contain.
def test_refine_ignores_unknown_locked_parameters(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()
    generated = client.post("/generate", json={"prompt": "Pluck"}).get_json()["parameters"]

    current = {name: value for name, value in CURRENT_PRESET.items() if name != "lfo_rate"}
    response = client.post(
        "/refine-preset",
        json={"prompt": "Pluck", "parameters": current, "locked_params": ["not_a_param", "lfo_rate"]},
    )

    assert response.status_code == 200
    assert response.get_json()["parameters"] == generated


# /refine-preset should return 400 for an empty or whitespace-only prompt.
def test_refine_rejects_empty_prompt(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/refine-preset", json={"prompt": "   ", "parameters": CURRENT_PRESET})

    assert response.status_code == 400
    assert "No prompt provided" in response.get_json()["error"]


# /refine-preset should return 400 when the current preset or the locked ids are malformed.
@pytest.mark.parametrize(
    "extra, message",
    [
        ({}, "'parameters' must be an object"),
        ({"parameters": [0.5, 0.5]}, "'parameters' must be an object"),
        ({"parameters": {"filter_cutoff": "high"}}, "must be a number"),
        ({"parameters": {"filter_cutoff": True}}, "must be a number"),
        ({"parameters": {}, "locked_params": "filter_cutoff"}, "must be a list"),
        ({"parameters": {}, "locked_params": [3]}, "must be a list"),
    ],
)
def test_refine_rejects_invalid_payload(monkeypatch, extra, message):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=True))
    client = server_module.app.test_client()

    response = client.post("/refine-preset", json={"prompt": "Pad", **extra})

    assert response.status_code == 400
    assert message in response.get_json()["error"]


# /refine-preset should fall back to the default model for an unknown model id, like /generate.
def test_refine_unknown_model_uses_default(monkeypatch):
    requested = []

    def fake_get_runtime(model_key=None):
        requested.append(model_key)
        return FakeRuntime(ready=True)

    monkeypatch.setattr(server_module, "_get_runtime", fake_get_runtime)
    monkeypatch.setattr(server_module, "_discover_models", lambda: {"charter_v1": {}})
    client = server_module.app.test_client()

    unknown = client.post(
        "/refine-preset",
        json={"prompt": "Pad", "parameters": CURRENT_PRESET, "model_name": "does_not_exist"},
    )
    known = client.post(
        "/refine-preset",
        json={"prompt": "Pad", "parameters": CURRENT_PRESET, "model_name": "charter_v1"},
    )

    assert unknown.status_code == 200
    assert known.status_code == 200
    assert requested == [server_module.DEFAULT_MODEL_KEY, "charter_v1"]


# /refine-preset should return 503 when the runtime's model is not ready.
def test_refine_returns_503_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(server_module, "_get_runtime", lambda model_key=None: FakeRuntime(ready=False))
    client = server_module.app.test_client()

    response = client.post("/refine-preset", json={"prompt": "Deep bass", "parameters": CURRENT_PRESET})

    assert response.status_code == 503


# /metrics/latest should return the most recent benchmark entry and its evaluation report.
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


# /metrics/latest should return 404 when the benchmark history file does not exist.
def test_metrics_latest_returns_404_when_missing(monkeypatch, tmp_path):
    missing_history = tmp_path / "missing-history.json"
    monkeypatch.setattr(server_module, "BENCHMARK_FILE", missing_history)

    client = server_module.app.test_client()
    response = client.get("/metrics/latest")

    assert response.status_code == 404


