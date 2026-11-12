"""The harmonia_v3 model behind /generate and /refine-preset (scripts/server.py)."""

import pytest

from scripts import server as server_module
from src import charter_v3
from tests.v3_fixtures import build_engine


@pytest.fixture()
def v3_client(tmp_path, monkeypatch):
    engine = build_engine(tmp_path / "harmonia_v3")
    monkeypatch.setenv("HARMONIA_PUSH_METRICS", "0")
    monkeypatch.setattr(server_module, "V3_MODEL_DIR", tmp_path / "harmonia_v3")

    def runtime():
        return server_module.InferenceRuntime(
            model=None, tokenizer=None, ready=True, model_version="harmonia_v3", model_hash="abc",
            param_keys=tuple(p.id for p in charter_v3.load().params), v3=engine,
        )

    monkeypatch.setattr(server_module, "_build_v3_runtime", runtime)
    server_module._get_runtime_cached.cache_clear()
    yield server_module.app.test_client()
    server_module._get_runtime_cached.cache_clear()


def plugin_preset(parameters: dict) -> dict:
    full = {p.id: p.to_normalized(p.default) for p in charter_v3.load().params}
    full.update(parameters)
    return full


# The values a model selector may send reach the v3 model.
@pytest.mark.parametrize("payload", [{"model_name": "harmonia_v3"}, {"model_name": "model-4"}, {"model_id": 4}, {"model": "v3"}])
def test_v3_aliases(payload):
    assert server_module._resolve_model_key(payload) == server_module.V3_MODEL_KEY


# /generate with the v3 model answers a charter-v3 preset the plugin's PresetLoader reads.
def test_generate_with_v3(v3_client):
    response = v3_client.post("/ai/generate-preset", json={"prompt": "porte qui claque", "model_name": "harmonia_v3"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["metadata"]["charter_version"].startswith("3")
    assert body["metadata"]["model_version"] == "harmonia_v3"
    assert body["metadata"]["name"] == "porte qui claque"
    assert charter_v3.validate_preset(body) == []
    assert body["generation"]["engine"] == body["metadata"]["engine"]
    again = v3_client.post("/generate", json={"prompt": "porte qui claque", "model_name": "harmonia_v3"}).get_json()
    assert again["parameters"] == body["parameters"]
    other = v3_client.post("/generate", json={"prompt": "porte qui claque", "model_name": "harmonia_v3", "variation": 2})
    assert other.get_json()["parameters"] != body["parameters"]


# /refine-preset with the v3 model starts from the plugin's preset: same engine, locked parameters untouched.
def test_refine_with_v3_keeps_locked_parameters(v3_client):
    current = plugin_preset({"engine": 0.0, "filter_cutoff": 0.123456789, "reverb_mix": 0.25, "osc_mix": 0.5})
    response = v3_client.post(
        "/ai/refine-preset",
        json={"prompt": "basse ronflante", "model_name": "harmonia_v3", "parameters": current,
              "locked_params": ["filter_cutoff", "reverb_mix", "not_a_parameter"], "strength": 0.6},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert charter_v3.validate_preset(body) == []
    assert body["metadata"]["engine"] == "analog"
    assert body["parameters"]["filter_cutoff"] == current["filter_cutoff"]
    assert body["parameters"]["reverb_mix"] == current["reverb_mix"]
    assert body["generation"]["strength"] == 0.6
    assert body["generation"]["similarity"] >= body["generation"]["similarity_before"]


# Invalid v3 options are rejected with 400.
@pytest.mark.parametrize("extra", [{"variation": -1}, {"variation": "abc"}, {"variation": True}, {"strength": 2}, {"strength": "a"}])
def test_v3_rejects_invalid_options(v3_client, extra):
    response = v3_client.post(
        "/refine-preset", json={"prompt": "door slam", "model_name": "harmonia_v3", "parameters": {}, **extra}
    )
    assert response.status_code == 400
    if "variation" in extra:
        response = v3_client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v3", **extra})
        assert response.status_code == 400


# Without downloaded model files, v3 answers 503 instead of crashing.
def test_v3_missing_files_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "V3_MODEL_DIR", tmp_path / "missing")
    server_module._get_runtime_cached.cache_clear()
    try:
        client = server_module.app.test_client()
        assert client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v3"}).status_code == 503
        refine = client.post("/refine-preset", json={"prompt": "door slam", "model_name": "harmonia_v3", "parameters": {}})
        assert refine.status_code == 503
    finally:
        server_module._get_runtime_cached.cache_clear()


# GET /models lists harmonia_v3 when its files are there; HARMONIA_DEFAULT_MODEL makes it the default model.
def test_models_lists_v3_and_default_override(v3_client, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "SAVED_MODELS_DIR", tmp_path / "saved_models")
    monkeypatch.setattr(server_module, "V2_MODEL_DIR", tmp_path / "no_v2")
    (tmp_path / "saved_models").mkdir()
    (tmp_path / "saved_models" / "model_charter_v1.pth").write_bytes(b"")
    by_id = {m["id"]: m for m in v3_client.get("/models").get_json()["models"]}
    assert set(by_id) == {"charter_v1", "harmonia_v3"}
    assert by_id["harmonia_v3"]["is_latest"] is False
    assert server_module._resolve_model_key({}) == server_module.DEFAULT_MODEL_KEY

    monkeypatch.setattr(server_module, "DEFAULT_MODEL_OVERRIDE", "harmonia_v3")
    by_id = {m["id"]: m for m in v3_client.get("/models").get_json()["models"]}
    assert by_id["harmonia_v3"]["is_latest"] is True and by_id["charter_v1"]["is_latest"] is False
    assert server_module._resolve_model_key({}) == server_module.V3_MODEL_KEY
    assert server_module._resolve_model_key({"model_name": "latest"}) == server_module.V3_MODEL_KEY
    body = v3_client.post("/generate", json={"prompt": "door slam"}).get_json()
    assert body["metadata"]["model_version"] == "harmonia_v3"


# The override is ignored while the model's files are missing: the default stays the latest .pth.
def test_default_override_needs_installed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "V3_MODEL_DIR", tmp_path / "missing")
    monkeypatch.setattr(server_module, "DEFAULT_MODEL_OVERRIDE", "harmonia_v3")
    assert server_module._resolve_model_key({}) == server_module.DEFAULT_MODEL_KEY


# GET /charter/v3 serves the charter the v3 presets follow.
def test_charter_v3_route(v3_client):
    body = v3_client.get("/charter/v3").get_json()
    assert body["version"] == charter_v3.load().version
    assert "engines" in body


# A comparative in the refine prompt is followed as a direction and reported in `generation`.
def test_refine_with_v3_follows_a_comparative(v3_client):
    current = plugin_preset({"engine": 0.0, "osc_1_waveform": 2 / 3, "filter_cutoff": 0.5})
    body = v3_client.post(
        "/refine-preset", json={"prompt": "Plus sombre", "model_name": "harmonia_v3", "parameters": current}
    ).get_json()
    assert body["generation"]["directives"] == ["brightness-"]
    assert body["parameters"]["filter_cutoff"] < current["filter_cutoff"]
    assert charter_v3.validate_preset(body) == []
