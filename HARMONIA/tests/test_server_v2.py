import pytest

from scripts import server as server_module
from src.charter import PARAM_NAMES
from tests.v2_fixtures import build_model_dir


@pytest.fixture()
def v2_client(tmp_path, monkeypatch):
    build_model_dir(tmp_path / "harmonia_v2")
    monkeypatch.setenv("HARMONIA_PUSH_METRICS", "0")
    monkeypatch.setattr(server_module, "V2_MODEL_DIR", tmp_path / "harmonia_v2")
    server_module._get_runtime_cached.cache_clear()
    yield server_module.app.test_client()
    server_module._get_runtime_cached.cache_clear()


# The app's model selector values reach the v2 model.
@pytest.mark.parametrize("payload", [{"model_name": "harmonia_v2"}, {"model_name": "model-3"}, {"model_id": 3}, {"model": "v2"}])
def test_v2_aliases(payload):
    assert server_module._resolve_model_key(payload) == server_module.V2_MODEL_KEY


# /generate with the v2 model returns the same charter payload as v1, plus the generation details.
def test_generate_with_v2(v2_client):
    response = v2_client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v2", "mode": "retrieval"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["metadata"]["model_version"] == "harmonia_v2"
    assert set(body["parameters"]) == set(PARAM_NAMES)
    assert len(body["values"]) == len(PARAM_NAMES)
    assert body["generation"]["mode"] == "retrieval"
    assert body["generation"]["bank_index"] == 0


# French prompts longer than the v1 token limit are accepted by v2 (truncated, not rejected).
def test_v2_accepts_long_prompts(v2_client):
    prompt = "porte qui claque dans une cathédrale immense avec une très longue réverbération " * 3
    response = v2_client.post("/generate", json={"prompt": prompt, "model_name": "harmonia_v2"})
    assert response.status_code == 200


# Without `mode`, the model's default mode (from its manifest) is used.
def test_v2_default_mode(v2_client):
    body = v2_client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v2"}).get_json()
    assert body["generation"]["mode"] == "retrieval"


# Invalid v2 options are rejected with 400.
@pytest.mark.parametrize("extra", [{"mode": "magic"}, {"variation": "abc"}, {"variation": True}])
def test_v2_rejects_invalid_options(v2_client, extra):
    response = v2_client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v2", **extra})
    assert response.status_code == 400


# Without downloaded model files, v2 answers 503 instead of crashing.
def test_v2_missing_files_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "V2_MODEL_DIR", tmp_path / "missing")
    server_module._get_runtime_cached.cache_clear()
    try:
        response = server_module.app.test_client().post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v2"})
        assert response.status_code == 503
    finally:
        server_module._get_runtime_cached.cache_clear()


# GET /models lists harmonia_v2 next to the .pth models of saved_models/, so the app's model menu offers it.
def test_models_lists_v2(v2_client, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "SAVED_MODELS_DIR", tmp_path / "saved_models")
    (tmp_path / "saved_models").mkdir()
    (tmp_path / "saved_models" / "model_charter_v1.pth").write_bytes(b"")
    body = v2_client.get("/models").get_json()
    assert body["count"] == 2
    by_id = {m["id"]: m for m in body["models"]}
    assert set(by_id) == {"charter_v1", "harmonia_v2"}
    assert by_id["harmonia_v2"]["model_version"] == "harmonia_v2"
    assert by_id["harmonia_v2"]["is_latest"] is False


# Without its files, harmonia_v2 is not offered.
def test_models_without_v2_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "V2_MODEL_DIR", tmp_path / "missing")
    monkeypatch.setattr(server_module, "SAVED_MODELS_DIR", tmp_path / "saved_models")
    body = server_module.app.test_client().get("/models").get_json()
    assert body == {"count": 0, "models": []}
