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
    monkeypatch.setattr(server_module, "V4_MODEL_DIR", tmp_path / "no_v4")  # a locally installed v4 is not part of these tests

    def runtime(model_key=server_module.V3_MODEL_KEY):
        return server_module.InferenceRuntime(
            model=None, tokenizer=None, ready=True, model_version=model_key, model_hash="abc",
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
    monkeypatch.setattr(server_module, "V4_MODEL_DIR", tmp_path / "missing_v4")
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
    monkeypatch.setattr(server_module, "V4_MODEL_DIR", tmp_path / "missing_v4")
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


# The warm-up loads the model once, plays a few sounds and leaves the server answering.
def test_warm_up_prepares_v3(v3_client):
    server_module._warm_up_v3()
    runtime = server_module._get_runtime(server_module.V3_MODEL_KEY)
    assert runtime.v3.cost > 0.0
    assert v3_client.post("/generate", json={"prompt": "door slam", "model_name": "harmonia_v3"}).status_code == 200


def test_env_numbers_fall_back_on_bad_values(monkeypatch):
    monkeypatch.setenv("HARMONIA_V3_TIME_BUDGET", "12.5")
    assert server_module._env_float("HARMONIA_V3_TIME_BUDGET", 0.0) == 12.5
    monkeypatch.setenv("HARMONIA_V3_TIME_BUDGET", "soon")
    assert server_module._env_float("HARMONIA_V3_TIME_BUDGET", 3.0) == 3.0
    monkeypatch.setenv("HARMONIA_V3_CANDIDATES", "x")
    assert server_module._env_int("HARMONIA_V3_CANDIDATES", 8) == 8


# harmonia_v4: the same run time with other weights, offered next to harmonia_v3 when its files are there.
@pytest.mark.parametrize("payload", [{"model_name": "harmonia_v4"}, {"model_name": "model-5"}, {"model_id": 5}, {"model": "v4"}])
def test_v4_aliases(payload):
    assert server_module._resolve_model_key(payload) == server_module.V4_MODEL_KEY


def test_v4_is_listed_served_and_warmed_up(v3_client, tmp_path, monkeypatch):
    from tests.v3_fixtures import build_model_dir

    monkeypatch.setattr(server_module, "SAVED_MODELS_DIR", tmp_path / "saved_models")
    monkeypatch.setattr(server_module, "V2_MODEL_DIR", tmp_path / "no_v2")
    assert server_module._warm_up_key() == server_module.V3_MODEL_KEY
    build_model_dir(tmp_path / "harmonia_v4")
    monkeypatch.setattr(server_module, "V4_MODEL_DIR", tmp_path / "harmonia_v4")
    by_id = {m["id"]: m for m in v3_client.get("/models").get_json()["models"]}
    assert set(by_id) == {"harmonia_v3", "harmonia_v4"}
    assert by_id["harmonia_v4"]["name"] == "Harmonia V4" and by_id["harmonia_v4"]["is_latest"] is False
    body = v3_client.post("/generate", json={"prompt": "porte qui claque", "model_name": "harmonia_v4"}).get_json()
    assert body["metadata"]["model_version"] == "harmonia_v4"
    assert charter_v3.validate_preset(body) == []
    v3 = v3_client.post("/generate", json={"prompt": "porte qui claque", "model_name": "harmonia_v3"}).get_json()
    assert v3["metadata"]["model_version"] == "harmonia_v3"
    assert server_module._warm_up_key() == server_module.V4_MODEL_KEY  # the newest model is prepared at start-up


# Models shipping the same CLAP audio tower share one copy in memory.
def test_listener_is_shared(tmp_path, monkeypatch):
    loaded = []

    class FakeListener:
        def __init__(self, model_dir):
            loaded.append(model_dir)

    import src.v3.listener as listener_module

    monkeypatch.setattr(listener_module, "ClapListener", FakeListener)
    monkeypatch.setattr(server_module, "_LISTENERS", {})
    same = {"files": {"clap_audio/model.safetensors": {"sha256": "aa"}}}
    first = server_module._shared_listener(tmp_path / "v3", same)
    assert server_module._shared_listener(tmp_path / "v4", same) is first
    other = server_module._shared_listener(tmp_path / "v5", {"files": {"clap_audio/model.safetensors": {"sha256": "bb"}}})
    assert other is not first and len(loaded) == 2


# A model whose generator plays recorded instruments is not served without the listen pack: it would hear
# them as silence.
def test_model_with_samples_needs_the_listen_pack(tmp_path, monkeypatch):
    import json

    from src.samples import player

    folder = tmp_path / "harmonia_v3"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"model_version": "harmonia_v3.9", "files": {},
                                                      "samples": {"pack": "listen", "release": "samples-1.0.0"}}))
    monkeypatch.setattr(server_module, "V3_MODEL_DIR", folder)
    monkeypatch.setattr(player, "default_bank", lambda: None)
    runtime = server_module._build_v3_runtime(server_module.V3_MODEL_KEY)
    assert not runtime.ready and "fetch_pack.py --pack listen" in runtime.error
