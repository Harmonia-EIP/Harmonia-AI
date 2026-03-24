import json

import torch

from scripts import server as server_module
from scripts.prepare_dataset import TARGET_PARAMS, convert_fxp_dump_to_json


class FakeTokenizer:
    def __call__(self, prompt, return_tensors, padding=False, truncation=False, max_length=None):
        _ = (prompt, return_tensors, padding, truncation, max_length)
        return {
            "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        _ = (input_ids, attention_mask)
        return torch.tensor([[0.5] * 9])


class FakeRuntime:
    def __init__(self, ready=True):
        self.model = FakeModel() if ready else None
        self.tokenizer = FakeTokenizer() if ready else None
        self.ready = ready
        self.error = ""
        self.model_version = "smoke-v1"
        self.model_hash = "smoke-hash"
        self.param_keys = tuple(server_module.PARAM_KEYS)
        self.tokenizer_max_length = 32


def test_prepare_and_generate_smoke(tmp_path, monkeypatch):
    raw_dump = tmp_path / "my_raw_dump.txt"
    processed = tmp_path / "presets.json"

    params = {key: 0.4 for key in TARGET_PARAMS}
    params["description"] = "Smoke_Preset.fxp"
    raw_dump.write_text(str(("smoke", params)), encoding="utf-8")

    convert_fxp_dump_to_json(raw_dump, processed)

    payload = json.loads(processed.read_text(encoding="utf-8"))
    assert payload and payload[0]["description"] == "Smoke Preset"

    monkeypatch.setattr(server_module, "_get_runtime", lambda: FakeRuntime(ready=True))
    client = server_module.app.test_client()
    response = client.post("/generate", json={"prompt": "smoke test"})

    assert response.status_code == 200
    body = response.get_json()
    assert "parameters" in body
    assert body["metadata"]["model_version"] == "smoke-v1"

