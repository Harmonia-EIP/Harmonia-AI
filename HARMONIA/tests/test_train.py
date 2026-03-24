import json

import torch

from scripts import train as train_module


def test_set_seed_is_deterministic_for_torch():
    train_module.set_seed(123)
    first = torch.rand(4)

    train_module.set_seed(123)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_save_benchmark_adds_seed_and_dataset_size(tmp_path, monkeypatch):
    benchmark_file = tmp_path / "history.json"

    monkeypatch.setattr(train_module, "BENCHMARK_FILE", benchmark_file)
    monkeypatch.setattr(train_module, "SEED", 999)

    train_module.save_benchmark(
        duration=2.5,
        final_loss=0.12,
        epoch_history=[0.8, 0.4, 0.12],
        dataset_size=5,
    )

    payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["seed"] == 999
    assert payload[0]["dataset_size"] == 5
    assert payload[0]["final_loss"] == 0.12


def test_compute_split_sizes_handles_small_and_large_datasets():
    assert train_module.compute_split_sizes(1, 0.2) == (1, 0)
    assert train_module.compute_split_sizes(10, 0.2) == (8, 2)
    assert train_module.compute_split_sizes(2, 0.9) == (1, 1)


def test_write_evaluation_report_creates_json(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "EVAL_REPORT_DIR", tmp_path)

    report_path = train_module.write_evaluation_report(
        {
            "model_version": "test-v2",
            "metrics": {"mse": 0.12, "mae": 0.24},
        }
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "test-v2"


def test_ensure_unique_model_version_adds_suffix_when_exists(tmp_path, monkeypatch):
    (tmp_path / "v1").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", tmp_path)

    assert train_module.ensure_unique_model_version("v1") == "v1-r2"


def test_compute_file_sha256_has_expected_length(tmp_path):
    file_path = tmp_path / "weights.bin"
    file_path.write_bytes(b"abc")

    digest = train_module.compute_file_sha256(file_path)
    assert len(digest) == 64


