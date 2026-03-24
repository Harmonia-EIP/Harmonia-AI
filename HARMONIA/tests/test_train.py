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

