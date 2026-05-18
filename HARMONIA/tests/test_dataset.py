import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

PresetDataset = import_module("src.dataset").PresetDataset


class DummyTokenizer:
    def __call__(self, text, padding, max_length, truncation, return_tensors):
        _ = (text, padding, truncation, return_tensors)
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_preset_dataset_loads_npy_and_exposes_param_keys(tmp_path):
    payload = [
        {
            "name": "Preset A",
            "raw_text": "Bright lead",
            "parameters": {
                "continuous": {"AmpEnv A Attack": 0.2, "AmpEnv A Decay": 0.5},
                "binary": {"Osc A1 Invert": 1.0},
                "categorical": {"Filter A Type": 1},
            },
        },
        {
            "name": "Preset B",
            "raw_text": "Dark bass",
            "parameters": {
                "continuous": {"AmpEnv A Attack": 0.7, "AmpEnv A Decay": 0.1},
                "binary": {"Osc A1 Invert": 0.0},
                "categorical": {"Filter A Type": 3},
            },
        },
    ]
    dataset_path = tmp_path / "presets.npy"
    np.save(dataset_path, np.array(payload, dtype=object))

    dataset = PresetDataset(dataset_path, DummyTokenizer(), tokenizer_max_length=16)

    assert dataset.param_keys == [
        "AmpEnv A Attack",
        "AmpEnv A Decay",
        "Osc A1 Invert",
        "Filter A Type",
    ]
    item = dataset[0]
    assert item["input_ids"].shape[0] == 16
    assert item["labels"].shape[0] == 4


def test_preset_dataset_normalizes_categorical_values(tmp_path):
    payload = [
        {
            "raw_text": "pad",
            "parameters": {
                "continuous": {"AmpEnv A Attack": 0.4},
                "binary": {},
                "categorical": {"Arp Mode": 2},
            },
        },
        {
            "raw_text": "bass",
            "parameters": {
                "continuous": {"AmpEnv A Attack": 0.1},
                "binary": {},
                "categorical": {"Arp Mode": 4},
            },
        },
    ]
    dataset_path = tmp_path / "presets.npy"
    np.save(dataset_path, np.array(payload, dtype=object))

    dataset = PresetDataset(dataset_path, DummyTokenizer(), normalize_categorical=True)

    labels_first = dataset[0]["labels"].tolist()
    labels_second = dataset[1]["labels"].tolist()
    assert labels_first[-1] == 0.5
    assert labels_second[-1] == 1.0

