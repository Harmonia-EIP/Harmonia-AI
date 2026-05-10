import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from src.charter import (
    BIPOLAR_INDICES,
    DISCRETE_INDICES,
    PARAM_NAMES,
    snap_discrete,
)


CHARTER_NAME_SET = set(PARAM_NAMES)


class PresetDataset(Dataset):
    def __init__(
        self,
        data_file,
        tokenizer,
        tokenizer_max_length=32,
        normalize_categorical=True,
        prompt_augment=False,
    ):
        self.data_file = Path(data_file)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.normalize_categorical = bool(normalize_categorical)
        self.prompt_augment = bool(prompt_augment)

        records = self._load_records(self.data_file)
        if not records:
            self.samples = []
            self.param_keys = []
            self.training_param_keys = []
            self._continuous_keys = []
            self._binary_keys = []
            self._categorical_keys = []
            self._categorical_max = {}
            self.charter_mode = False
            return

        (
            self._continuous_keys,
            self._binary_keys,
            self._categorical_keys,
        ) = self._extract_group_keys(records)

        all_keys = set(self._continuous_keys) | set(self._binary_keys) | set(self._categorical_keys)
        self.charter_mode = bool(all_keys) and all_keys.issubset(CHARTER_NAME_SET)

        if self.charter_mode:
            self.param_keys = list(PARAM_NAMES)
            self.training_param_keys = list(PARAM_NAMES)
        else:
            self.param_keys = self._continuous_keys + self._binary_keys + self._categorical_keys
            self.training_param_keys = self._continuous_keys + self._binary_keys

        self._categorical_max = self._compute_categorical_max(records)

        self.samples = [
            {
                "prompt": self._extract_prompt(item),
                "labels": self._flatten_parameters(item),
            }
            for item in records
        ]

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported JSON dataset format in {path}")

    def _load_records(self, path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            if np is None:
                raise RuntimeError("numpy is required to load .npy datasets")
            loaded = np.load(path, allow_pickle=True)
            if isinstance(loaded, np.ndarray):
                payload = loaded.tolist()
            else:
                payload = loaded
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [payload]
            raise ValueError(f"Unsupported NPY dataset format in {path}")

        return self._read_json(path)

    @staticmethod
    def _extract_prompt(item):
        for key in ("raw_text", "description", "name"):
            value = item.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _as_numeric_dict(payload):
        if not isinstance(payload, dict):
            return {}

        numeric = {}
        for key, value in payload.items():
            try:
                numeric[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return numeric

    def _extract_group_keys(self, records):
        continuous = set()
        binary = set()
        categorical = set()

        for item in records:
            params = item.get("parameters")
            if isinstance(params, dict):
                continuous.update(self._as_numeric_dict(params.get("continuous", {})).keys())
                binary.update(self._as_numeric_dict(params.get("binary", {})).keys())
                categorical.update(self._as_numeric_dict(params.get("categorical", {})).keys())
            elif isinstance(params, list):
                for idx in range(len(params)):
                    continuous.add(f"param_{idx:04d}")

        continuous_keys = sorted(continuous)
        binary_keys = sorted(binary)
        categorical_keys = sorted(categorical)
        return continuous_keys, binary_keys, categorical_keys

    def _compute_categorical_max(self, records):
        if not self._categorical_keys:
            return {}

        max_values = {k: 1.0 for k in self._categorical_keys}
        for item in records:
            params = item.get("parameters")
            if not isinstance(params, dict):
                continue

            categorical = self._as_numeric_dict(params.get("categorical", {}))
            for key in self._categorical_keys:
                value = categorical.get(key)
                if value is None:
                    continue
                max_values[key] = max(max_values[key], float(value))

        return max_values

    def _flatten_parameters(self, item):
        params = item.get("parameters", {})
        if isinstance(params, list):
            vector = [float(v) for v in params]
            if len(vector) < len(self.param_keys):
                vector.extend([0.0] * (len(self.param_keys) - len(vector)))
            return torch.tensor(vector[: len(self.param_keys)], dtype=torch.float32)

        continuous = self._as_numeric_dict(params.get("continuous", {}))
        binary = self._as_numeric_dict(params.get("binary", {}))
        categorical = self._as_numeric_dict(params.get("categorical", {}))

        if self.charter_mode:
            vector = []
            for idx, key in enumerate(PARAM_NAMES):
                if key in continuous:
                    value = continuous[key]
                elif key in categorical:
                    raw = categorical[key]
                    divisor = self._categorical_max.get(key, 1.0) or 1.0
                    if self.normalize_categorical and divisor > 1.0:
                        raw = raw / divisor
                    value = snap_discrete(raw, idx) if idx in DISCRETE_INDICES else raw
                else:
                    value = 0.5 if idx in BIPOLAR_INDICES else 0.0
                vector.append(max(0.0, min(1.0, float(value))))
            return torch.tensor(vector, dtype=torch.float32)

        vector = []
        vector.extend(continuous.get(k, 0.0) for k in self._continuous_keys)
        vector.extend(binary.get(k, 0.0) for k in self._binary_keys)
        for key in self._categorical_keys:
            value = categorical.get(key, 0.0)
            if self.normalize_categorical:
                divisor = self._categorical_max.get(key, 1.0) or 1.0
                value = value / divisor
            vector.append(value)

        return torch.tensor(vector, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _augment_prompt(self, prompt):
        if not self.prompt_augment or not prompt:
            return prompt
        import random

        prefixes = ("", "a ", "an ", "warm ", "bright ", "deep ", "soft ", "hard ", "")
        suffixes = ("", " preset", " sound", " patch", "", "")
        return f"{random.choice(prefixes)}{prompt}{random.choice(suffixes)}".strip()  # nosec B311

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample.get("prompt", "") if isinstance(sample, dict) else ""
        labels = sample.get("labels") if isinstance(sample, dict) else None
        if labels is None:
            labels = torch.zeros(len(self.param_keys), dtype=torch.float32)

        prompt = self._augment_prompt(prompt)

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }
