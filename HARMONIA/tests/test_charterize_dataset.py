import json
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.charterize_dataset import (
    _extract_description,
    _flatten_sylenth_parameters,
    charterize,
)
from src.charter import PARAM_NAMES


def _sample_sylenth_record() -> dict:
    return {
        "name": "Acid Bass",
        "raw_text": "Acid Bass",
        "parameters": {
            "continuous": {
                "AmpEnv A Attack": 0.0,
                "AmpEnv A Decay": 0.3,
                "AmpEnv A Sustain": 0.0,
                "AmpEnv A Release": 0.15,
                "Filter A Cutoff": 0.45,
                "Filter A Reso": 0.85,
                "Osc A2 Volume": 0.0,
                "Osc A2 Detune": 0.0,
                "ModEnv 1 Decay": 0.4,
                "LFO 1 Rate": 0.5,
                "LFO 1 Gain": 0.0,
                "FilterCtl KeyTrk": 0.6,
                "Distort DryWet": 0.4,
                "Reverb Dry/Wet": 0.1,
            },
            "binary": {"Sw DistOnOff": 1.0, "Sw ReverbOnOff": 1.0},
            "categorical": {
                "Osc A1 Waveform": 0.667,
                "Osc A2 Waveform": 0.0,
                "Filter A Type": 0.0,
            },
        },
    }


def test_flatten_merges_three_groups():
    flat = _flatten_sylenth_parameters(_sample_sylenth_record()["parameters"])
    assert "AmpEnv A Attack" in flat
    assert "Osc A1 Waveform" in flat
    assert "Sw DistOnOff" in flat


def test_flatten_accepts_already_flat():
    flat = _flatten_sylenth_parameters({"Filter A Cutoff": 0.4, "AmpEnv A Attack": 0.0})
    assert flat == {"Filter A Cutoff": 0.4, "AmpEnv A Attack": 0.0}


def test_extract_description_strips_fxp_and_underscores():
    desc = _extract_description({"name": "Acid_Bass.fxp"})
    assert desc == "Acid Bass"


def test_charterize_npy_round_trip(tmp_path):
    src = tmp_path / "src.npy"
    out = tmp_path / "out.npy"
    np.save(src, np.array([_sample_sylenth_record()], dtype=object))

    stats = charterize(src, out, anchor_path=None, verbose=False)

    assert stats["input_records"] == 1
    assert stats["converted"] == 1
    assert stats["skipped"] == 0
    assert stats["anchors_added"] == 0
    assert stats["output_records"] == 1
    assert stats["profile_name"] in {"sylenth1", "legacy"}
    payload = np.load(out, allow_pickle=True).tolist()
    assert len(payload) == 1

    record = payload[0]
    assert set(record.keys()) == {"raw_text", "description", "parameters"}
    params = record["parameters"]
    flat = {**params["continuous"], **params["categorical"]}
    assert set(flat) == set(PARAM_NAMES)
    assert all(0.0 <= v <= 1.0 for v in flat.values())


def test_charterize_json_output_format(tmp_path):
    src = tmp_path / "src.npy"
    out = tmp_path / "out.json"
    np.save(src, np.array([_sample_sylenth_record()], dtype=object))

    charterize(src, out, anchor_path=None, verbose=False)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert "parameters" in payload[0]


def test_charterize_appends_anchors(tmp_path):
    src = tmp_path / "src.npy"
    out = tmp_path / "out.npy"
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps([
            {
                "label": "TestAnchor",
                "prompts": ["test anchor prompt"],
                "parameters": {
                    "continuous": {"filter_cutoff": 0.5},
                    "categorical": {"osc_1_waveform": 0.667},
                },
            }
        ]),
        encoding="utf-8",
    )
    np.save(src, np.array([_sample_sylenth_record()], dtype=object))

    stats = charterize(src, out, anchor_path=anchors, verbose=False)
    assert stats["anchors_added"] == 1
    assert stats["output_records"] == 2
