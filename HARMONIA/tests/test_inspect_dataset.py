import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts import inspect_dataset
from src.charter import PARAM_NAMES


def _fake_sylenth_dataset(tmp_path: Path) -> Path:
    record = {
        "name": "Acid Bass",
        "raw_text": "Acid Bass",
        "parameters": {
            "continuous": {
                "AmpEnv A Attack": 0.0,
                "AmpEnv A Decay": 0.3,
                "AmpEnv A Sustain": 0.0,
                "AmpEnv A Release": 0.1,
                "Filter A Cutoff": 0.45,
                "Filter A Reso": 0.8,
                "Osc A2 Volume": 0.2,
                "Osc A2 Detune": 0.4,
                "ModEnv 1 Decay": 0.3,
                "LFO 1 Rate": 0.5,
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
    path = tmp_path / "dataset.npy"
    np.save(path, np.array([record], dtype=object))
    return path


def test_tokens_strips_punctuation():
    assert inspect_dataset._tokens("Osc A1 Waveform") == ["osc", "a1", "waveform"]
    assert inspect_dataset._tokens("Distort Dry/Wet!") == ["distort", "dry", "wet"]


def test_score_rewards_token_overlap():
    direct = inspect_dataset._score("AmpEnv A Attack", "ampenv attack")
    irrelevant = inspect_dataset._score("Phaser CenterFreq", "ampenv attack")
    assert direct > irrelevant
    assert direct > 0.55


def test_collect_keys_discovers_native_keys(tmp_path):
    path = _fake_sylenth_dataset(tmp_path)
    counts = inspect_dataset.collect_keys(path, sample_limit=5)
    assert "Filter A Cutoff" in counts
    assert "Osc A1 Waveform" in counts


def test_suggest_for_param_returns_top_candidates():
    keys = ["Filter A Cutoff", "Filter B Cutoff", "Phaser CenterFreq", "Junk"]
    suggestions = inspect_dataset.suggest_for_param("filter_cutoff", keys, top=2)
    assert suggestions, "should find at least one candidate"
    assert suggestions[0][0] == "Filter A Cutoff"
    assert len(suggestions) <= 2


def test_build_profile_skeleton_covers_all_charter_params():
    keys = Counter({
        "Osc A1 Waveform": 1,
        "Osc A2 Waveform": 1,
        "Osc A2 Volume": 1,
        "Osc A2 Detune": 1,
        "Filter A Cutoff": 1,
        "Filter A Reso": 1,
        "Filter A Type": 1,
        "AmpEnv A Attack": 1,
        "AmpEnv A Decay": 1,
        "AmpEnv A Sustain": 1,
        "AmpEnv A Release": 1,
        "LFO 1 Rate": 1,
        "FilterCtl KeyTrk": 1,
        "Distort DryWet": 1,
        "Sw DistOnOff": 1,
        "Reverb Dry/Wet": 1,
        "Sw ReverbOnOff": 1,
        "ModEnv 1 Decay": 1,
        "xModEnv1 Dest1Am": 1,
    })
    profile = inspect_dataset.build_profile_skeleton("myprofile", keys, top=2)
    assert profile["name"] == "myprofile"
    assert set(profile["params"]) == set(PARAM_NAMES)
    # Distortion/reverb scaffolds should propose a gated strategy when a switch is detected.
    assert profile["params"]["distortion_mix"]["strategy"] == "gated"
    assert profile["params"]["reverb_mix"]["strategy"] == "gated"


def test_inspect_writes_profile_when_requested(tmp_path, monkeypatch):
    path = _fake_sylenth_dataset(tmp_path)
    profiles_dir = tmp_path / "profiles"
    monkeypatch.setattr(inspect_dataset, "PROFILES_DIR", profiles_dir)

    monkeypatch.setattr(sys, "argv", [
        "inspect_dataset.py",
        "--input", str(path),
        "--suggest-profile", "fakesynth",
        "--sample-limit", "5",
        "--top", "2",
    ])

    exit_code = inspect_dataset.main()
    assert exit_code == 0
    written = profiles_dir / "fakesynth.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["name"] == "fakesynth"
    assert set(payload["params"]) == set(PARAM_NAMES)


def test_inspect_refuses_overwrite_without_flag(tmp_path, monkeypatch):
    path = _fake_sylenth_dataset(tmp_path)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "existing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(inspect_dataset, "PROFILES_DIR", profiles_dir)

    monkeypatch.setattr(sys, "argv", [
        "inspect_dataset.py",
        "--input", str(path),
        "--suggest-profile", "existing",
        "--sample-limit", "5",
    ])
    assert inspect_dataset.main() == 1
