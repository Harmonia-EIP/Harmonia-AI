import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import BIPOLAR_INDICES, PARAM_NAMES
from src.dataset_profiles import (
    apply_profile,
    autodetect_profile,
    list_available_profiles,
    load_profile,
    profile_matches,
)


def _minimal_profile_dict():
    return {
        "name": "minimal",
        "detect": {"required_keys": ["Cutoff"]},
        "params": {
            name: {"strategy": "constant", "value": 0.5 if i in BIPOLAR_INDICES else 0.0}
            for i, name in enumerate(PARAM_NAMES)
        },
    }


def test_sylenth1_profile_ships_with_repo():
    available = list_available_profiles()
    assert "sylenth1" in available

    profile = load_profile("sylenth1")
    assert profile["name"] == "sylenth1"
    assert set(profile["params"]) == set(PARAM_NAMES)


def test_load_profile_rejects_missing_params(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"name": "x", "params": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(str(broken))


def test_apply_profile_direct_strategy():
    profile = _minimal_profile_dict()
    profile["params"]["filter_cutoff"] = {"strategy": "direct", "key": "Cutoff", "default": 0.0}
    vector = apply_profile(profile, {"Cutoff": 0.7})
    assert len(vector) == 20
    assert vector[PARAM_NAMES.index("filter_cutoff")] == pytest.approx(0.7)


def test_apply_profile_clamps_and_snaps_discrete():
    profile = _minimal_profile_dict()
    profile["params"]["filter_cutoff"] = {"strategy": "direct", "key": "Cutoff"}
    profile["params"]["osc_1_waveform"] = {"strategy": "direct", "key": "Wave"}

    vector = apply_profile(profile, {"Cutoff": 1.4, "Wave": 0.51})
    cutoff = vector[PARAM_NAMES.index("filter_cutoff")]
    waveform = vector[PARAM_NAMES.index("osc_1_waveform")]
    assert cutoff == pytest.approx(1.0)            # clamped
    assert waveform in (0.0, 1.0/3.0, 2.0/3.0, 1.0) # snapped to one of 4 steps


def test_apply_profile_gated_strategy_blocks_disabled():
    profile = _minimal_profile_dict()
    profile["params"]["reverb_mix"] = {
        "strategy": "gated", "value": "Wet", "gate": "Sw", "default": 0.0,
    }
    enabled = apply_profile(profile, {"Wet": 0.8, "Sw": 1.0})
    disabled = apply_profile(profile, {"Wet": 0.8, "Sw": 0.0})
    idx = PARAM_NAMES.index("reverb_mix")
    assert enabled[idx] == pytest.approx(0.8)
    assert disabled[idx] == pytest.approx(0.0)


def test_apply_profile_max_strategy():
    profile = _minimal_profile_dict()
    profile["params"]["noise_level"] = {"strategy": "max", "keys": ["A", "B"], "default": 0.0}
    vector = apply_profile(profile, {"A": 0.2, "B": 0.7})
    assert vector[PARAM_NAMES.index("noise_level")] == pytest.approx(0.7)


def test_apply_profile_bipolar_amount_preserves_neutral():
    profile = _minimal_profile_dict()
    profile["params"]["filter_env_amount"] = {
        "strategy": "bipolar_amount", "key": "Amt", "default": 0.5,
    }
    vector = apply_profile(profile, {})
    assert vector[PARAM_NAMES.index("filter_env_amount")] == pytest.approx(0.5)


def test_apply_profile_routed_amount():
    profile = _minimal_profile_dict()
    profile["params"]["lfo_to_pitch"] = {
        "strategy": "routed_amount",
        "amount_key": "Amt", "dest_key": "Dest",
        "expected_dest": 0.0, "tolerance": 0.05, "default": 0.0,
    }
    idx = PARAM_NAMES.index("lfo_to_pitch")
    matching = apply_profile(profile, {"Amt": 0.9, "Dest": 0.0})
    non_matching = apply_profile(profile, {"Amt": 0.9, "Dest": 0.5})
    assert matching[idx] == pytest.approx(abs(0.9 * 2 - 1))
    assert non_matching[idx] == pytest.approx(0.0)


def test_profile_matches_handles_detection_keys():
    profile = _minimal_profile_dict()
    ok, _ = profile_matches(profile, ["Cutoff", "Whatever"])
    assert ok is True
    missing, reason = profile_matches(profile, ["Whatever"])
    assert missing is False
    assert "required" in reason


def test_autodetect_returns_sylenth1_when_keys_match():
    sample_keys = ["AmpEnv A Attack", "AmpEnv A Decay", "Filter A Cutoff", "Osc A1 Waveform"]
    detected = autodetect_profile(sample_keys)
    assert detected is not None
    assert detected["name"] == "sylenth1"


def test_autodetect_returns_none_for_alien_dataset():
    sample_keys = ["unrelated_param_a", "unrelated_param_b"]
    assert autodetect_profile(sample_keys) is None
