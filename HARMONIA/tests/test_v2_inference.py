import json

import numpy as np
import pytest

from scripts.v2.fetch_model import sha256, verify_model_dir
from src.charter import PARAM_NAMES, normalise_vector
from src.v2.inference import HarmoniaV2
from tests.v2_fixtures import build_model_dir


@pytest.fixture()
def model_dir(tmp_path):
    params = build_model_dir(tmp_path / "harmonia_v2")
    return tmp_path / "harmonia_v2", params


# Retrieval returns the bank preset whose audio embedding is closest to the prompt.
def test_retrieval_picks_nearest_bank_preset(model_dir):
    path, params = model_dir
    engine = HarmoniaV2(path)
    door = engine.generate("door slam", mode="retrieval")
    piano = engine.generate("soft piano", mode="retrieval")
    assert door.bank_index == 0 and piano.bank_index == 1
    assert door.values == pytest.approx(normalise_vector(params[0].astype(float).tolist()), abs=1e-3)
    assert door.similarity == pytest.approx(1.0, abs=0.02)


# Every mode returns 20 normalized values with discrete parameters snapped to their steps.
@pytest.mark.parametrize("mode", ["hybrid", "retrieval", "neural"])
def test_every_mode_returns_a_charter_vector(model_dir, mode):
    engine = HarmoniaV2(model_dir[0])
    result = engine.generate("wind", mode=mode)
    assert len(result.values) == len(PARAM_NAMES)
    assert all(0.0 <= v <= 1.0 for v in result.values)
    assert result.values == normalise_vector(result.values)


# Variations are reproducible for a given seed and stay among the retrieved candidates.
def test_variation_is_deterministic(model_dir):
    engine = HarmoniaV2(model_dir[0])
    first = engine.generate("door slam", variation=7)
    again = engine.generate("door slam", variation=7)
    assert first.bank_index == again.bank_index
    assert first.bank_index < len(engine.bank_params)


# Negative variation seeds are accepted (the app may send any integer).
def test_negative_variation(model_dir):
    assert HarmoniaV2(model_dir[0]).generate("door slam", variation=-5).bank_index is not None


# An unknown mode is rejected.
def test_unknown_mode_is_rejected(model_dir):
    with pytest.raises(ValueError):
        HarmoniaV2(model_dir[0]).generate("door slam", mode="magic")


# fetch_model refuses a model directory whose files do not match their manifest checksums.
def test_verify_model_dir_detects_tampering(model_dir):
    path, _ = model_dir
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {"bank.npz": {"sha256": sha256(path / "bank.npz")}}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verify_model_dir(path)
    np.savez(path / "bank.npz", emb=np.zeros((1, 512), dtype=np.int8))
    with pytest.raises(RuntimeError):
        verify_model_dir(path)
