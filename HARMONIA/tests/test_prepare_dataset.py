import json

from scripts.prepare_dataset import (
    SYLENTH_KEY_MAP,
    TARGET_PARAMS,
    clean_content,
    convert_fxp_dump_to_json,
)
from src.charter import PARAM_NAMES


def test_clean_content_balances_parentheses_and_trailing_comma():
    raw = "'Preset', {'description': 'Soft_Pad.fxp'},"
    cleaned = clean_content(raw)
    assert cleaned.startswith("(")
    assert cleaned.endswith(")")


def test_target_params_match_charter():
    assert tuple(TARGET_PARAMS) == PARAM_NAMES


def test_convert_fxp_dump_to_json_creates_charter_record(tmp_path):
    input_file = tmp_path / "raw.txt"
    output_file = tmp_path / "presets.json"

    sylenth_params = {sylenth_key: 0.5 for sylenth_key in SYLENTH_KEY_MAP}
    sylenth_params["description"] = "My_Soft_Pad.fxp"
    input_file.write_text(str(("my_preset", sylenth_params)), encoding="utf-8")

    written = convert_fxp_dump_to_json(input_file, output_file, anchor_path=None)

    assert written == 1
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(payload) == 1

    record = payload[0]
    assert record["description"] == "My Soft Pad"
    parameters = record["parameters"]
    assert set(parameters) == {"continuous", "binary", "categorical"}

    flat = {}
    flat.update(parameters["continuous"])
    flat.update(parameters["categorical"])
    assert set(flat) == set(PARAM_NAMES)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in flat.values())


def test_convert_fxp_dump_injects_anchors_by_default(tmp_path):
    input_file = tmp_path / "raw.txt"
    output_file = tmp_path / "presets.json"

    sylenth_params = {sylenth_key: 0.5 for sylenth_key in SYLENTH_KEY_MAP}
    sylenth_params["description"] = "Some_Preset.fxp"
    input_file.write_text(str(("preset", sylenth_params)), encoding="utf-8")

    written = convert_fxp_dump_to_json(input_file, output_file)

    assert written > 1
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    descriptions = {entry["description"] for entry in payload}
    assert "Hard Electro Lead" in descriptions


def test_convert_fxp_dump_to_json_handles_missing_file(tmp_path):
    missing_input = tmp_path / "missing.txt"
    output_file = tmp_path / "presets.json"

    written = convert_fxp_dump_to_json(missing_input, output_file, anchor_path=None)

    assert written == 0
    assert not output_file.exists()
