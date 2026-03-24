import json

from scripts.prepare_dataset import TARGET_PARAMS, clean_content, convert_fxp_dump_to_json


def test_clean_content_balances_parentheses_and_trailing_comma():
	raw = "'Preset', {'description': 'Soft_Pad.fxp'},"
	cleaned = clean_content(raw)
	assert cleaned.startswith("(")
	assert cleaned.endswith(")")


def test_convert_fxp_dump_to_json_creates_expected_dataset(tmp_path):
	input_file = tmp_path / "raw.txt"
	output_file = tmp_path / "presets.json"

	params = {key: 0.5 for key in TARGET_PARAMS}
	params["description"] = "My_Soft_Pad.fxp"
	input_file.write_text(str(("my_preset", params)), encoding="utf-8")

	convert_fxp_dump_to_json(input_file, output_file)

	assert output_file.exists()
	payload = json.loads(output_file.read_text(encoding="utf-8"))
	assert len(payload) == 1
	assert payload[0]["description"] == "My Soft Pad"
	assert len(payload[0]["parameters"]) == len(TARGET_PARAMS)
	assert all(isinstance(v, float) for v in payload[0]["parameters"])


def test_convert_fxp_dump_to_json_handles_missing_file(tmp_path):
	missing_input = tmp_path / "missing.txt"
	output_file = tmp_path / "presets.json"

	convert_fxp_dump_to_json(missing_input, output_file)

	assert not output_file.exists()

