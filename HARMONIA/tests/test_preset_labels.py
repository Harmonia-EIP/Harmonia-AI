"""Readable preset text from human-written names and categories."""

from src.presets.labels import expand, preset_text


def test_synth_programmer_shorthand_is_expanded():
    assert expand("TOTO HMND1") == "toto hammond organ"
    assert expand("E.PIANO 1") == "electric piano"
    assert expand("ELEC.PNO A") == "electric piano a"
    assert expand("SynLead.71") == "synth lead"


def test_preset_text_keeps_descriptive_cartridges_only():
    text = preset_text({"name": "AFRICA  1", "aliases": ["AFRICA BR1"], "cartridge": "BRASS_04", "category": ""})
    assert text == "africa. africa br. brass"
    assert preset_text({"name": "Axel F", "aliases": [], "cartridge": "TX7-17C"}) == "axel f"
