"""Charter v3 (charter/harmonia_v3.json): one source for the plugin, the backend and the AI."""

import math
import shutil
import subprocess

import pytest

from scripts.v3 import charter_cpp
from src import charter_v3 as C
from src.presets import dx7, dx7_render
from src.synth import params as v2_params
from src.synth import v3_params as P
from src.v3 import codecs
from src.v3.vocabulary import TYPES
from tests.test_preset_readers import _bulk, _packed_voice


def _voice(tmp_path):
    path = tmp_path / "cart.syx"
    path.write_bytes(_bulk([_packed_voice()] * 32))
    return next(dx7.read_file(path))


def test_parameters_are_unique_and_in_the_ai_vector_order():
    charter = C.load()
    ids = [p.id for p in charter.params]
    assert len(ids) == len(set(ids)) == 1 + 46 + 6 * 21 + 19
    assert C.ids("analog") == P.NAMES
    assert P.NAMES[:20] == tuple(v2_params.PARAM_NAMES)  # the v2 order the upgrade rules index


def test_dx7_parameters_follow_the_codec_layout():
    expected = []
    for fields in [codecs.OP_FIELDS] * 6 + [codecs.GLOBAL_FIELDS]:
        for name, kind, size in fields:
            highest = size - 1 if kind == "cat" else size  # the codec counts classes, the charter stores the top value
            expected += [(name, highest)] * (4 if kind == "list" else 1)
    got = [(p.dx7_path[1], int(p.max)) for p in C.load().engine_params("dx7")]
    assert got == expected
    for p in C.load().engine_params("dx7"):
        if p.choices:
            assert len(p.choices) == p.max + 1


def test_dx7_voice_round_trips_through_charter_values(tmp_path):
    voice = _voice(tmp_path)
    values = C.dx7_values(voice)
    assert values["dx7_op1_output_level"] == 85 and values["dx7_algorithm"] == 4
    assert dx7_render.to_patch(C.dx7_voice(values, voice["name"])) == dx7_render.to_patch(voice)
    init = C.dx7_voice({})  # every default: the DX7 INIT VOICE
    assert [op["output_level"] for op in init["operators"]] == [99, 0, 0, 0, 0, 0]
    assert init["transpose"] == 24 and init["operators"][0]["levels"] == [99, 99, 99, 0]
    dx7_render.to_patch(init)


def test_normalization_matches_juce_normalisable_range():
    for p in C.load().params:
        if p.centre is not None:
            assert p.to_normalized(p.centre) == pytest.approx(0.5)
        for n in (0.0, 0.125, 0.5, 0.875, 1.0):
            v = p.from_normalized(n)
            assert p.min <= v <= p.max
            assert p.from_normalized(p.to_normalized(v)) == pytest.approx(v, rel=1e-9, abs=1e-9)
        if p.kind in ("choice", "bool"):
            k = len(p.choices)
            assert [p.to_normalized(i) for i in range(k)] == pytest.approx([i / (k - 1) for i in range(k)])
    by_id = C.load().by_id
    # halves round like the plugin's JUCE parameters (checked against JUCE 8.0.8): int / choice to even, bool up
    assert by_id["unison_voices"].snap(8.5) == 8 and by_id["unison_voices"].snap(9.5) == 10
    assert by_id["filter_slope"].from_normalized(0.5) == 0 and by_id["dx7_lfo_wave"].snap(2.5) == 2
    assert by_id["osc_sync"].from_normalized(0.5) == 1
    cutoff = by_id["filter_cutoff"]
    # juce::NormalisableRange::setSkewForCentre (1000) on 20..20000
    assert cutoff.skew == pytest.approx(math.log(0.5) / math.log(980 / 19980))


def test_presets_round_trip_and_invalid_ones_are_reported(tmp_path):
    doc = C.make_preset("analog", {"filter_cutoff": 1200.0, "unison_voices": 7, "osc_1_waveform": 2}, "Test",
                        brief={"family": "bass", "attack": "instant", "references": ["Minimoog"]}, generated_by="test")
    assert C.validate_preset(doc) == [] and doc["metadata"]["charter_version"] == "3.0.0"
    engine, values = C.read_preset(doc)
    assert engine == "analog" and values["unison_voices"] == 7 and values["filter_cutoff"] == pytest.approx(1200.0)
    assert values["reverb_size"] == C.load().by_id["reverb_size"].default

    dx = C.make_preset("dx7", C.dx7_values(_voice(tmp_path)), "E.PIANO 1")
    engine, values = C.read_preset(dx)
    assert engine == "dx7" and values["dx7_op6_output_level"] == 90

    assert C.validate_preset({"parameters": {"engine": 0.0, "nope": 0.5}}) == ["unknown parameter 'nope'"]
    assert "'parameters' must hold 'engine'" in C.validate_preset({"parameters": {"osc_mix": 0.5}})
    assert C.validate_preset({"parameters": {"engine": 0.0, "osc_mix": 1.5}})
    assert C.validate_preset({"parameters": {"engine": 0.0}, "brief": {"attack": "sudden"}})
    with pytest.raises(ValueError):
        C.read_preset({"parameters": {"engine": 2.0}})


def test_v2_upgrade_rules():
    v2 = dict(zip(v2_params.PARAM_NAMES, [0.0] * 20))
    v2.update(filter_env_amount=0.5, lfo_to_pitch=0.2, lfo_to_cutoff=0.5, reverb_mix=1.0, filter_cutoff=800.0)
    v3 = C.upgrade_v2(v2)
    assert v3["filter_env_amount"] == pytest.approx(2.0)  # v2 scaled its -1..1 by 4 octaves
    assert v3["lfo_to_pitch"] == pytest.approx(10.0) and v3["lfo_to_cutoff"] == pytest.approx(1.0)
    assert v3["reverb_size"] == pytest.approx(0.9) and v3["filter_cutoff"] == pytest.approx(800.0)
    assert v3["unison_voices"] == 1  # new parameters keep their neutral default


def test_sound_brief_vocabulary():
    fields = {f["id"]: f for f in C.load().raw["sound_brief"]["fields"]}
    assert fields["family"]["values"] == TYPES
    assert C.validate_brief({"family": "pad", "space": "hall", "references": ["Juno"]}) == []
    assert C.validate_brief({"colour": "red"}) == ["brief: unknown field 'colour'"]


def test_generated_headers_are_up_to_date():
    for name, text in charter_cpp.render(C.load()).items():
        assert (charter_cpp.OUT / name).read_text(encoding="utf-8") == text, f"run scripts/v3/charter_cpp.py ({name})"


@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
def test_cpp_table_normalizes_like_python(tmp_path):
    program = tmp_path / "check.cpp"
    program.write_text("""
#include "HarmoniaCharterV3.h"
#include <cstdio>
int main()
{
    std::array<float, 20> v2 {};
    for (std::size_t i = 0; i < v2.size(); ++i)
        v2[i] = 0.05f * (float) (i + 1);
    for (const auto& p : HarmoniaCharter::params)
    {
        for (float t : { 0.0f, 0.125f, 0.25f, 0.5f, 0.875f, 1.0f })
        {
            const float v = HarmoniaCharter::fromNormalized (p, t);
            std::printf ("%.*s %g %.9g %.9g\\n", (int) p.id.size(), p.id.data(), t, v, HarmoniaCharter::toNormalized (p, v));
        }
        if (p.engine == HarmoniaCharter::Engine::Analog)
            std::printf ("%.*s v2 %.9g 0\\n", (int) p.id.size(), p.id.data(), HarmoniaCharter::upgradeFromV2 (p, v2));
    }
}
""")
    exe = tmp_path / "check"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(charter_cpp.OUT), str(program),
                    "-o", str(exe)], check=True)
    lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split("\n")
    by_id = C.load().by_id
    v2 = {name: 0.05 * (i + 1) for i, name in enumerate(v2_params.PARAM_NAMES)}
    upgraded = C.upgrade_v2(v2)
    checked = 0
    for line in filter(None, lines):
        pid, t, value, normalized = line.split()
        p = by_id[pid]
        if t == "v2":
            assert float(value) == pytest.approx(upgraded[pid], rel=1e-5, abs=1e-5), pid
        else:
            physical = p.from_normalized(float(t))
            assert float(value) == pytest.approx(physical, rel=1e-5, abs=1e-5), (pid, t)
            # compared at Python's own value: a skew < 1 magnifies float rounding just above the minimum
            assert float(normalized) == pytest.approx(p.to_normalized(physical), abs=1e-5), (pid, t)
        checked += 1
    assert checked == len(by_id) * 6 + 46
