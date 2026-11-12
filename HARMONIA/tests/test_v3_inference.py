"""v3 run time (src/v3/inference.py): prompt -> charter-v3 preset, and the refine of an existing preset."""

import numpy as np
import pytest
import torch

from src import charter_v3
from src.charter import PARAM_NAMES
from src.synth import v3_params as P
from src.v3 import codecs
from src.v3 import directives as D
from src.v3.diffusion import Diffusion
from src.v3.inference import HarmoniaV3, lock_mask, structure_mask
from tests.v3_fixtures import build_engine, dx7_available


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    return build_engine(tmp_path_factory.mktemp("harmonia_v3"))


def full_preset(parameters: dict) -> dict:
    """What the plugin sends as the current preset: every parameter of the charter, normalized."""
    full = {p.id: p.to_normalized(p.default) for p in charter_v3.load().params}
    full.update(parameters)
    return full


# A generation is a valid charter-v3 preset of one of the engines, with that engine's parameters.
def test_generate_returns_a_valid_v3_preset(engine):
    result = engine.generate("door slam")
    assert charter_v3.validate_preset(result.preset) == []
    assert result.engine in engine.engines
    assert result.preset["metadata"]["charter_version"] == charter_v3.load().version
    assert result.preset["metadata"]["engine"] == result.engine
    assert set(result.preset["parameters"]) == {"engine", *charter_v3.ids(result.engine)}
    read_engine, _ = charter_v3.read_preset(result.preset)
    assert read_engine == result.engine


# The same prompt gives the same preset; a variation gives another one.
def test_generate_is_repeatable_and_variations_differ(engine):
    first = engine.generate("soft pad")
    assert engine.generate("soft pad").preset == first.preset
    assert engine.generate("soft pad", variation=1).preset != first.preset


# The search after the draw only keeps a sound that is closer to the prompt.
def test_polish_never_moves_away_from_the_prompt(engine):
    info = engine.generate("acid pluck lead").info
    assert info["similarity"] >= info["drawn_similarity"]


# Refine keeps the engine, returns the locked parameters exactly as sent and does not move away from the prompt.
def test_refine_keeps_locked_parameters_and_the_engine(engine):
    current = full_preset(engine.generate("grosse basse ronflante").preset["parameters"])
    current["engine"] = 0.0
    locked = ["filter_cutoff", "amp_attack", "reverb_mix"]
    current.update(filter_cutoff=0.123456789, amp_attack=0.3141592, reverb_mix=0.0)
    result = engine.refine("grosse basse ronflante", current, locked=locked)
    assert result.engine == "analog"
    assert charter_v3.validate_preset(result.preset) == []
    for key in locked:
        assert result.preset["parameters"][key] == current[key]
    assert result.info["similarity"] >= result.info["similarity_before"]


# At the default strength a refine keeps the tunings and the voice count: it must not transpose the sound.
def test_refine_keeps_the_pitch_structure(engine):
    current = full_preset({"engine": 0.0, "osc_1_coarse": P.to_normalized(P.BY_NAME["osc_1_coarse"], -12),
                           "osc_2_coarse": P.to_normalized(P.BY_NAME["osc_2_coarse"], 7),
                           "unison_voices": P.to_normalized(P.BY_NAME["unison_voices"], 4), "osc_mix": 0.5})
    refined = engine.refine("brighter", current).preset
    _, values = charter_v3.read_preset(refined)
    assert (values["osc_1_coarse"], values["osc_2_coarse"], values["unison_voices"]) == (-12, 7, 4)


# With every parameter locked there is nothing to move: the preset comes back unchanged.
def test_refine_with_everything_locked_changes_nothing(engine):
    current = full_preset({"engine": 0.0, "osc_mix": 0.37, "filter_cutoff": 0.61})
    result = engine.refine("brighter", current, locked=list(P.NAMES))
    assert result.info["changed"] is False
    for name in P.NAMES:
        assert result.preset["parameters"][name] == current[name]


# A v1/v2 preset (20 parameters on the v2 ranges) is refined on the analog engine; unknown ids are ignored.
def test_refine_accepts_a_v2_preset(engine):
    current = {name: 0.4 for name in PARAM_NAMES}
    current["not_a_parameter"] = 0.5
    assert HarmoniaV3.is_v3(current) is False
    result = engine.refine("warm pad", current, locked=["osc_mix"])
    assert result.engine == "analog"
    assert charter_v3.validate_preset(result.preset) == []
    assert result.preset["parameters"]["osc_mix"] == pytest.approx(0.4, abs=1e-6)


# The strength is clamped to 0..1 and reported.
def test_refine_strength_is_clamped(engine):
    current = full_preset({"engine": 0.0})
    assert engine.refine("x", current, strength=7).info["strength"] == 1.0
    assert engine.refine("x", current, strength=-1).info["strength"] == 0.0


@pytest.mark.skipif(not dx7_available(), reason="DX7 renderer not built (scripts/v3/build_dx7.py)")
def test_refine_a_dx7_voice_keeps_its_engine_and_locks(engine):
    voice = charter_v3.dx7_voice({})
    current = full_preset(charter_v3.make_preset("dx7", charter_v3.dx7_values(voice), "init")["parameters"])
    locked = ["dx7_op1_output_level", "dx7_feedback"]
    current["dx7_op1_output_level"] = 0.5
    result = engine.refine("bell", current, locked=locked)
    assert result.engine == "dx7"
    assert charter_v3.validate_preset(result.preset) == []
    for key in locked:
        assert result.preset["parameters"][key] == current[key]
    # the algorithm and the operator ratios are structure: held at the default strength
    before, after = charter_v3.read_preset({"parameters": current})[1], charter_v3.read_preset(result.preset)[1]
    for key in ("dx7_algorithm", "dx7_op1_freq_coarse", "dx7_transpose"):
        assert after[key] == before[key]


# The lock mask of a DX7 parameter covers exactly the positions of the codec vector that hold it.
def test_dx7_lock_mask_matches_the_codec():
    voice = charter_v3.dx7_voice({})
    base = codecs.dx7_encode(voice)
    for param_id in ("dx7_op3_output_level", "dx7_op1_eg_rate_2", "dx7_algorithm", "dx7_lfo_speed"):
        mask = lock_mask("dx7", [param_id])
        assert mask.any()
        changed = base.copy()
        changed[mask] = -changed[mask] if param_id != "dx7_algorithm" else np.roll(changed[mask], 3)
        before = charter_v3.dx7_values(codecs.dx7_decode(base))
        after = charter_v3.dx7_values(codecs.dx7_decode(changed))
        assert [k for k in before if before[k] != after[k]] == [param_id]


def test_masks_cover_the_right_analog_parameters():
    assert [P.NAMES[i] for i in np.nonzero(lock_mask("analog", ["reverb_mix", "dx7_feedback", "nope"]))[0]] == ["reverb_mix"]
    assert {P.NAMES[i] for i in np.nonzero(structure_mask("analog"))[0]} == {"osc_1_coarse", "osc_2_coarse", "unison_voices"}


# Kept values of a variation come out exactly as they went in; the others are drawn again.
def test_diffusion_keeps_masked_values():
    torch.manual_seed(0)
    diffusion = Diffusion.create(dim=6, cond_dim=4, width=16, depth=1)
    start = torch.linspace(-0.8, 0.8, 6).repeat(5, 1)
    keep = torch.tensor([True, False, True, False, False, True])
    out = diffusion.sample(torch.zeros(5, 4), steps=20, start=start, strength=0.5, keep=keep)
    torch.testing.assert_close(out[:, keep], start[:, keep])
    assert not torch.allclose(out[:, ~keep], start[:, ~keep])
    with pytest.raises(ValueError):
        diffusion.sample(torch.zeros(5, 4), keep=keep)


# Comparatives are read in French and English, one direction per measure; descriptions hold none.
@pytest.mark.parametrize("prompt, expected", [
    ("plus sombre", ["brightness-"]),
    ("Plus brillant et plus de réverb", ["brightness+", "space+"]),
    ("make it darker with a slower attack", ["brightness-", "attack+"]),
    ("attaque plus longue", ["attack+"]),
    ("un peu plus long", ["length+"]),
    ("sharper attack, shorter", ["attack-", "length-"]),
    ("moins de reverb", ["space-"]),
    ("punchier", ["attack-"]),
    ("Porte qui claque dans une cathédrale", []),
    ("sombre pad", []),
    ("Grosse basse ronflante", []),
])
def test_comparatives_are_read(prompt, expected):
    assert [d.label for d in D.read(prompt)] == expected


def test_amount_words_scale_the_move():
    assert D.amount("un peu plus sombre") == 0.5
    assert D.amount("much brighter") == 1.6
    assert D.amount("plus sombre") == 1.0


# The measures follow the sound: a duller, slower, longer note reads lower brightness, longer attack and length.
def test_measures_follow_the_sound():
    t = np.arange(4 * D.SR) / D.SR
    bright = np.sin(2 * np.pi * 3000 * t) * (t < 1.5)
    dull = np.sin(2 * np.pi * 200 * t) * np.minimum(t / 0.5, 1.0) * np.where(t < 1.5, 1.0, np.exp(-(t - 1.5) * 3))
    a, b = D.measure(bright), D.measure(dull)
    assert a["brightness"] > b["brightness"] + 1.0
    assert b["attack"] > a["attack"] + 1.0
    assert b["length"] > a["length"]
    assert b["space"] > a["space"]


# A comparative moves the setting that carries it, the right way, and the measured sound follows.
@pytest.mark.parametrize("prompt, parameter, sign, measure", [
    ("plus sombre", "filter_cutoff", -1, "brightness"),
    ("brighter", "filter_cutoff", +1, "brightness"),
    ("plus de reverb", "reverb_mix", +1, "space"),
    ("plus long", "amp_release", +1, "length"),
    ("slower attack", "amp_attack", +1, "attack"),
])
def test_refine_follows_a_comparative(engine, prompt, parameter, sign, measure):
    current = full_preset({"engine": 0.0, "osc_1_waveform": 2 / 3, "filter_cutoff": 0.5, "reverb_mix": 0.2,
                           "amp_release": 0.3, "amp_attack": 0.2, "amp_sustain": 0.8})
    result = engine.refine(prompt, current)
    assert result.info["directives"] == [d.label for d in D.read(prompt)]
    assert result.info["changed"] is True
    assert sign * (result.preset["parameters"][parameter] - current[parameter]) > 0
    assert sign * (result.info["measures_after"][measure] - result.info["measures_before"][measure]) > 0
    # nothing else of the sound was asked to change
    untouched = [n for n in ("osc_mix", "osc_1_waveform", "lfo_rate", "distortion_mix") if n != parameter]
    for name in untouched:
        assert result.preset["parameters"][name] == pytest.approx(current[name], abs=1e-6)


# A locked control is not moved by a comparative either.
def test_comparative_respects_locks(engine):
    current = full_preset({"engine": 0.0, "osc_1_waveform": 2 / 3, "filter_cutoff": 0.5})
    result = engine.refine("plus sombre", current, locked=["filter_cutoff", "filter_env_amount"])
    assert result.preset["parameters"]["filter_cutoff"] == current["filter_cutoff"]
    assert result.preset["parameters"]["filter_env_amount"] == current["filter_env_amount"]


# "un peu" moves less far than the plain comparative.
def test_un_peu_moves_less(engine):
    current = full_preset({"engine": 0.0, "osc_1_waveform": 2 / 3, "filter_cutoff": 0.6})
    plain = engine.refine("plus sombre", current).preset["parameters"]["filter_cutoff"]
    slight = engine.refine("un peu plus sombre", current).preset["parameters"]["filter_cutoff"]
    assert plain < slight < current["filter_cutoff"]


# The DX7 mode plays dry: a reverb comparative is reported as unsupported there, the others are followed.
@pytest.mark.skipif(not dx7_available(), reason="DX7 renderer not built (scripts/v3/build_dx7.py)")
def test_comparatives_on_a_dx7_voice(engine):
    voice = charter_v3.dx7_voice({})
    current = full_preset(charter_v3.make_preset("dx7", charter_v3.dx7_values(voice), "init")["parameters"])
    assert engine.refine("more reverb", current).info.get("unsupported") == ["space+"]
    result = engine.refine("plus long", current)
    assert result.engine == "dx7" and result.info["directives"] == ["length+"]
    assert result.info["measures_after"]["length"] >= result.info["measures_before"]["length"]
