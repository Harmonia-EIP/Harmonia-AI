"""Preset converters (OB-Xf, Surge XT -> Harmonia v3) and the sound matching tools."""

import math

import numpy as np

from src.presets import matching as M
from src.presets.convert_obxf import obxf_to_v3
from src.presets.convert_surge import surge_to_v3
from src.synth import v3_params as P


def _value(physical, name):
    return physical[P.INDEX[name]]


def test_obxf_scalings_follow_the_plugin():
    p = obxf_to_v3({"Osc1SawWave": 1.0, "Osc2PulseWave": 1.0, "Osc2Pitch": 0.75, "Osc1Mix": 1.0, "Osc2Mix": 1.0,
                    "FilterCutoff": 0.5, "FilterKeyFollow": 0.0, "Filter4PoleMode": 1.0, "FilterEnvAmount": 0.6,
                    "AmpEnvSustain": 0.5, "Unison": 1.0, "UnisonVoices": 2 / 32})
    assert _value(p, "osc_1_waveform") == 2 and _value(p, "osc_2_waveform") == 3
    assert _value(p, "osc_2_coarse") == 12 and _value(p, "osc_mix") == 0.5
    assert math.isclose(_value(p, "filter_cutoff"), 440 * 2 ** (15 / 12), rel_tol=1e-6)  # pitch index 60 - 45
    assert _value(p, "filter_slope") == 1 and math.isclose(_value(p, "filter_env_amount"), 7.0)
    assert _value(p, "unison_voices") == 3 and _value(p, "amp_sustain") == 0.5


def test_surge_units_are_converted():
    params = {"a_osc1_type": 0, "a_osc1_param0": 1.0, "a_osc1_octave": -1, "a_level_o1": 1.0, "a_mute_o1": 0,
              "a_filter1_type": 2, "a_filter1_cutoff": 12.0, "a_filter1_envmod": 24.0, "a_env1_decay": 0.0,
              "a_env1_sustain": 0.25, "a_lfo0_rate": 2.0, "a_lfo0_shape": 0}
    modulation = [{"target": "a_pitch", "source": 17, "index": 0, "depth": 0.3}]
    p = surge_to_v3(params, modulation)
    assert _value(p, "osc_1_waveform") == 3 and _value(p, "osc_1_coarse") == -12
    assert math.isclose(_value(p, "filter_cutoff"), 880.0) and _value(p, "filter_slope") == 1
    assert math.isclose(_value(p, "filter_env_amount"), 2.0) and math.isclose(_value(p, "amp_decay"), 1000.0)
    assert math.isclose(_value(p, "lfo_to_pitch"), 30.0) and math.isclose(_value(p, "lfo_rate"), 4.0)


def test_distance_ignores_loudness_and_sees_timbre():
    t = np.arange(int(0.5 * M.SR)) / M.SR
    sine = [np.sin(2 * np.pi * 220 * t)]
    square = [np.sign(np.sin(2 * np.pi * 220 * t))]
    assert M.distance(M.features(sine), M.features([0.1 * sine[0]])) < 1e-6
    assert M.distance(M.features(sine), M.features(square)) > 5.0


def test_cma_es_finds_a_quadratic_minimum():
    target = np.array([0.2, 0.7, 0.4])
    x, best, evals = M.cma_es(lambda x: float(np.sum((x - target) ** 2)), np.full(3, 0.5), 0.2, 600,
                              np.random.default_rng(0))
    assert best < 1e-4 and evals <= 600
    np.testing.assert_allclose(x, target, atol=0.02)


def test_optimizer_cannot_add_what_the_original_lacks():
    start = P.physical_from_dict({"osc_1_waveform": 2, "lfo_to_pitch": 30.0, "noise_level": 0.0})
    free, lower, upper = M.search_space(start)
    names = [P.NAMES[i] for i in free]
    assert "noise_level" not in names and "distortion_mix" not in names and "chorus_mix" not in names
    assert "velocity_to_filter" not in names and "delay_time" not in names  # no delay: its time is moot
    k = names.index("lfo_to_pitch")
    x0 = P.normalize(start)[P.INDEX["lfo_to_pitch"]]
    assert math.isclose(upper[k] - x0, M.NUDGE) and "lfo_rate" in names


def test_width_counts_only_for_stereo_audio():
    t = np.arange(int(0.5 * M.SR)) / M.SR
    tone = np.sign(np.sin(2 * np.pi * 220 * t))
    mono = M.features([np.stack([tone, tone])])
    wide = M.features([np.stack([tone, np.roll(tone, 37)])])
    parts = M.perceptual.components(mono, wide)
    assert parts["width"] > 10.0 and M.perceptual.components(mono, mono)["width"] == 0.0


def test_global_transpose_and_scene_octave_are_converted():
    ob = obxf_to_v3({"Transpose": 0.25, "Osc1Pitch": 0.5, "Osc2Pitch": 0.75, "Osc1Mix": 1.0})
    assert _value(ob, "osc_1_coarse") == -12 and _value(ob, "osc_2_coarse") == 0
    sg = surge_to_v3({"a_octave": -1, "a_osc1_type": 0, "a_level_o1": 1.0, "a_mute_o1": 0}, [])
    assert _value(sg, "osc_1_coarse") == -12


def test_arpeggios_sequences_and_templates_are_not_playable():
    from src.presets.filters import playable

    assert not playable({"name": "Evolution Arp", "category": "Keys"})
    assert not playable({"name": "Init Saw", "category": "Templates"})
    seq = {"source": "surge", "name": "Pulse", "category": "Bass", "params": {"a_lfo0_shape": 7},
           "modulation": [{"target": "a_osc1_pitch", "source": 17, "index": 0, "depth": 12.0}]}
    assert not playable(seq) and playable({**seq, "params": {"a_lfo0_shape": 0}})
