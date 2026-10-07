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
