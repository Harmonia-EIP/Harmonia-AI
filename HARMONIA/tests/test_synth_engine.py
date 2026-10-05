import numpy as np
import pytest

from src.charter import PARAM_INDEX, PARAM_NAMES
from src.synth.engine import ENGINE_APP_1_0, ENGINE_APP_1_1, render_preset
from src.synth.params import APP_PARAM_SPECS, convert_from_0to1, to_physical
from src.synth.sampling import SOURCES, sample_bank

SR = 48000


def _preset(**overrides):
    values = dict.fromkeys(PARAM_NAMES, 0.0)
    values.update(osc_1_waveform=2 / 3, osc_mix=0.0, filter_cutoff=0.8, filter_resonance=0.1, amp_attack=0.0,
                  amp_decay=0.6, amp_sustain=0.7, amp_release=0.5, filter_env_amount=0.5, reverb_mix=0.0)
    values.update(overrides)
    return [values[name] for name in PARAM_NAMES]


def _centroid(signal):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / SR)
    return float((spectrum * freqs).sum() / spectrum.sum())


# Normalized values map to the app's physical values exactly like JUCE (skewed log ranges, choice rounding).
@pytest.mark.parametrize(
    "name, normalized, expected",
    [
        ("filter_cutoff", 0.5, 1000.0),
        ("amp_attack", 0.5, 100.0),
        ("amp_release", 0.5, 500.0),
        ("lfo_rate", 0.5, 2.0),
        ("osc_2_detune", 1.0, 100.0),
        ("distortion_mix", 0.25, 0.25 ** (1 / 1.8)),
        ("filter_type", 0.2, 0.0),
        ("filter_type", 0.3, 1.0),
        ("filter_type", 0.8, 2.0),
        ("osc_1_waveform", 2 / 3, 2.0),
    ],
)
def test_convert_from_0to1_matches_juce(name, normalized, expected):
    assert convert_from_0to1(APP_PARAM_SPECS[name], normalized) == pytest.approx(expected, rel=1e-3, abs=1e-3)


# to_physical() keeps the charter order and rejects vectors of the wrong size.
def test_to_physical_order_and_size():
    physical = to_physical([0.5] * len(PARAM_NAMES))
    assert physical.shape == (len(PARAM_NAMES),)
    assert physical[PARAM_INDEX["filter_cutoff"]] == pytest.approx(1000.0, rel=1e-3)
    with pytest.raises(ValueError):
        to_physical([0.5] * 3)


# The same preset and seed always render the same samples.
def test_render_is_deterministic():
    preset = _preset(noise_level=0.3)
    a = render_preset(preset, total_seconds=0.5, hold_seconds=0.2)
    b = render_preset(preset, total_seconds=0.5, hold_seconds=0.2)
    assert a.shape == (SR // 2,)
    assert np.array_equal(a, b)


# App 1.0 divides by zero when the resonance is 0 (NaN output); the fixed engine stays finite.
def test_zero_resonance_nan_only_in_app_1_0():
    preset = _preset(filter_resonance=0.0)
    assert np.isnan(render_preset(preset, total_seconds=0.3, hold_seconds=0.1, q_mode=ENGINE_APP_1_0)).any()
    assert np.isfinite(render_preset(preset, total_seconds=0.3, hold_seconds=0.1, q_mode=ENGINE_APP_1_1)).all()


# A sustain-0 note stops dead at note-off in app 1.0, but releases from its current level once fixed.
def test_release_tail_after_note_off():
    preset = _preset(amp_decay=1.0, amp_sustain=0.0, amp_release=0.8)
    hold = 0.3
    window = slice(int((hold + 0.01) * SR), int((hold + 0.2) * SR))
    old = render_preset(preset, hold_seconds=hold, total_seconds=1.0, q_mode=ENGINE_APP_1_0)
    fixed = render_preset(preset, hold_seconds=hold, total_seconds=1.0, q_mode=ENGINE_APP_1_1)
    assert np.abs(old[window]).max() == 0.0
    assert np.sqrt((fixed[window] ** 2).mean()) > 0.01


# With the Q mapping fix, a low resonance no longer darkens the sound far below its cutoff.
def test_fixed_filter_is_brighter_for_low_resonance():
    preset = _preset(filter_cutoff=0.6, filter_resonance=0.05, filter_env_amount=0.5)
    old = render_preset(preset, note=57, hold_seconds=0.5, total_seconds=0.5, q_mode=ENGINE_APP_1_0)
    fixed = render_preset(preset, note=57, hold_seconds=0.5, total_seconds=0.5, q_mode=ENGINE_APP_1_1)
    assert _centroid(fixed) > 1.5 * _centroid(old)


# The bank sampler returns valid charter vectors from the three sources.
def test_sample_bank_shapes_and_ranges():
    params, sources, labels = sample_bank(300, seed=1)
    assert params.shape == (300, len(PARAM_NAMES))
    assert params.min() >= 0.0 and params.max() <= 1.0
    assert set(np.unique(sources)) == set(range(len(SOURCES)))
    assert any(labels)
