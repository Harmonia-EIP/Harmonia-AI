"""Harmonia v3 analog engine: v2 compatibility, new modules and parameter mapping."""

import numpy as np

from src.synth import v3_params as P
from src.synth.engine import ENGINE_APP_1_1
from src.synth.engine import render_physical as render_v2
from src.synth.engine_v3 import render

SR = 48000


def _v3(**values):
    base = {"osc_1_waveform": 2, "filter_cutoff": 4000, "amp_decay": 400, "amp_sustain": 0.6}
    base.update(values)
    return P.physical_from_dict(base)


def _spectrum(mono, start=4800, size=32768):
    return np.abs(np.fft.rfft(mono[start:start + size] * np.hanning(size))), np.fft.rfftfreq(size, 1 / SR)


def test_v2_preset_sounds_the_same_in_v3_for_alias_free_waveforms():
    # sustain 1 and no filter envelope: v2's linear and v3's exponential segments do not come into play
    v2 = np.array([0, 1, 0.3, 7, 0, 3000, 0.3, 0, 5, 400, 1.0, 300, 0.0, 300, 5, 0.3, 0.2, 0.5, 0.2, 0.4])
    hold = int(1.4 * SR)
    old = np.mean(render_v2(v2, 60, 100 / 127, 1.5, 4.0, SR, ENGINE_APP_1_1, 1), axis=0)[:hold]
    new = render(P.from_v2_physical(v2)).mean(axis=0)[:hold]
    assert np.sqrt(((old - new) ** 2).mean()) / np.sqrt((old ** 2).mean()) < 0.01


def test_band_limited_saw_has_less_aliasing_than_v2():
    v2 = np.array([2, 2, 0, 0, 0, 20000, 0, 0, 1, 400, 1, 300, 0, 300, 5, 0, 0, 0, 0, 0])
    note = 100  # 2637 Hz: harmonics above Nyquist fold back between the partials
    old = np.mean(render_v2(v2, note, 100 / 127, 1.5, 2.0, SR, ENGINE_APP_1_1, 1), axis=0)
    new = render(P.from_v2_physical(v2), note=note, total_seconds=2.0).mean(axis=0)
    f0 = 440 * 2 ** ((note - 69) / 12)

    def inharmonic(mono):
        spec, freqs = _spectrum(mono, size=65536)
        harmonic = np.abs(freqs / f0 - np.round(freqs / f0)) * f0 < 30
        return spec[~harmonic].sum() / spec.sum()

    assert inharmonic(new) < 0.5 * inharmonic(old)


def test_coarse_tuning_moves_the_pitch_by_semitones():
    audio = render(_v3(osc_1_waveform=0, osc_1_coarse=12), note=57).mean(axis=0)  # A3 + 12 = A4
    spec, freqs = _spectrum(audio)
    assert abs(freqs[spec.argmax()] - 440.0) < 2.0


def test_unison_spreads_voices_in_stereo_and_single_voice_is_mono():
    mono = render(_v3())
    wide = render(_v3(unison_voices=5, unison_detune=20))
    assert np.abs(mono[0] - mono[1]).max() < 1e-9
    assert np.abs(wide[0] - wide[1]).mean() > 0.01


def test_24db_slope_removes_more_highs_than_12db():
    def highs(slope):
        spec, freqs = _spectrum(render(_v3(filter_cutoff=500, filter_slope=slope)).mean(axis=0))
        return spec[freqs > 4000].sum()

    assert highs(1) < 0.3 * highs(0)


def test_pitch_envelope_starts_high_and_settles():
    audio = render(_v3(osc_1_waveform=0, pitch_env_amount=12, pitch_env_decay=200), note=57).mean(axis=0)

    def peak(start):
        spec, freqs = _spectrum(audio, start=start, size=2048)
        return freqs[spec.argmax()]

    assert peak(0) > 300 and abs(peak(24000) - 220) < 30


def test_normalized_mapping_round_trips_and_midpoints():
    physical = P.physical_from_dict({"filter_cutoff": 1000, "unison_voices": 4, "lfo_waveform": 3})
    back = P.denormalize(P.normalize(physical))
    np.testing.assert_allclose(back, physical, rtol=1e-6, atol=1e-6)
    assert abs(P.to_normalized(P.BY_NAME["filter_cutoff"], 1000) - 0.5) < 1e-6
    assert len(P.NAMES) == 45 and len(set(P.NAMES)) == 45
