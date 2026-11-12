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


def test_v2_preset_keeps_its_timbre_in_v3():
    # sustain 1, no filter envelope, no reverb: what differs is only v3's analog drift, oversampling
    # (sub-millisecond latency) and band-limited waveforms, so compare band levels, not samples
    from src.presets import perceptual

    v2 = np.array([0, 1, 0.3, 7, 0, 3000, 0.3, 0, 5, 400, 1.0, 300, 0.0, 300, 5, 0.3, 0.2, 0.5, 0.2, 0.0])
    hold = int(1.4 * SR)
    old = np.mean(render_v2(v2, 60, 100 / 127, 1.5, 4.0, SR, ENGINE_APP_1_1, 1), axis=0)[:hold]
    new = render(P.from_v2_physical(v2)).mean(axis=0)[:hold]
    parts = perceptual.components(perceptual.describe([old]), perceptual.describe([new]))
    # ~1.3 dB: the empty bands between partials lose v2's aliasing junk, the 12-16 kHz bands gain real harmonics
    assert parts["timbre"] < 1.5 and parts["envelope"] < 1.0


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


def test_stereo_width_spreads_unison_voices_and_zero_width_stays_mono():
    mono = render(_v3(unison_voices=5, unison_detune=20))
    wide = render(_v3(unison_voices=16, unison_detune=20, stereo_width=0.8))
    two_osc = render(_v3(osc_2_waveform=2, osc_mix=0.5, osc_2_detune=10, stereo_width=1.0))
    assert np.abs(mono[0] - mono[1]).max() < 1e-9
    assert np.abs(wide[0] - wide[1]).mean() > 0.01
    assert np.abs(two_osc[0] - two_osc[1]).mean() > 0.01  # one voice: osc1 left, osc2 right


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
    assert len(P.NAMES) == 46 and len(set(P.NAMES)) == 46


def test_cross_mod_with_sync_stays_harmonic():
    audio = render(_v3(osc_2_waveform=2, osc_mix=0.7, osc_sync=1, osc_2_coarse=7, fm_amount=12), note=60)
    spec, freqs = _spectrum(audio.mean(axis=0), size=65536)
    f0 = 440 * 2 ** ((60 - 69) / 12)
    band = (freqs > 200) & (freqs < 12000)
    off = np.abs(freqs / f0 - np.round(freqs / f0)) * f0 > 10
    assert spec[band & off].sum() / spec[band].sum() < 0.1


def test_fdn_reverb_is_smoother_than_freeverb():
    from src.synth.engine import _reverb_stereo
    from src.synth.engine_v3 import _reverb_fdn

    x = np.zeros(int(3 * SR))
    x[:2400] = np.random.default_rng(0).standard_normal(2400)

    def ripple(tail):
        seg = tail[int(0.4 * SR):int(2.4 * SR)]
        db = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hanning(seg.size))) + 1e-12)
        freqs = np.fft.rfftfreq(seg.size, 1 / SR)
        idx = np.flatnonzero((freqs > 200) & (freqs < 8000))[::50]
        smooth = [db[(freqs > f * 2 ** (-1 / 6)) & (freqs < f * 2 ** (1 / 6))].mean() for f in freqs[idx]]
        return np.std(db[idx] - smooth)

    left, _ = _reverb_stereo(x.copy(), x.copy(), 1.0, SR, False)
    fdn = _reverb_fdn(np.stack([x, x]), 1.0, 0.9, SR)[0]
    assert ripple(fdn - 0.6 * x) < ripple(left - 0.6 * x) - 4.0
