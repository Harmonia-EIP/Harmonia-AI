"""Offline renderer of the Harmonia synth, faithful to the app's C++ engine.

Ports, sample by sample:
- HarmoniaVoice::renderNextBlock (Harmonia-APP/src/components/Synth.cpp)
- juce::ADSR, juce::dsp::StateVariableTPTFilter, juce::Reverb (Freeverb + 10 ms parameter smoothing)
- HarmoniaAudioProcessor::processBlock reverb settings (Harmonia-APP/src/PluginProcessor.cpp)

One note is rendered per call (the app's 8-voice polyphony is irrelevant for a single note).
"""

from __future__ import annotations

import math

import numba
import numpy as np

from src.charter import PARAM_INDEX
from src.synth.params import to_physical

SAMPLE_RATE = 48000

# Engine behaviours. ENGINE_APP_1_0 reproduces the app up to 2026-10, ENGINE_APP_1_1 the fixed engine:
# - filter Q: 1.0 uses the 0..0.95 resonance value directly as the SVF Q (0 -> NaN);
#   1.1 maps it to Q = 0.707 * 17^(resonance / 0.95), i.e. 0.707 (flat) .. ~12.
# - envelopes: 1.0 calls ADSR::setParameters() at every block, so right after note-off the release
#   rate becomes sustain / release (a note with sustain 0 stops dead, the filter envelope drops to 0);
#   1.1 releases from the level reached at note-off, as juce::ADSR::noteOff() intends.
ENGINE_APP_1_0 = 0
ENGINE_APP_1_1 = 1
Q_MODE_APP = ENGINE_APP_1_0
Q_MODE_MAPPED = ENGINE_APP_1_1

_IDX = {name: idx for name, idx in PARAM_INDEX.items()}

# ADSR states
_IDLE, _ATTACK, _DECAY, _SUSTAIN, _RELEASE = 0, 1, 2, 3, 4


@numba.njit(cache=True)
def _adsr_rate(distance, seconds, sr):
    return distance / (seconds * sr) if seconds > 0.0 else -1.0


@numba.njit(cache=True)
def _adsr_next(state, value, attack_rate, decay_rate, release_rate, sustain):
    if state == _IDLE:
        return state, 0.0
    if state == _ATTACK:
        value += attack_rate
        if value >= 1.0:
            value = 1.0
            state = _DECAY if decay_rate > 0.0 else _SUSTAIN
    elif state == _DECAY:
        value -= decay_rate
        if value <= sustain:
            value = sustain
            state = _SUSTAIN
    elif state == _SUSTAIN:
        value = sustain
    elif state == _RELEASE:
        value -= release_rate
        if value <= 0.0:
            state = _IDLE
            value = 0.0
            return state, value
    return state, value


@numba.njit(cache=True)
def _render_wave(wave, phase):
    if wave == 0:
        return math.sin(phase * 2.0 * math.pi)
    if wave == 1:
        return 4.0 * abs(phase - 0.5) - 1.0
    if wave == 2:
        return 2.0 * phase - 1.0
    return -1.0 if phase < 0.5 else 1.0


@numba.njit(cache=True)
def _render_voice(p, note, velocity, hold_samples, total_samples, sr, q_mode, seed):
    out = np.zeros(total_samples)
    np.random.seed(seed)

    w1 = int(p[0])
    w2 = int(p[1])
    mix = min(1.0, max(0.0, p[2]))
    detune = p[3]
    noise = min(1.0, max(0.0, p[4]))
    base_cutoff = min(20000.0, max(20.0, p[5]))
    resonance = min(0.95, max(0.0, p[6]))
    ftype = int(p[7])
    attack, decay, sustain, release = p[8] * 0.001, p[9] * 0.001, p[10], p[11] * 0.001
    fenv_amount = p[12]
    fenv_decay = p[13] * 0.001
    lfo_rate = min(20.0, max(0.1, p[14]))
    lfo_to_pitch, lfo_to_cutoff, vel_to_filter = p[15], p[16], p[17]
    distortion = p[18]

    if q_mode == Q_MODE_MAPPED:
        q = 0.707 * 17.0 ** (resonance / 0.95)
    else:
        q = resonance
    r2 = 1.0 / q if q > 0.0 else np.inf

    base_freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
    vel = min(1.0, max(0.0, velocity))

    # Amp ADSR (juce::ADSR). A fresh voice holds default parameters at noteOn, so it starts in attack.
    a_rate = _adsr_rate(1.0, attack, sr)
    d_rate = _adsr_rate(1.0 - sustain, decay, sr)
    a_state, a_val = _ATTACK, 0.0
    if a_rate <= 0.0:
        a_state = _DECAY if d_rate > 0.0 else _SUSTAIN
    a_rel_rate = 0.0

    # Filter envelope: AD only (attack 1 ms, sustain 0, release 50 ms).
    f_attack, f_sustain, f_release = 0.001, 0.0, 0.05
    fa_rate = _adsr_rate(1.0, f_attack, sr)
    fd_rate = _adsr_rate(1.0 - f_sustain, fenv_decay, sr)
    f_state, f_val = _ATTACK, 0.0
    f_rel_rate = 0.0

    # Oscillator/LFO phases are accumulated in float32 like the C++ engine: in float64 the
    # waveform edges drift by a sample after a few seconds (inaudible, but breaks parity tests).
    f32 = np.float32
    base_freq = f32(base_freq)
    detune_ratio = f32(2.0 ** (detune / 1200.0))
    lfo_inc = f32(f32(lfo_rate) / f32(sr))
    inv_sr = f32(sr)
    phase1 = f32(0.0)
    phase2 = f32(0.0)
    lfo_phase = f32(0.0)
    one = f32(1.0)
    s1 = 0.0
    s2 = 0.0
    gain = 0.4 + 0.6 * vel

    for i in range(total_samples):
        if i == hold_samples:
            if a_state != _IDLE:
                if q_mode == ENGINE_APP_1_0:
                    a_rel_rate = _adsr_rate(sustain, release, sr)
                else:
                    a_rel_rate = a_val / (release * sr)
                a_state = _RELEASE
                if a_rel_rate <= 0.0:
                    a_state, a_val = _IDLE, 0.0
            if f_state != _IDLE:
                if q_mode == ENGINE_APP_1_0:
                    f_state, f_val = _IDLE, 0.0
                else:
                    f_rel_rate = f_val / (f_release * sr)
                    f_state = _RELEASE
        if a_state == _IDLE and i >= hold_samples:
            break

        lfo = math.sin(lfo_phase * 2.0 * math.pi)
        lfo_phase = f32(lfo_phase + lfo_inc)
        if lfo_phase >= one:
            lfo_phase = f32(lfo_phase - one)

        pitch_ratio = f32(2.0 ** ((lfo * lfo_to_pitch * 50.0) / 1200.0))
        f1 = f32(base_freq * pitch_ratio)
        f2 = f32(f32(base_freq * pitch_ratio) * detune_ratio)

        smp1 = _render_wave(w1, phase1)
        smp2 = _render_wave(w2, phase2)
        n = np.random.uniform(-1.0, 1.0)
        sample = (smp1 * (1.0 - mix) + smp2 * mix) * (1.0 - noise) + n * noise

        phase1 = f32(phase1 + f32(f1 / inv_sr))
        phase2 = f32(phase2 + f32(f2 / inv_sr))
        if phase1 >= one:
            phase1 = f32(phase1 - one)
        if phase2 >= one:
            phase2 = f32(phase2 - one)

        f_state, f_val = _adsr_next(f_state, f_val, fa_rate, fd_rate, f_rel_rate, f_sustain)
        octaves = fenv_amount * f_val * 4.0 + lfo * lfo_to_cutoff * 2.0 + vel_to_filter * (vel - 0.5) * 4.0
        cutoff = min(20000.0, max(20.0, base_cutoff * 2.0 ** octaves))

        g = math.tan(math.pi * cutoff / sr)
        h = 1.0 / (1.0 + r2 * g + g * g)
        y_hp = h * (sample - s1 * (g + r2) - s2)
        y_bp = y_hp * g + s1
        s1 = y_hp * g + y_bp
        y_lp = y_bp * g + s2
        s2 = y_bp * g + y_lp
        if ftype == 1:
            sample = y_bp
        elif ftype == 2:
            sample = y_hp
        else:
            sample = y_lp

        if distortion > 0.0001:
            drive = 1.0 + distortion * 12.0
            dirty = math.tanh(sample * drive) / math.tanh(drive) * 0.95
            sample = sample * (1.0 - distortion) + dirty * distortion

        a_state, a_val = _adsr_next(a_state, a_val, a_rate, d_rate, a_rel_rate, sustain)
        out[i] = sample * a_val * gain

    return out


@numba.njit(cache=True)
def _smoothed(current, target, steps, k):
    # juce::SmoothedValue (linear): reaches the target after `steps` samples.
    if k >= steps:
        return target
    return current + (target - current) * (k + 1) / steps


@numba.njit(cache=True)
def _reverb_stereo(left, right, wet_level, sr, smooth_from_defaults):
    n = left.shape[0]
    comb_tunings = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617])
    allpass_tunings = np.array([556, 441, 341, 225])
    spread = 23
    isr = int(sr)

    comb_sizes = np.empty((2, 8), dtype=np.int64)
    for i in range(8):
        comb_sizes[0, i] = (isr * comb_tunings[i]) // 44100
        comb_sizes[1, i] = (isr * (comb_tunings[i] + spread)) // 44100
    ap_sizes = np.empty((2, 4), dtype=np.int64)
    for i in range(4):
        ap_sizes[0, i] = (isr * allpass_tunings[i]) // 44100
        ap_sizes[1, i] = (isr * (allpass_tunings[i] + spread)) // 44100

    max_comb = comb_sizes.max()
    max_ap = ap_sizes.max()
    comb_buf = np.zeros((2, 8, max_comb))
    comb_idx = np.zeros((2, 8), dtype=np.int64)
    comb_last = np.zeros((2, 8))
    ap_buf = np.zeros((2, 4, max_ap))
    ap_idx = np.zeros((2, 4), dtype=np.int64)

    # processBlock: wetLevel = wet, dryLevel = 1 - 0.7 wet, roomSize = 0.55 + 0.35 wet, damping 0.4, width 1.
    room = 0.55 + wet_level * 0.35
    target_damp = 0.4 * 0.4
    target_feedback = room * 0.28 + 0.7
    target_dry = (1.0 - wet_level * 0.7) * 2.0
    target_wet1 = 0.5 * (wet_level * 3.0) * 2.0
    target_wet2 = 0.0
    # In a running app the 10 ms smoothing settled long before the note: start at the targets.
    # A freshly prepared plugin instead ramps from juce::Reverb's default Parameters().
    steps = int(math.floor(0.01 * sr))
    if smooth_from_defaults:
        start_damp, start_feedback, start_dry, start_wet1, start_wet2 = 0.5 * 0.4, 0.5 * 0.28 + 0.7, 0.4 * 2.0, 0.99, 0.0
    else:
        start_damp, start_feedback, start_dry, start_wet1, start_wet2 = target_damp, target_feedback, target_dry, target_wet1, target_wet2
    in_gain = 0.015

    out_l = np.empty(n)
    out_r = np.empty(n)
    for i in range(n):
        inp = (left[i] + right[i]) * in_gain
        damp = _smoothed(start_damp, target_damp, steps, i)
        fb = _smoothed(start_feedback, target_feedback, steps, i)
        acc_l = 0.0
        acc_r = 0.0
        for ch in range(2):
            acc = 0.0
            for j in range(8):
                idx = comb_idx[ch, j]
                output = comb_buf[ch, j, idx]
                comb_last[ch, j] = output * (1.0 - damp) + comb_last[ch, j] * damp
                comb_buf[ch, j, idx] = inp + comb_last[ch, j] * fb
                comb_idx[ch, j] = (idx + 1) % comb_sizes[ch, j]
                acc += output
            for j in range(4):
                idx = ap_idx[ch, j]
                buffered = ap_buf[ch, j, idx]
                ap_buf[ch, j, idx] = acc + buffered * 0.5
                ap_idx[ch, j] = (idx + 1) % ap_sizes[ch, j]
                acc = buffered - acc
            if ch == 0:
                acc_l = acc
            else:
                acc_r = acc
        dry = _smoothed(start_dry, target_dry, steps, i)
        wet1 = _smoothed(start_wet1, target_wet1, steps, i)
        wet2 = _smoothed(start_wet2, target_wet2, steps, i)
        out_l[i] = acc_l * wet1 + acc_r * wet2 + left[i] * dry
        out_r[i] = acc_r * wet1 + acc_l * wet2 + right[i] * dry
    return out_l, out_r


@numba.njit(cache=True)
def render_physical(p, note, velocity, hold_seconds, total_seconds, sr, q_mode, seed, smooth_from_defaults=False):
    """Render one note from physical parameter values; returns (left, right) float64 arrays."""
    total = int(total_seconds * sr)
    hold = int(hold_seconds * sr)
    voice = _render_voice(p, note, velocity, hold, total, sr, q_mode, seed)
    return _reverb_stereo(voice, voice.copy(), p[19], sr, smooth_from_defaults)


def render_preset(
    normalized,
    note: int = 60,
    velocity: float = 100 / 127,
    hold_seconds: float = 1.5,
    total_seconds: float = 4.0,
    sr: int = SAMPLE_RATE,
    q_mode: int = Q_MODE_APP,
    seed: int = 0x4321,
) -> np.ndarray:
    """Render a charter preset (20 normalized values) to a mono float32 signal."""
    physical = to_physical(normalized)
    left, right = render_physical(physical, note, velocity, hold_seconds, total_seconds, sr, q_mode, seed)
    return (0.5 * (left + right)).astype(np.float32)


@numba.njit(cache=True, parallel=True)
def render_batch_physical(params, notes, velocity, hold_seconds, total_seconds, sr, q_mode, seed):
    """Render every preset at every note; returns float32 array (n_presets, n_notes, samples)."""
    n = params.shape[0]
    m = notes.shape[0]
    total = int(total_seconds * sr)
    out = np.zeros((n, m, total), dtype=np.float32)
    for k in numba.prange(n * m):
        i = k // m
        j = k % m
        left, right = render_physical(params[i], notes[j], velocity, hold_seconds, total_seconds, sr, q_mode, seed + k, False)
        for t in range(total):
            out[i, j, t] = 0.5 * (left[t] + right[t])
    return out


@numba.njit(cache=True)
def _audible_rms(signal, sr, cutoff_hz):
    # Two cascaded one-pole high-pass filters (12 dB/oct): energy the listener actually hears.
    a = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0
    acc = 0.0
    for t in range(signal.shape[0]):
        x = signal[t]
        if not math.isfinite(x):
            return np.nan
        h1 = a * (y1 + x - x1)
        x1 = x
        y1 = h1
        h2 = a * (y2 + h1 - x2)
        x2 = h1
        y2 = h2
        acc += h2 * h2
    return math.sqrt(acc / signal.shape[0])


@numba.njit(cache=True, parallel=True)
def audible_rms_batch_physical(params, notes, velocity, hold_seconds, total_seconds, sr, q_mode, cutoff_hz):
    """Audible (high-passed) RMS of every preset at every note, without keeping the audio."""
    n = params.shape[0]
    m = notes.shape[0]
    out = np.zeros((n, m), dtype=np.float32)
    for k in numba.prange(n * m):
        i = k // m
        j = k % m
        left, right = render_physical(params[i], notes[j], velocity, hold_seconds, total_seconds, sr, q_mode, k, False)
        out[i, j] = _audible_rms(0.5 * (left + right), sr, cutoff_hz)
    return out
