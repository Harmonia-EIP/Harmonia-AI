"""Harmonia v3 analog engine (reference implementation for the app's C++ port).

One note per call, stereo out. Signal flow per voice:
  2 oscillators (sine/triangle/band-limited saw and pulse, coarse tuning, OB-X style cross-modulation of
  osc2's pitch by osc1, osc2 hard-synced to osc1, ring modulation, slight analog drift) x unison (1..16 detuned voices) spread in stereo + noise
  -> state-variable filter (12 or 24 dB, envelope/LFO/velocity/keyboard tracking) -> distortion
  -> amp ADSR (velocity, tremolo), all at twice the sample rate against aliasing (sync, cross-mod and
  distortion would otherwise fold back as grit); then chorus -> delay -> reverb (feedback delay network).
Parameters: src/synth/v3_params.py.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from src.synth import v3_params as P

SAMPLE_RATE = 48000
_IDLE, _ATTACK, _DECAY, _SUSTAIN, _RELEASE = 0, 1, 2, 3, 4
MAX_UNISON = 16
DRIFT_CENTS = 2.0  # analog-style pitch wander per oscillator
DRIFT_EVERY = 64  # samples between drift updates
PITCH_ENV_SHAPE = 5.0
OVERSAMPLE = 2  # the voice runs at twice the sample rate, then a half-band filter brings it back
OSC_SPLIT = 0.5  # at full width, osc1 sits this far left of its voice and osc2 this far right
CHORUS_RATE, CHORUS_CENTER_MS, CHORUS_DEPTH_MS = 0.5, 3.5, 1.75
DELAY_DAMP_HZ = 6000.0


@numba.njit(cache=True)
def _poly_blep(t, dt):
    if dt <= 0.0:
        return 0.0
    if t < dt:
        t = t / dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@numba.njit(cache=True)
def _osc(wave, phase, dt, pw):
    """One oscillator sample; saw and pulse are band-limited with PolyBLEP."""
    if wave == 0:
        return math.sin(2.0 * math.pi * phase)
    if wave == 1:
        return 4.0 * abs(phase - 0.5) - 1.0
    if wave == 2:
        return 2.0 * phase - 1.0 - _poly_blep(phase, dt)
    value = -1.0 if phase < pw else 1.0
    shifted = phase - pw
    if shifted < 0.0:
        shifted += 1.0
    return value - _poly_blep(phase, dt) + _poly_blep(shifted, dt)


FLOOR = 1e-4  # -80 dB: an exponential segment reaches its target within this distance at its time


@numba.njit(cache=True)
def _adsr_step(state, value, a_rate, d_coef, sustain, r_coef):
    """Linear attack, exponential decay (to the sustain level) and release, like analog envelopes."""
    if state == _ATTACK:
        value += a_rate
        if value >= 1.0:
            value = 1.0
            state = _DECAY
    elif state == _DECAY:
        value = sustain + (value - sustain) * d_coef
        if value - sustain <= FLOOR:
            value = sustain
            state = _SUSTAIN
    elif state == _SUSTAIN:
        value = sustain
    elif state == _RELEASE:
        value *= r_coef
        if value <= FLOOR:
            value = 0.0
            state = _IDLE
    return state, value


@numba.njit(cache=True)
def _exp_coef(seconds, sr):
    """Per-sample factor that shrinks a distance to FLOOR in `seconds`."""
    return math.exp(math.log(FLOOR) / (seconds * sr)) if seconds > 0.0 else 0.0


@numba.njit(cache=True)
def _adsr_start(attack_s, decay_s, sustain, sr):
    a_rate = 1.0 / (attack_s * sr) if attack_s > 0.0 else -1.0
    d_coef = _exp_coef(decay_s, sr)
    if a_rate > 0.0:
        return _ATTACK, 0.0, a_rate, d_coef
    return _DECAY, 1.0, a_rate, d_coef


@numba.njit(cache=True)
def _svf(x, g, r2, state, k):
    """juce::dsp::StateVariableTPTFilter step; returns (lowpass, bandpass, highpass)."""
    h = 1.0 / (1.0 + r2 * g + g * g)
    s1 = state[k, 0]
    s2 = state[k, 1]
    y_hp = h * (x - s1 * (g + r2) - s2)
    y_bp = y_hp * g + s1
    state[k, 0] = y_hp * g + y_bp
    y_lp = y_bp * g + s2
    state[k, 1] = y_bp * g + y_lp
    return y_lp, y_bp, y_hp


@numba.njit(cache=True)
def _lfo_value(wave, phase, held):
    if wave == 0:
        return math.sin(2.0 * math.pi * phase)
    if wave == 1:
        return 1.0 - 4.0 * abs(phase - 0.5)
    if wave == 2:
        return 1.0 - 2.0 * phase
    if wave == 3:
        return 1.0 if phase < 0.5 else -1.0
    return held


@numba.njit(cache=True)
def _render_voice(p, note, velocity, hold, total, sr, seed):
    out = np.zeros((2, total))
    np.random.seed(seed)

    w1, w2 = int(p[0]), int(p[1])
    mix, detune, noise = p[2], p[3], p[4]
    base_cutoff, resonance, ftype = p[5], p[6], int(p[7])
    attack, decay, sustain, release = p[8] * 1e-3, p[9] * 1e-3, p[10], p[11] * 1e-3
    fenv_amount, fenv_decay = p[12], p[13] * 1e-3
    lfo_rate, lfo_pitch_cents, lfo_cutoff_oct, vel_filter = p[14], p[15], p[16], p[17]
    distortion = p[18]
    coarse1, coarse2, pw0, sync, fm, ring = p[20], p[21], p[22], p[23] > 0.5, p[24], p[25]
    n_uni = int(min(MAX_UNISON, max(1, p[26])))
    uni_cents = p[27]
    slope24 = p[28] > 0.5
    keytrack = p[29]
    fenv_attack, fenv_sustain, fenv_release = p[30] * 1e-3, p[31], p[32] * 1e-3
    vel_amp = p[33]
    penv_amount, penv_decay = p[34], p[35] * 1e-3
    lfo_wave, lfo_delay = int(p[36]), p[37] * 1e-3
    lfo_amp, lfo_pw = p[38], p[39]
    width = min(1.0, max(0.0, p[45]))

    vel = min(1.0, max(0.0, velocity))
    q = 0.707 * 17.0 ** (min(0.95, max(0.0, resonance)) / 0.95)  # Harmonia-App#40 mapping
    r2 = 1.0 / q
    r2_flat = 1.0 / 0.7071067811865476
    base_freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
    gain = (1.0 - vel_amp) + vel_amp * vel
    key_oct = keytrack * (note - 60) / 12.0 + vel_filter * (vel - 0.5) * 4.0

    a_state, a_val, a_rate, d_coef = _adsr_start(attack, decay, sustain, sr)
    f_state, f_val, fa_rate, fd_coef = _adsr_start(fenv_attack, fenv_decay, fenv_sustain, sr)
    a_rel = _exp_coef(release, sr)
    f_rel = _exp_coef(fenv_release, sr)

    # unison voices: detune spread, stereo position and start phase (a single voice starts at 0);
    # within each voice osc1 leans left and osc2 right as the width grows
    uni_off = np.zeros(n_uni)
    gains = np.ones((n_uni, 4))  # osc1 L, osc1 R, osc2 L, osc2 R
    ph1 = np.zeros(n_uni)
    ph2 = np.zeros(n_uni)
    for u in range(n_uni):
        pos = 2.0 * u / (n_uni - 1) - 1.0 if n_uni > 1 else 0.0
        uni_off[u] = pos * uni_cents / 100.0
        for k, split in ((0, -OSC_SPLIT), (2, OSC_SPLIT)):
            pan = min(1.0, max(-1.0, (pos + split) * width))
            angle = (pan + 1.0) * math.pi / 4.0
            gains[u, k] = math.cos(angle) * math.sqrt(2.0)
            gains[u, k + 1] = math.sin(angle) * math.sqrt(2.0)
        if n_uni > 1:
            ph1[u] = np.random.random()
            ph2[u] = np.random.random()
    uni_norm = 1.0 / math.sqrt(n_uni)
    # analog drift: each oscillator of each voice wanders by a couple of cents (two slow sines,
    # random rates and phases), updated at control rate
    drift_rate = np.random.uniform(0.07, 0.35, (n_uni, 2, 2))
    drift_phase = np.random.uniform(0.0, 2.0 * math.pi, (n_uni, 2, 2))
    drift = np.zeros((n_uni, 2))

    filt = np.zeros((4, 2))  # (channel x stage, state)
    lfo_phase = 0.0
    lfo_held = np.random.uniform(-1.0, 1.0)
    for i in range(total):
        if i == hold:
            if a_state != _IDLE:
                a_state = _RELEASE
            if f_state != _IDLE:
                f_state = _RELEASE
        if a_state == _IDLE and i >= hold:
            break
        t = i / sr

        # LFO (with fade-in)
        lfo = _lfo_value(lfo_wave, lfo_phase, lfo_held)
        if lfo_delay > 0.0 and t < lfo_delay:
            lfo *= t / lfo_delay
        lfo_phase += lfo_rate / sr
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0
            lfo_held = np.random.uniform(-1.0, 1.0)

        semis = lfo * lfo_pitch_cents / 100.0
        if penv_amount != 0.0 and penv_decay > 0.0:
            semis += penv_amount * math.exp(-PITCH_ENV_SHAPE * t / penv_decay)  # ~0.7 % left at the decay time
        if i % DRIFT_EVERY == 0:
            for u in range(n_uni):
                for k in range(2):
                    drift[u, k] = DRIFT_CENTS / 150.0 * (
                        math.sin(2.0 * math.pi * drift_rate[u, k, 0] * t + drift_phase[u, k, 0])
                        + 0.5 * math.sin(2.0 * math.pi * drift_rate[u, k, 1] * t + drift_phase[u, k, 1]))
        pw = min(0.95, max(0.05, pw0 + lfo * lfo_pw * 0.45))

        left = 0.0
        right = 0.0
        for u in range(n_uni):
            st = semis + uni_off[u]
            dt1 = base_freq * 2.0 ** ((coarse1 + st + drift[u, 0]) / 12.0) / sr
            o1 = _osc(w1, ph1[u], dt1, pw)
            # cross-modulation as on the OB-X: osc1 bends osc2's pitch (fm semitones per unit of osc1);
            # with osc2 synced to osc1 the result stays harmonic
            dt2 = base_freq * 2.0 ** ((coarse2 + st + drift[u, 1] + fm * o1) / 12.0 + detune / 1200.0) / sr
            dt2 = min(dt2, 0.45)
            o2 = _osc(w2, ph2[u], dt2, pw)
            a1 = o1 * (1.0 - mix)
            a2 = o2 * mix
            if ring > 0.0:
                a1 *= 1.0 - ring
                a2 *= 1.0 - ring
                centre = o1 * o2 * ring
                left += centre
                right += centre
            left += a1 * gains[u, 0] + a2 * gains[u, 2]
            right += a1 * gains[u, 1] + a2 * gains[u, 3]
            ph1[u] += dt1
            wrapped = ph1[u] >= 1.0
            if wrapped:
                ph1[u] -= 1.0
            if sync and wrapped:
                ph2[u] = ph1[u] / dt1 * dt2 if dt1 > 0.0 else 0.0
            else:
                ph2[u] += dt2
                if ph2[u] >= 1.0:
                    ph2[u] -= math.floor(ph2[u])
        left *= uni_norm
        right *= uni_norm
        if noise > 0.0:
            n = np.random.uniform(-1.0, 1.0)
            left = left * (1.0 - noise) + n * noise
            right = right * (1.0 - noise) + n * noise

        f_state, f_val = _adsr_step(f_state, f_val, fa_rate, fd_coef, fenv_sustain, f_rel)
        octaves = fenv_amount * f_val + lfo * lfo_cutoff_oct + key_oct
        cutoff = min(20000.0, max(20.0, base_cutoff * 2.0 ** octaves))
        g = math.tan(math.pi * min(cutoff, 0.49 * sr) / sr)

        a_state, a_val = _adsr_step(a_state, a_val, a_rate, d_coef, sustain, a_rel)
        amp = a_val * gain
        if lfo_amp > 0.0:
            amp *= 1.0 - lfo_amp * 0.5 * (1.0 - lfo)

        for ch in range(2):
            x = left if ch == 0 else right
            lp, bp, hp = _svf(x, g, r2, filt, ch)
            y = lp if ftype == 0 else (bp if ftype == 1 else hp)
            if slope24:
                lp, bp, hp = _svf(y, g, r2_flat, filt, ch + 2)
                y = lp if ftype == 0 else (bp if ftype == 1 else hp)
            if distortion > 0.0001:
                drive = 1.0 + distortion * 12.0
                y = y * (1.0 - distortion) + math.tanh(y * drive) / math.tanh(drive) * 0.95 * distortion
            out[ch, i] = y * amp
    return out


@numba.njit(cache=True)
def _read_delay(buf, pos, delay_samples):
    size = buf.shape[0]
    d = pos - delay_samples
    while d < 0.0:
        d += size
    i0 = int(d)
    frac = d - i0
    i1 = (i0 + 1) % size
    return buf[i0] * (1.0 - frac) + buf[i1] * frac


@numba.njit(cache=True)
def _chorus(stereo, mix, sr):
    """Juno-style stereo chorus: one modulated delay per channel, LFOs in opposite phase."""
    n = stereo.shape[1]
    size = int(0.02 * sr)
    bufs = np.zeros((2, size))
    out = np.empty_like(stereo)
    for i in range(n):
        lfo = math.sin(2.0 * math.pi * CHORUS_RATE * i / sr)
        for ch in range(2):
            bufs[ch, i % size] = stereo[ch, i]
            ms = CHORUS_CENTER_MS + CHORUS_DEPTH_MS * (lfo if ch == 0 else -lfo)
            wet = _read_delay(bufs[ch], i % size, ms * 1e-3 * sr)
            out[ch, i] = stereo[ch, i] * (1.0 - 0.5 * mix) + wet * 0.5 * mix
    return out


@numba.njit(cache=True)
def _delay(stereo, time_ms, feedback, mix, sr):
    n = stereo.shape[1]
    delay_samples = max(1.0, time_ms * 1e-3 * sr)
    size = int(delay_samples) + 2
    bufs = np.zeros((2, size))
    damp = math.exp(-2.0 * math.pi * DELAY_DAMP_HZ / sr)
    lp = np.zeros(2)
    out = np.empty_like(stereo)
    for i in range(n):
        for ch in range(2):
            wet = _read_delay(bufs[ch], i % size, delay_samples)
            lp[ch] = wet * (1.0 - damp) + lp[ch] * damp
            bufs[ch, i % size] = stereo[ch, i] + lp[ch] * feedback
            out[ch, i] = stereo[ch, i] + wet * mix
    return out


def _half_band(taps: int = 39, cutoff_hz: float = 21000.0, rate: float = 96000.0) -> np.ndarray:
    """Kaiser-windowed sinc low-pass for the 2x -> 1x decimation (flat to ~19 kHz, ~60 dB above 28 kHz)."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2.0 * cutoff_hz / rate * n) * np.kaiser(taps, 8.0)
    return h / h.sum()


DECIMATOR = _half_band()


@numba.njit(cache=True)
def _decimate(x, h):
    n_out = x.shape[1] // OVERSAMPLE
    half = h.shape[0] // 2
    out = np.zeros((x.shape[0], n_out))
    for ch in range(x.shape[0]):
        for i in range(n_out):
            centre = i * OVERSAMPLE
            acc = 0.0
            for k in range(h.shape[0]):
                j = centre + k - half
                if 0 <= j < x.shape[1]:
                    acc += h[k] * x[ch, j]
            out[ch, i] = acc
    return out


# Feedback-delay-network reverb: 8 modulated delay lines mixed by a Householder matrix after 4 diffusing
# all-passes. Smoother than Freeverb's combs, whose fixed resonances sound metallic on sustained notes.
FDN_MS = np.array([31.7, 37.3, 41.9, 47.3, 53.9, 59.1, 67.3, 73.1])
FDN_MOD_HZ = np.array([0.11, 0.17, 0.23, 0.29, 0.13, 0.19, 0.31, 0.37])
FDN_MOD_MS = 0.4
DIFFUSER_MS = np.array([4.7, 3.6, 12.7, 9.3])
DIFFUSER_GAIN = 0.6
REVERB_PREDELAY_MS = 12.0
REVERB_DAMP_HZ = 7000.0
REVERB_WET_GAIN = 2.4  # median wet level of the v2 (Freeverb) reverb at the same mix (saw notes, 3 pitches)
# Tail spectral ripple (std of dB around a 1/3-octave average, white noise in): Freeverb 19.0, this 11.8.


@numba.njit(cache=True)
def _reverb_fdn(stereo, wet, size, sr):
    n = stereo.shape[1]
    dry_gain = (1.0 - wet * 0.7) * 2.0  # v2 dry law (juce::Reverb's dry level x 2)
    if wet <= 0.0:
        return stereo * dry_gain
    t60 = 0.3 + 11.7 * size * size  # seconds
    scale = 0.6 + 0.6 * size
    lengths = FDN_MS * 1e-3 * sr * scale
    fb = np.empty(8)
    for j in range(8):
        fb[j] = 10.0 ** (-3.0 * lengths[j] / (t60 * sr))
    damp = math.exp(-2.0 * math.pi * REVERB_DAMP_HZ / sr)
    # energy-normalized input: the impulse response carries about the same energy whatever the decay time
    in_gain = math.sqrt(1.0 - np.mean(fb) ** 2)
    size_buf = int(lengths.max() + FDN_MOD_MS * 1e-3 * sr) + 4
    lines = np.zeros((8, size_buf))
    lp = np.zeros(8)
    diff_len = np.empty(4, dtype=np.int64)
    for d in range(4):
        diff_len[d] = int(DIFFUSER_MS[d] * 1e-3 * sr)
    diff = np.zeros((4, diff_len.max() + 1))
    diff_pos = np.zeros(4, dtype=np.int64)
    pre_len = int(REVERB_PREDELAY_MS * 1e-3 * sr) + 1
    pre = np.zeros(pre_len)
    out = np.empty_like(stereo)
    taps = np.zeros(8)
    mods = np.zeros(8)
    for i in range(n):
        x = 0.5 * (stereo[0, i] + stereo[1, i]) * in_gain
        delayed = pre[i % pre_len]
        pre[i % pre_len] = x
        for d in range(4):  # Schroeder all-pass diffusers
            k = diff_pos[d]
            buf = diff[d, k]
            v = delayed + DIFFUSER_GAIN * buf
            diff[d, k] = v
            delayed = buf - DIFFUSER_GAIN * v
            diff_pos[d] = (k + 1) % diff_len[d]
        total = 0.0
        if i % 16 == 0:  # slow modulation: control rate is enough
            for j in range(8):
                mods[j] = FDN_MOD_MS * 1e-3 * sr * math.sin(2.0 * math.pi * FDN_MOD_HZ[j] * i / sr)
        for j in range(8):
            y = _read_delay(lines[j], i % size_buf, lengths[j] + mods[j])
            lp[j] = y * (1.0 - damp) + lp[j] * damp
            taps[j] = lp[j] * fb[j]
            total += taps[j]
        total *= 2.0 / 8.0
        for j in range(8):  # Householder feedback matrix: I - 2/N
            lines[j, i % size_buf] = taps[j] - total + delayed
        left = taps[0] - taps[1] + taps[2] - taps[3] + taps[4] - taps[5] + taps[6] - taps[7]
        right = taps[0] + taps[1] - taps[2] - taps[3] + taps[4] + taps[5] - taps[6] - taps[7]
        out[0, i] = stereo[0, i] * dry_gain + left * wet * REVERB_WET_GAIN / 2.0
        out[1, i] = stereo[1, i] * dry_gain + right * wet * REVERB_WET_GAIN / 2.0
    return out


@numba.njit(cache=True)
def _render_physical(p, note, velocity, hold_seconds, total_seconds, sr, seed, decimator):
    total = int(total_seconds * sr)
    fast = sr * OVERSAMPLE
    voice = _render_voice(p, note, velocity, int(hold_seconds * fast), total * OVERSAMPLE, fast, seed)
    stereo = _decimate(voice, decimator)
    if p[40] > 0.0:
        stereo = _chorus(stereo, p[40], sr)
    if p[43] > 0.0:
        stereo = _delay(stereo, p[41], p[42], p[43], sr)
    return _reverb_fdn(stereo, p[19], p[44], sr)


def render_physical(p, note, velocity, hold_seconds, total_seconds, sr, seed):
    """One note from a physical v3 vector (v3_params order); returns a (2, samples) float64 array."""
    return _render_physical(np.asarray(p, dtype=np.float64), note, velocity, hold_seconds, total_seconds, sr,
                            seed, DECIMATOR)


def render(physical, note: int = 60, velocity: float = 100 / 127, hold_seconds: float = 1.5,
           total_seconds: float = 4.0, sr: int = SAMPLE_RATE, seed: int = 0x4321) -> np.ndarray:
    """(2, samples) float32 audio of one note for a physical v3 vector."""
    p = np.asarray(physical, dtype=np.float64)
    if p.shape != (P.N_PARAMS,):
        raise ValueError(f"expected {P.N_PARAMS} physical values, got {p.shape}")
    return render_physical(p, note, velocity, hold_seconds, total_seconds, sr, seed).astype(np.float32)


def render_batch(params, notes, velocity, hold_seconds, total_seconds, sr, seed):
    """Mono float32 (n_presets, n_notes, samples) for many physical vectors."""
    return _render_batch(np.asarray(params, dtype=np.float64), np.asarray(notes), velocity, hold_seconds,
                         total_seconds, sr, seed, DECIMATOR)


@numba.njit(cache=True, parallel=True)
def _render_batch(params, notes, velocity, hold_seconds, total_seconds, sr, seed, decimator):
    n, m = params.shape[0], notes.shape[0]
    total = int(total_seconds * sr)
    out = np.zeros((n, m, total), dtype=np.float32)
    for k in numba.prange(n * m):
        i, j = k // m, k % m
        stereo = _render_physical(params[i], notes[j], velocity, hold_seconds, total_seconds, sr, seed + k, decimator)
        for t in range(total):
            out[i, j, t] = 0.5 * (stereo[0, t] + stereo[1, t])
    return out
