"""Harmonia v3 analog engine (reference implementation for the app's C++ port).

One note per call, stereo out. Signal flow per voice:
  2 oscillators (sine/triangle/band-limited saw and pulse, coarse tuning, osc2 -> osc1 phase modulation,
  osc2 hard-synced to osc1, ring modulation) x unison (1..7 detuned voices spread in stereo) + noise
  -> state-variable filter (12 or 24 dB, envelope/LFO/velocity/keyboard tracking) -> distortion
  -> amp ADSR (velocity, tremolo); then chorus -> delay -> reverb (Freeverb, as in v2).
Parameters: src/synth/v3_params.py.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from src.synth import v3_params as P
from src.synth.engine import _reverb_stereo

SAMPLE_RATE = 48000
_IDLE, _ATTACK, _DECAY, _SUSTAIN, _RELEASE = 0, 1, 2, 3, 4
MAX_UNISON = 7
UNISON_SPREAD = 0.8  # stereo width of the unison voices (-1..1 pan)
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


@numba.njit(cache=True)
def _adsr_step(state, value, a_rate, d_rate, sustain, r_rate):
    if state == _ATTACK:
        value += a_rate
        if value >= 1.0:
            value = 1.0
            state = _DECAY if d_rate > 0.0 else _SUSTAIN
    elif state == _DECAY:
        value -= d_rate
        if value <= sustain:
            value = sustain
            state = _SUSTAIN
    elif state == _SUSTAIN:
        value = sustain
    elif state == _RELEASE:
        value -= r_rate
        if value <= 0.0:
            value = 0.0
            state = _IDLE
    return state, value


@numba.njit(cache=True)
def _adsr_start(attack_s, decay_s, sustain, sr):
    a_rate = 1.0 / (attack_s * sr) if attack_s > 0.0 else -1.0
    d_rate = (1.0 - sustain) / (decay_s * sr) if decay_s > 0.0 else -1.0
    if a_rate > 0.0:
        return _ATTACK, 0.0, a_rate, d_rate
    if d_rate > 0.0:
        return _DECAY, 1.0, a_rate, d_rate
    return _SUSTAIN, sustain, a_rate, d_rate


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

    vel = min(1.0, max(0.0, velocity))
    q = 0.707 * 17.0 ** (min(0.95, max(0.0, resonance)) / 0.95)  # Harmonia-App#40 mapping
    r2 = 1.0 / q
    r2_flat = 1.0 / 0.7071067811865476
    base_freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
    gain = (1.0 - vel_amp) + vel_amp * vel
    key_oct = keytrack * (note - 60) / 12.0 + vel_filter * (vel - 0.5) * 4.0

    a_state, a_val, a_rate, d_rate = _adsr_start(attack, decay, sustain, sr)
    f_state, f_val, fa_rate, fd_rate = _adsr_start(fenv_attack, fenv_decay, fenv_sustain, sr)
    a_rel = 0.0
    f_rel = 0.0

    # unison voices: detune spread, stereo position and start phase (a single voice starts at 0)
    uni_off = np.zeros(n_uni)
    uni_l = np.ones(n_uni)
    uni_r = np.ones(n_uni)
    ph1 = np.zeros(n_uni)
    ph2 = np.zeros(n_uni)
    if n_uni > 1:
        for u in range(n_uni):
            pos = 2.0 * u / (n_uni - 1) - 1.0
            uni_off[u] = pos * uni_cents / 100.0
            angle = (pos * UNISON_SPREAD + 1.0) * math.pi / 4.0
            uni_l[u] = math.cos(angle) * math.sqrt(2.0)
            uni_r[u] = math.sin(angle) * math.sqrt(2.0)
            ph1[u] = np.random.random()
            ph2[u] = np.random.random()
    uni_norm = 1.0 / math.sqrt(n_uni)

    filt = np.zeros((4, 2))  # (channel x stage, state)
    lfo_phase = 0.0
    lfo_held = np.random.uniform(-1.0, 1.0)
    for i in range(total):
        if i == hold:
            if a_state != _IDLE:
                a_rel = a_val / (release * sr) if release > 0.0 else a_val
                a_state = _RELEASE
            if f_state != _IDLE:
                f_rel = f_val / (fenv_release * sr) if fenv_release > 0.0 else f_val
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
        if penv_amount != 0.0 and penv_decay > 0.0 and t < penv_decay:
            semis += penv_amount * (1.0 - t / penv_decay)
        pw = min(0.95, max(0.05, pw0 + lfo * lfo_pw * 0.45))

        left = 0.0
        right = 0.0
        for u in range(n_uni):
            st = semis + uni_off[u]
            dt1 = base_freq * 2.0 ** ((coarse1 + st) / 12.0) / sr
            dt2 = base_freq * 2.0 ** ((coarse2 + st) / 12.0 + detune / 1200.0) / sr
            o2 = _osc(w2, ph2[u], dt2, pw)
            if fm > 0.0:
                pm = ph1[u] + fm * o2 / (2.0 * math.pi)
                pm -= math.floor(pm)
                o1 = _osc(w1, pm, dt1, pw)
            else:
                o1 = _osc(w1, ph1[u], dt1, pw)
            s = o1 * (1.0 - mix) + o2 * mix
            if ring > 0.0:
                s = s * (1.0 - ring) + o1 * o2 * ring
            left += s * uni_l[u]
            right += s * uni_r[u]
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

        f_state, f_val = _adsr_step(f_state, f_val, fa_rate, fd_rate, fenv_sustain, f_rel)
        octaves = fenv_amount * f_val * 4.0 + lfo * lfo_cutoff_oct + key_oct
        cutoff = min(20000.0, max(20.0, base_cutoff * 2.0 ** octaves))
        g = math.tan(math.pi * min(cutoff, 0.49 * sr) / sr)

        a_state, a_val = _adsr_step(a_state, a_val, a_rate, d_rate, sustain, a_rel)
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


@numba.njit(cache=True)
def render_physical(p, note, velocity, hold_seconds, total_seconds, sr, seed):
    """One note from a physical v3 vector (v3_params order); returns a (2, samples) float64 array."""
    total = int(total_seconds * sr)
    stereo = _render_voice(p, note, velocity, int(hold_seconds * sr), total, sr, seed)
    if p[40] > 0.0:
        stereo = _chorus(stereo, p[40], sr)
    if p[43] > 0.0:
        stereo = _delay(stereo, p[41], p[42], p[43], sr)
    left, right = _reverb_size(stereo, p[19], p[44], sr)
    out = np.empty((2, total))
    out[0] = left
    out[1] = right
    return out


@numba.njit(cache=True)
def _reverb_size(stereo, wet, size, sr):
    # v2 reverb with the room size as its own parameter (v2 tied it to the mix: 0.55 + 0.35 * mix)
    return _reverb_stereo_room(stereo[0], stereo[1], wet, size, sr)


@numba.njit(cache=True)
def _reverb_stereo_room(left, right, wet_level, room, sr):
    if wet_level <= 0.0:  # juce::Reverb dry gain is dryLevel * 2 (v2 always runs through it)
        return left * 2.0, right * 2.0
    return _freeverb(left, right, wet_level, room, sr)


@numba.njit(cache=True)
def _freeverb(left, right, wet_level, room, sr):
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
    comb_buf = np.zeros((2, 8, comb_sizes.max()))
    comb_idx = np.zeros((2, 8), dtype=np.int64)
    comb_last = np.zeros((2, 8))
    ap_buf = np.zeros((2, 4, ap_sizes.max()))
    ap_idx = np.zeros((2, 4), dtype=np.int64)
    damp = 0.4 * 0.4
    fb = room * 0.28 + 0.7
    dry = (1.0 - wet_level * 0.7) * 2.0
    wet1 = 0.5 * (wet_level * 3.0) * 2.0
    out_l = np.empty(n)
    out_r = np.empty(n)
    for i in range(n):
        inp = (left[i] + right[i]) * 0.015
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
        out_l[i] = acc_l * wet1 + left[i] * dry
        out_r[i] = acc_r * wet1 + right[i] * dry
    return out_l, out_r


def render(physical, note: int = 60, velocity: float = 100 / 127, hold_seconds: float = 1.5,
           total_seconds: float = 4.0, sr: int = SAMPLE_RATE, seed: int = 0x4321) -> np.ndarray:
    """(2, samples) float32 audio of one note for a physical v3 vector."""
    p = np.asarray(physical, dtype=np.float64)
    if p.shape != (P.N_PARAMS,):
        raise ValueError(f"expected {P.N_PARAMS} physical values, got {p.shape}")
    return render_physical(p, note, velocity, hold_seconds, total_seconds, sr, seed).astype(np.float32)


@numba.njit(cache=True, parallel=True)
def render_batch(params, notes, velocity, hold_seconds, total_seconds, sr, seed):
    """Mono float32 (n_presets, n_notes, samples) for many physical vectors."""
    n, m = params.shape[0], notes.shape[0]
    total = int(total_seconds * sr)
    out = np.zeros((n, m, total), dtype=np.float32)
    for k in numba.prange(n * m):
        i, j = k // m, k % m
        stereo = render_physical(params[i], notes[j], velocity, hold_seconds, total_seconds, sr, seed + k)
        for t in range(total):
            out[i, j, t] = 0.5 * (stereo[0, t] + stereo[1, t])
    return out
