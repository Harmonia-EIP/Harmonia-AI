"""Perceptual comparison of two renders of the same preset (original synth vs Harmonia), v2.

The first sound-matching metric (log-mel L1) could be lowered by adding noise, LFO wobble or distortion
that the original does not have, and it weighted treble far more than bass. This one compares, after
matching loudness:
- timbre: 1/3-octave band levels (every octave weighs the same, so bass counts), below a floor 60 dB
  under the original's peak ignored;
- envelope: broadband loudness every 5 ms (attack, impact, decay, length);
- movement: how much each band's level fluctuates over time (LFO, tremolo, wobble);
- noisiness: spectral flatness per band (tonal vs noisy);
- width: per band, how loud the side (L - R) is against the middle (L + R), when given stereo audio.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

SR = 48000
N_FFT = 8192
HOP = 1024
BANDS = 1000.0 * 2.0 ** (np.arange(-15, 15) / 3.0)  # 31 Hz .. 16 kHz, 1/3 octave
RANGE_DB = 60.0
ENV_HOP = 240  # 5 ms
# Checked against 60 listening ratings (Spearman with the score): timbre alone -0.49, + envelope -0.52,
# + movement -0.52 (kept: it is what keeps LFO wobble honest); the noisiness term did not help (-0.46).
WEIGHTS = {"timbre": 1.0, "envelope": 0.5, "movement": 1.0, "noise": 0.0, "width": 0.15}
WIDTH_FLOOR_DB = -40.0  # side/mid ratio of a mono sound


def _band_matrix(n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    lo, hi = BANDS * 2 ** (-1 / 6), BANDS * 2 ** (1 / 6)
    return np.stack([((freqs >= a) & (freqs < b)).astype(np.float64) for a, b in zip(lo, hi)])


_BANDS = _band_matrix()


def _power(mono: np.ndarray) -> np.ndarray:
    frames = 1 + max(0, len(mono) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(mono[idx] * np.hanning(N_FFT), axis=1)) ** 2


def describe(notes_audio: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-note perceptual features after loudness normalization over all notes together.

    Each note is mono (samples,) or stereo (2, samples); timbre, envelope and movement use the middle."""
    audio = [np.nan_to_num(np.asarray(a, dtype=np.float64)) for a in notes_audio]
    stereo = all(a.ndim == 2 for a in audio)
    mids = [0.5 * (a[0] + a[1]) if a.ndim == 2 else a for a in audio]
    rms = math.sqrt(np.mean([np.mean(m ** 2) for m in mids])) + 1e-9
    mids = [m * (0.1 / rms) for m in mids]
    bands, flat, env, width = [], [], [], []
    counts = np.maximum(_BANDS.sum(axis=1), 1.0)
    for n, a in enumerate(mids):
        power = _power(a)
        if stereo:
            side = _power(0.5 * (audio[n][0] - audio[n][1]) * (0.1 / rms)).mean(axis=0) @ _BANDS.T
            mid = power.mean(axis=0) @ _BANDS.T
            width.append(np.maximum(10 * np.log10((side + 1e-12) / (mid + 1e-12)), WIDTH_FLOOR_DB))
        band_power = power @ _BANDS.T / counts
        bands.append(10 * np.log10(band_power + 1e-12))
        # spectral flatness per band over the whole note: geometric / arithmetic mean of the bins
        mean_spec = power.mean(axis=0) + 1e-12
        geo = np.exp((np.log(mean_spec)[None, :] * _BANDS).sum(axis=1) / counts)
        arith = (mean_spec[None, :] * _BANDS).sum(axis=1) / counts
        flat.append(10 * np.log10(geo / arith + 1e-12))
        frames = len(a) // ENV_HOP
        env.append(10 * np.log10(np.mean(a[:frames * ENV_HOP].reshape(frames, ENV_HOP) ** 2, axis=1) + 1e-12))
    out = {"bands": np.stack(bands), "flatness": np.stack(flat), "envelope": np.stack(env), "bins": counts}
    if stereo:
        out["width"] = np.stack(width)
    return out


def _floored(x: np.ndarray, ref_max: float) -> np.ndarray:
    return np.maximum(x, ref_max - RANGE_DB)


def _movement(bands: np.ndarray) -> np.ndarray:
    """Per note and band: std of the level's fast variation (level minus its 0.2 s moving average)."""
    k = 9  # ~0.2 s at a 1024-sample hop
    kernel = np.ones(k) / k
    out = np.zeros(bands.shape[0::2])
    for n in range(bands.shape[0]):
        for b in range(bands.shape[2]):
            x = bands[n, :, b]
            smooth = np.convolve(np.pad(x, k // 2, mode="edge"), kernel, mode="valid")
            out[n, b] = np.std(x - smooth)
    return out


def components(original: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray]) -> Dict[str, float]:
    ref = original["bands"].max()
    bo, bc = _floored(original["bands"], ref), _floored(candidate["bands"], ref)
    frames = min(bo.shape[1], bc.shape[1])
    audible = (bo[:, :frames] > ref - RANGE_DB) | (bc[:, :frames] > ref - RANGE_DB)
    timbre = float(np.abs(bo[:, :frames] - bc[:, :frames])[audible].mean()) if audible.any() else 0.0
    eref = original["envelope"].max()
    eo, ec = _floored(original["envelope"], eref), _floored(candidate["envelope"], eref)
    n = min(eo.shape[1], ec.shape[1])
    envelope = float(np.abs(eo[:, :n] - ec[:, :n]).mean())
    loud = bo.mean(axis=1) > ref - 40  # bands that carry the sound
    movement = float(np.abs(_movement(bo) - _movement(bc))[loud].mean()) if loud.any() else 0.0
    wide = (original["bins"] >= 8)[None, :] & loud  # flatness is only meaningful with several bins
    noise = float(np.abs(original["flatness"] - candidate["flatness"])[wide].mean()) if wide.any() else 0.0
    width = 0.0
    if "width" in original and "width" in candidate and loud.any():
        width = float(np.abs(original["width"] - candidate["width"])[loud].mean())
    return {"timbre": timbre, "envelope": envelope, "movement": movement, "noise": noise, "width": width}


def distance(original: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray]) -> float:
    parts = components(original, candidate)
    return float(sum(WEIGHTS[k] * v for k, v in parts.items()))
