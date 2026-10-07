"""Sound matching: how far a Harmonia v3 preset sounds from an original recording, and an optimizer.

Distance = mean absolute difference of log-mel spectrograms (dB, two time resolutions) over a few notes,
after matching overall loudness (synths differ in output level, not what we compare). The optimizer is a
compact CMA-ES over the continuous parameters (normalized 0..1); discrete ones (waveforms, filter type,
slope, sync, LFO shape, unison count, coarse tuning) stay as the rule-based conversion set them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.synth import v3_params as P
from src.synth.engine_v3 import render_physical

SR = 48000
NOTES = (48, 60, 72)
HOLD_SECONDS = 1.5
TOTAL_SECONDS = 3.5
VELOCITY = 100
RESOLUTIONS = ((2048, 512, 96), (512, 128, 48))  # (fft size, hop, mel bands)
DB_FLOOR = -80.0
TARGET_RMS = 0.1

FIXED = {"osc_1_waveform", "osc_2_waveform", "filter_type", "osc_sync", "filter_slope", "lfo_waveform",
         "unison_voices", "osc_1_coarse", "osc_2_coarse"}
FREE_INDEX = np.array([i for i, name in enumerate(P.NAMES) if name not in FIXED])


def _mel_filters(n_fft: int, n_mels: int, sr: int = SR, fmin: float = 30.0, fmax: float = 16000.0) -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    points = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bank = np.zeros((n_mels, freqs.size))
    for m in range(n_mels):
        lo, mid, hi = points[m], points[m + 1], points[m + 2]
        bank[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0.0, None)
    return bank


_FILTERS = {(n_fft, n_mels): _mel_filters(n_fft, n_mels) for n_fft, _, n_mels in RESOLUTIONS}


def log_mel(mono: np.ndarray, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    frames = 1 + (len(mono) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(mono[idx] * np.hanning(n_fft), axis=1)) ** 2
    return 10.0 * np.log10(spec @ _FILTERS[(n_fft, n_mels)].T + 1e-12)


def features(notes_audio: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Per resolution: (notes, frames, mels) dB, loudness-normalized over all notes together."""
    stacked = [np.nan_to_num(np.asarray(a, dtype=np.float64)) for a in notes_audio]
    rms = math.sqrt(np.mean([np.mean(a ** 2) for a in stacked])) + 1e-9
    stacked = [a * (TARGET_RMS / rms) for a in stacked]
    return [np.maximum(np.stack([log_mel(a, n_fft, hop, n_mels) for a in stacked]), DB_FLOOR)
            for n_fft, hop, n_mels in RESOLUTIONS]


def distance(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    return float(np.mean([np.mean(np.abs(x - y)) for x, y in zip(a, b)]))


def render_notes(physical: np.ndarray, notes: Sequence[int] = NOTES, seed: int = 7) -> List[np.ndarray]:
    p = np.asarray(physical, dtype=np.float64)
    return [render_physical(p, note, VELOCITY / 127, HOLD_SECONDS, TOTAL_SECONDS, SR, seed).mean(axis=0)
            for note in notes]


def preset_distance(physical: np.ndarray, target: List[np.ndarray]) -> float:
    return distance(features(render_notes(physical)), target)


@dataclass
class MatchResult:
    physical: np.ndarray
    start_distance: float
    distance: float
    evaluations: int


def cma_es(f: Callable[[np.ndarray], float], x0: np.ndarray, sigma: float, budget: int,
           rng: np.random.Generator, popsize: Optional[int] = None) -> Tuple[np.ndarray, float, int]:
    """Minimize f over [0, 1]^n (clipped) with a standard (mu/mu_w, lambda)-CMA-ES."""
    n = x0.size
    lam = popsize or 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    mean = x0.copy()
    pc = np.zeros(n)
    ps = np.zeros(n)
    b = np.eye(n)
    d = np.ones(n)
    c = np.eye(n)
    best_x, best_f = x0.copy(), f(x0)
    evals = 1
    gen = 0
    while evals + lam <= budget:
        z = rng.standard_normal((lam, n))
        y = z @ (b * d).T
        xs = np.clip(mean + sigma * y, 0.0, 1.0)
        fs = np.array([f(x) for x in xs])
        evals += lam
        order = np.argsort(fs)
        if fs[order[0]] < best_f:
            best_f, best_x = float(fs[order[0]]), xs[order[0]].copy()
        y_sel = (xs[order[:mu]] - mean) / sigma
        y_w = w @ y_sel
        mean = mean + sigma * y_w
        c_inv_sqrt = b @ np.diag(1 / d) @ b.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (c_inv_sqrt @ y_w)
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        c = (1 - c1 - cmu) * c + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * c) \
            + cmu * (y_sel.T * w) @ y_sel
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        c = np.triu(c) + np.triu(c, 1).T
        eigvals, b = np.linalg.eigh(c)
        d = np.sqrt(np.maximum(eigvals, 1e-20))
        gen += 1
    return best_x, best_f, evals


DISCRETE_CHOICES = {
    "osc_1_waveform": (0, 1, 2, 3), "osc_2_waveform": (0, 1, 2, 3), "filter_type": (0, 1, 2),
    "filter_slope": (0, 1), "osc_sync": (0, 1), "unison_voices": (1, 3, 5, 7),
}


def discrete_search(start: np.ndarray, target: List[np.ndarray], names=tuple(DISCRETE_CHOICES)) -> Tuple[np.ndarray, float, int]:
    """One pass of coordinate descent over the discrete parameters (each tried at every value)."""
    best = np.asarray(start, dtype=np.float64).copy()
    best_d = preset_distance(best, target)
    evals = 1
    for name in names:
        i = P.INDEX[name]
        for value in DISCRETE_CHOICES[name]:
            if value == best[i]:
                continue
            trial = best.copy()
            trial[i] = value
            d = preset_distance(trial, target)
            evals += 1
            if d < best_d:
                best, best_d = trial, d
    return best, best_d, evals


def match(start: np.ndarray, target: List[np.ndarray], budget: int = 480, sigma: float = 0.15,
          seed: int = 0, search_discrete: bool = False) -> MatchResult:
    """Refine `start` (physical) so it sounds like `target` features: optionally the discrete parameters
    first (one coordinate-descent pass), then the continuous ones with CMA-ES."""
    start = np.asarray(start, dtype=np.float64)
    start_distance = preset_distance(start, target)
    extra = 0
    if search_discrete:
        start, _, extra = discrete_search(start, target)
    normalized = P.normalize(start)

    def to_physical(x: np.ndarray) -> np.ndarray:
        full = normalized.copy()
        full[FREE_INDEX] = x
        return P.denormalize(full)

    def objective(x: np.ndarray) -> float:
        return preset_distance(to_physical(x), target)

    x, best, evals = cma_es(objective, normalized[FREE_INDEX], sigma, budget - extra, np.random.default_rng(seed))
    current = preset_distance(start, target)
    if best >= current:
        return MatchResult(start, start_distance, current, evals + extra)
    return MatchResult(to_physical(x), start_distance, best, evals + extra)


def describe(physical: np.ndarray) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(P.NAMES, physical) if abs(v - P.DEFAULTS[P.INDEX[name]]) > 1e-9}
