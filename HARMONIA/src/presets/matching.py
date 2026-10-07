"""Sound matching: how far a Harmonia v3 preset sounds from an original recording, and an optimizer.

Distance: src/presets/perceptual.py (1/3-octave timbre, loudness envelope, movement), over a few notes
after matching loudness. The optimizer is a compact CMA-ES over the continuous parameters (normalized
0..1), kept near the rule-based conversion: it may not add what the original does not have (noise, LFO,
FM, ring mod, distortion, effects, pitch envelope), may only nudge those it has, and pays for every move
away from the conversion. Discrete parameters stay as the conversion set them (Surge: waveforms, filter
type and slope are searched first). The first version, without these limits, lowered its log-mel
distance by adding noise, LFO and distortion that listeners heard as wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np

from src.presets import perceptual
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
         "unison_voices", "osc_1_coarse", "osc_2_coarse",
         # one fixed velocity is rendered: these only shift level/brightness, which other parameters own
         "velocity_to_amp", "velocity_to_filter"}
FREE_INDEX = np.array([i for i, name in enumerate(P.NAMES) if name not in FIXED])
# Present in the original only if the conversion set them: never added, only nudged (+-NUDGE).
ADD_ONLY_IF_PRESENT = ("noise_level", "ring_mod", "fm_amount", "distortion_mix", "chorus_mix", "delay_mix",
                       "reverb_mix", "lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp", "lfo_to_pw", "pitch_env_amount")
# Only meaningful when their module is on.
DEPENDS_ON = {"lfo_rate": ("lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp", "lfo_to_pw"),
              "lfo_delay": ("lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp", "lfo_to_pw"),
              "delay_time": ("delay_mix",), "delay_feedback": ("delay_mix",), "reverb_size": ("reverb_mix",),
              "pitch_env_decay": ("pitch_env_amount",), "unison_detune": ("unison_voices",)}
NUDGE = 0.15
STAY_NEAR = 10.0  # cost of the mean squared move (normalized units) away from the conversion


def features(notes_audio: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return perceptual.describe(notes_audio)


def distance(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    """Perceptual distance of candidate `b` from original `a` (not symmetric)."""
    return perceptual.distance(a, b)


def render_notes(physical: np.ndarray, notes: Sequence[int] = NOTES, seed: int = 7) -> List[np.ndarray]:
    """Stereo (2, samples) renders, one per note."""
    p = np.asarray(physical, dtype=np.float64)
    return [render_physical(p, note, VELOCITY / 127, HOLD_SECONDS, TOTAL_SECONDS, SR, seed) for note in notes]


def preset_distance(physical: np.ndarray, target: Dict[str, np.ndarray]) -> float:
    return distance(target, features(render_notes(physical)))


def search_space(start: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices the optimizer may move and their normalized bounds around the conversion."""
    normalized = P.normalize(start)
    free, lo, hi = [], [], []
    for i in FREE_INDEX:
        name = P.NAMES[i]
        if name in ADD_ONLY_IF_PRESENT and start[i] == 0.0:
            continue
        deps = DEPENDS_ON.get(name)
        if deps == ("unison_voices",):
            if start[P.INDEX["unison_voices"]] <= 1:
                continue
        elif deps and all(start[P.INDEX[d]] == 0.0 for d in deps):
            continue
        free.append(i)
        if name in ADD_ONLY_IF_PRESENT:
            lo.append(max(0.0, normalized[i] - NUDGE))
            hi.append(min(1.0, normalized[i] + NUDGE))
        else:
            lo.append(0.0)
            hi.append(1.0)
    return np.array(free, dtype=int), np.array(lo), np.array(hi)


@dataclass
class MatchResult:
    physical: np.ndarray
    start_distance: float
    distance: float
    evaluations: int


def cma_es(f: Callable[[np.ndarray], float], x0: np.ndarray, sigma: float, budget: int,
           rng: np.random.Generator, popsize: Optional[int] = None, lower: Optional[np.ndarray] = None,
           upper: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, int]:
    """Minimize f over a box (default [0, 1]^n, samples clipped) with a standard (mu/mu_w, lambda)-CMA-ES."""
    lower = np.zeros_like(x0) if lower is None else lower
    upper = np.ones_like(x0) if upper is None else upper
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
        xs = np.clip(mean + sigma * y, lower, upper)
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
    "filter_slope": (0, 1),
}


def discrete_search(start: np.ndarray, target: Dict[str, np.ndarray],
                    names=tuple(DISCRETE_CHOICES)) -> Tuple[np.ndarray, float, int]:
    """One pass of coordinate descent over the discrete parameters (each tried at every value), after
    trying the whole preset an octave up and down (catches octave settings the conversion missed)."""
    best = np.asarray(start, dtype=np.float64).copy()
    best_d = preset_distance(best, target)
    evals = 1
    for shift in (-12, 12):
        trial = best.copy()
        for name in ("osc_1_coarse", "osc_2_coarse"):
            trial[P.INDEX[name]] = P.snap(P.BY_NAME[name], trial[P.INDEX[name]] + shift)
        d = preset_distance(trial, target)
        evals += 1
        if d < best_d - 0.5:  # clearly better only: an octave is a big change
            best, best_d = trial, d
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


def match(start: np.ndarray, target: Dict[str, np.ndarray], budget: int = 480, sigma: float = 0.15,
          seed: int = 0, search_discrete: bool = False) -> MatchResult:
    """Refine `start` (physical) so it sounds like the `target` features: optionally the discrete parameters
    first (one coordinate-descent pass), then the allowed continuous ones with CMA-ES, staying near `start`."""
    start = np.asarray(start, dtype=np.float64)
    start_distance = preset_distance(start, target)
    extra = 0
    if search_discrete:
        start, _, extra = discrete_search(start, target)
    current = preset_distance(start, target)
    free, lower, upper = search_space(start)
    if free.size == 0:
        return MatchResult(start, start_distance, current, extra + 1)
    normalized = P.normalize(start)
    x0 = normalized[free]

    def to_physical(x: np.ndarray) -> np.ndarray:
        full = normalized.copy()
        full[free] = x
        return P.denormalize(full)

    def objective(x: np.ndarray) -> float:
        return preset_distance(to_physical(x), target) + STAY_NEAR * float(np.mean((x - x0) ** 2))

    x, best, evals = cma_es(objective, x0, sigma, budget - extra, np.random.default_rng(seed),
                            lower=lower, upper=upper)
    if best >= current:
        return MatchResult(start, start_distance, current, evals + extra)
    physical = to_physical(x)
    return MatchResult(physical, start_distance, preset_distance(physical, target), evals + extra)


def describe(physical: np.ndarray) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(P.NAMES, physical) if abs(v - P.DEFAULTS[P.INDEX[name]]) > 1e-9}
