"""Which octave a recording really plays (#41).

Libraries name their files after the note ("Violin_A2_v1.wav") but not all with the same octave numbering
(C4 or C3 for middle C). Playing an instrument an octave off would ruin it, so each instrument is checked by
ear: for a few of its recordings, the note named in the file is compared with what the audio plays, octave
by octave, and the instrument gets the octave shift most of its recordings agree on.

A candidate pitch f scores how much louder the spectrum is at its harmonics (f, 2f, 3f...) than half-way
between them: the true pitch has energy on its harmonics and little in between; an octave too low puts its
"harmonics" between the true ones, an octave too high leaves true harmonics where it expects nothing.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

HARMONICS = 6
SHIFTS = (-24, -12, 0, 12, 24)


def frequency(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def _segment(audio: np.ndarray, sample_rate: int, seconds: float = 0.5) -> np.ndarray:
    """Mono, from just after the onset (the attack is noisy), seconds long."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    peak = float(np.abs(mono).max()) + 1e-12
    onset = int(np.argmax(np.abs(mono) > 0.1 * peak))
    start = onset + int(0.05 * sample_rate)
    return mono[start:start + int(seconds * sample_rate)]


def harmonic_score(spectrum: np.ndarray, bin_hz: float, f0: float) -> float:
    def level(f: float) -> float:
        i = int(round(f / bin_hz))
        if i >= len(spectrum) - 2:
            return np.nan
        return float(np.log(spectrum[max(0, i - 2):i + 3].max() + 1e-9))

    on = [level(h * f0) for h in range(1, HARMONICS + 1)]
    off = [level((h + 0.5) * f0) for h in range(1, HARMONICS + 1)]
    pairs = [(a, b) for a, b in zip(on, off) if not (np.isnan(a) or np.isnan(b))]
    if len(pairs) < 2:
        return -np.inf
    return float(np.mean([a - b for a, b in pairs]))


def octave_shift(audio: np.ndarray, sample_rate: int, named_note: int) -> Optional[int]:
    """Semitones to add to the named note so it matches the audio (a multiple of 12), or None when the
    recording has no clear pitch."""
    segment = _segment(np.asarray(audio, dtype=np.float64), sample_rate)
    if len(segment) < sample_rate // 10:
        return None
    window = np.hanning(len(segment))
    size = 1 << int(np.ceil(np.log2(len(segment) * 4)))
    spectrum = np.abs(np.fft.rfft(segment * window, size))
    bin_hz = sample_rate / size
    scores = {shift: harmonic_score(spectrum, bin_hz, frequency(named_note + shift)) for shift in SHIFTS
              if 20.0 <= frequency(named_note + shift) <= sample_rate / (2 * HARMONICS)}
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, score = ranked[0]
    if score < 1.0 or (len(ranked) > 1 and score - ranked[1][1] < 0.5):
        return None  # no clear pitch (noise, percussion) or no clear winner
    return best


def agreed_shift(shifts: Sequence[Optional[int]], min_clear: int = 4, share: float = 0.75) -> Optional[int]:
    """The shift at least `share` of the recordings with a clear pitch agree on (and at least `min_clear` of
    them have one), else None: an instrument whose octave is in doubt is better left out than played wrong."""
    clear = [s for s in shifts if s is not None]
    if len(clear) < min_clear:
        return None
    values, counts = np.unique(clear, return_counts=True)
    best = int(values[np.argmax(counts)])
    return best if counts.max() >= share * len(clear) else None
