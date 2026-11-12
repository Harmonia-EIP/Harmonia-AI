"""Relative instructions of a refine prompt: "plus sombre", "brighter", "more reverb", "attaque plus lente"...

CLAP hears what a sound is, not how it compares with another one: asked to refine towards "darker", the v3
search made the sound brighter 6 times out of 8 (CHANGELOG, 0.3.1). A comparative is not a description, it
is a direction on something that can be measured on the sound. This module reads the comparatives a prompt
holds (French and English), names what each one moves (brightness, attack, length, space), says which
settings move it on each engine, and measures the result so a move is only kept when the sound really
went the asked way.

The word lists only say what a comparative means; they are not training data.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

SR = 48000
MEASURES = ("brightness", "attack", "length", "space")

# (measure, direction): phrases, matched on lower-case text without accents. Direction +1 raises the measure:
# brighter, slower attack (longer attack time), longer, more reverb.
_PHRASES: Dict[Tuple[str, int], Tuple[str, ...]] = {
    ("brightness", +1): (
        r"plus brillante?s?", r"plus claire?s?", r"plus aigue?s?", r"plus d'aigus?", r"moins sombres?", r"moins sourde?s?",
        r"moins etouffee?s?", r"brighter", r"more bright(ness)?", r"less dark", r"more treble", r"more highs?",
        r"less muffled", r"open(er)? the filter", r"ouvr\w+ le filtre"),
    ("brightness", -1): (
        r"plus sombres?", r"plus sourde?s?", r"plus etouffee?s?", r"moins brillante?s?", r"moins claire?s?",
        r"moins aigue?s?", r"moins d'aigus?", r"darker", r"more dark", r"less bright(ness)?", r"less treble",
        r"less highs?", r"more muffled", r"clos\w+ the filter", r"ferm\w+ le filtre"),
    ("attack", +1): (
        r"attaque plus (lente|douce|longue|progressive)", r"attaque moins (rapide|dure|seche|franche)", r"moins d'attaque",
        r"moins percutante?s?", r"slower attack", r"softer attack", r"longer attack", r"less attack", r"less punchy?",
        r"fade ?in"),
    ("attack", -1): (
        r"attaque plus (rapide|dure|seche|franche|courte|nette)", r"attaque moins (lente|douce|longue)", r"plus d'attaque",
        r"plus percutante?s?", r"faster attack", r"quicker attack", r"harder attack", r"shorter attack", r"sharper attack",
        r"more attack", r"more punchy?", r"punchier", r"snappier"),
    ("length", +1): (
        r"plus long(ue)?s?", r"plus de release", r"plus de relachement", r"release plus (long(ue)?|lente?)", r"moins courte?s?",
        r"longer", r"more release", r"longer (release|tail|decay)", r"more sustain", r"plus de sustain"),
    ("length", -1): (
        r"plus courte?s?", r"moins long(ue)?s?", r"moins de release", r"release plus (courte?|rapide)", r"shorter",
        r"less release", r"shorter (release|tail|decay)"),
    ("space", +1): (
        r"plus de reverb\w*", r"plus d'espace", r"plus d'echo", r"plus de delay", r"plus spacieu(x|se)", r"plus ample",
        r"more reverb\w*", r"more space", r"more echo", r"more delay", r"more room", r"bigger room", r"wetter",
        r"dans une (plus )?grande (salle|piece)", r"plus de profondeur"),
    ("space", -1): (
        r"moins de reverb\w*", r"moins d'espace", r"moins d'echo", r"moins de delay", r"sans reverb\w*", r"plus sec(he)?s?",
        r"less reverb\w*", r"less space", r"less echo", r"less delay", r"no reverb\w*", r"drier", r"dry"),
}
_COMPILED = [(measure, direction, re.compile(r"(?<![a-z])(?:" + "|".join(phrases) + r")(?![a-z])"))
             for (measure, direction), phrases in _PHRASES.items()]
# Longest phrases first inside a group is not enough across groups ("attaque plus longue" vs "plus longue"):
# attack phrases are removed from the text before the others are searched.
_ORDER = ("attack", "space", "brightness", "length")


@dataclass(frozen=True)
class Directive:
    measure: str
    direction: int  # +1 raises the measure, -1 lowers it

    @property
    def label(self) -> str:
        return f"{self.measure}{'+' if self.direction > 0 else '-'}"


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip()


# "un peu plus sombre", "much brighter": how much of the usual step
_LESS = re.compile(r"(?<![a-z])(un peu|un petit peu|legerement|a peine|un poil|slightly|a bit|a little|a touch|subtly)(?![a-z])")
_MORE = re.compile(r"(?<![a-z])(beaucoup|bien plus|nettement|vraiment|tres|much|a lot|way|far|really|very)(?![a-z])")


def amount(prompt: str) -> float:
    """Factor on the size of the move asked by the prompt's wording: 0.5 for "un peu", 1.6 for "beaucoup"."""
    text = plain(prompt)
    if _LESS.search(text):
        return 0.5
    if _MORE.search(text):
        return 1.6
    return 1.0


def read(prompt: str) -> List[Directive]:
    """The comparatives of a prompt, one per measure (the first direction found wins)."""
    text = plain(prompt)
    found: Dict[str, int] = {}
    for measure in _ORDER:
        for name, direction, pattern in _COMPILED:
            if name != measure:
                continue
            if pattern.search(text):
                found.setdefault(measure, direction)
                text = pattern.sub(" ", text)
    return [Directive(m, found[m]) for m in MEASURES if m in found]


# --- what moves each measure ------------------------------------------------------------------------------

# Analog engine: (parameter, weight). A positive weight moves the parameter's normalized value the way the
# measure goes up; the first one is the main control, the others follow with less.
ANALOG_MOVES: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "brightness": (("filter_cutoff", 1.0), ("filter_env_amount", 0.25)),
    "attack": (("amp_attack", 1.0), ("filter_env_attack", 0.5)),
    "length": (("amp_release", 1.0), ("amp_decay", 0.5), ("filter_env_release", 0.5)),
    "space": (("reverb_mix", 1.0), ("reverb_size", 0.4)),
}
# DX7 mode: operator fields (all six operators) and the way their value goes when the measure goes up. The
# DX7 mode has no reverb: "space" has nothing to move there.
DX7_MOVES: Dict[str, Tuple[Tuple[str, int, float], ...]] = {
    "brightness": (("output_level", None, 1.0),),
    "attack": (("rates", 0, -1.0),),  # a lower first rate is a slower attack
    "length": (("rates", 3, -1.0),),  # a lower fourth rate is a longer release
    "space": (),
}
STEP = 0.14  # move of the main control, in normalized units, at strength 0.5 and scale 1
SCALES = (0.5, 1.0, 1.8, 3.0)
# Change of the measure (natural log) a refine aims for at strength 0.5: about x1.4 on the spectral centroid,
# x1.8 on the attack time, x1.5 on the length.
GOALS = {"brightness": 0.35, "attack": 0.6, "length": 0.4, "space": 0.4}
MIN_GAIN = 0.03  # below this the sound did not really move


def goal(measure: str, strength: float) -> float:
    return GOALS[measure] * (0.4 + 1.2 * strength)


# --- measures ---------------------------------------------------------------------------------------------

def measure(audio: np.ndarray, hold_seconds: float = 1.5) -> Dict[str, float]:
    """Natural logs of: spectral centroid of the held part (Hz), attack time (s, start to 90 % of the peak of
    the 10 ms RMS envelope), length (s, from the peak until the envelope stays 30 dB under it) and space (how
    much sound is left after the key is released, against the held part)."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    held = audio[:int(hold_seconds * SR)]
    spectrum = np.abs(np.fft.rfft(held * np.hanning(len(held))))
    freqs = np.fft.rfftfreq(len(held), 1.0 / SR)
    centroid = float((spectrum * freqs).sum() / max(float(spectrum.sum()), 1e-12))
    hop = SR // 100
    frames = len(audio) // hop
    env = np.sqrt(np.mean(audio[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    peak = float(env.max())
    if peak < 1e-7:
        return {"brightness": np.log(max(centroid, 1.0)), "attack": np.log(0.005), "length": np.log(0.01),
                "space": np.log(1e-4)}
    k = int(np.argmax(env))
    attack = (int(np.argmax(env >= 0.9 * peak)) + 0.5) * hop / SR
    above = np.nonzero(env[k:] > peak * 10 ** (-30 / 20))[0]
    length = (int(above[-1]) + 1) * hop / SR if len(above) else hop / SR
    release_at = int((hold_seconds + 0.25) * SR)
    tail = float(np.sqrt(np.mean(audio[release_at:] ** 2)))
    body = float(np.sqrt(np.mean(held ** 2))) + 1e-9
    return {"brightness": float(np.log(max(centroid, 1.0))), "attack": float(np.log(attack)),
            "length": float(np.log(length)), "space": float(np.log(tail / body + 1e-4))}


def measure_notes(notes_audio: Sequence[np.ndarray]) -> Dict[str, float]:
    """Mean of the measures over the notes of one preset."""
    rows = [measure(a) for a in notes_audio]
    return {m: float(np.mean([r[m] for r in rows])) for m in MEASURES}
