"""Random charter presets for the v2 preset bank.

Three sources, so the bank covers both musical presets and sound-design territory:
- "concept": jittered variations of the 100 hand-written concepts of synthesize_dataset.py;
- "archetype": envelope/timbre families (impacts, plucks, sustained, swells, noise textures);
- "uniform": uniformly random vectors, for coverage.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from src.charter import DISCRETE_INDICES, PARAM_INDEX, PARAM_NAMES

SOURCES = ("concept", "archetype", "uniform")
DEFAULT_MIX = {"concept": 0.30, "archetype": 0.45, "uniform": 0.25}

WAVE_STEPS = np.array([0.0, 1 / 3, 2 / 3, 1.0])
FILTER_STEPS = np.array([0.0, 0.5, 1.0])

# Envelope/timbre families: (low, high) uniform ranges in normalized charter space.
ARCHETYPES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "impact": {
        "amp_attack": (0.0, 0.08), "amp_decay": (0.0, 0.55), "amp_sustain": (0.0, 0.08),
        "amp_release": (0.15, 0.6), "noise_level": (0.3, 1.0), "filter_env_decay": (0.0, 0.5),
    },
    "pluck": {
        "amp_attack": (0.0, 0.1), "amp_decay": (0.3, 0.8), "amp_sustain": (0.0, 0.35),
        "amp_release": (0.3, 0.75), "noise_level": (0.0, 0.15), "filter_env_decay": (0.2, 0.7),
    },
    "sustained": {
        "amp_attack": (0.0, 0.35), "amp_decay": (0.2, 0.8), "amp_sustain": (0.5, 1.0),
        "amp_release": (0.1, 0.65), "noise_level": (0.0, 0.2), "filter_env_decay": (0.1, 0.9),
    },
    "swell": {
        "amp_attack": (0.45, 1.0), "amp_decay": (0.3, 1.0), "amp_sustain": (0.5, 1.0),
        "amp_release": (0.5, 1.0), "noise_level": (0.0, 0.3), "reverb_mix": (0.3, 1.0),
    },
    "texture": {
        "amp_attack": (0.0, 0.9), "amp_decay": (0.2, 1.0), "amp_sustain": (0.2, 1.0),
        "amp_release": (0.3, 1.0), "noise_level": (0.5, 1.0), "lfo_to_cutoff": (0.2, 1.0),
    },
}


def _sparse(rng: np.random.Generator, p_zero: float, a: float, b: float) -> float:
    return 0.0 if rng.random() < p_zero else float(rng.beta(a, b))


def sample_archetype(rng: np.random.Generator) -> np.ndarray:
    family = ARCHETYPES[list(ARCHETYPES)[int(rng.integers(len(ARCHETYPES)))]]
    v = np.empty(len(PARAM_NAMES))
    v[PARAM_INDEX["osc_1_waveform"]] = rng.choice(WAVE_STEPS)
    v[PARAM_INDEX["osc_2_waveform"]] = rng.choice(WAVE_STEPS)
    v[PARAM_INDEX["osc_mix"]] = rng.uniform()
    v[PARAM_INDEX["osc_2_detune"]] = _sparse(rng, 0.3, 1.2, 4.0)
    v[PARAM_INDEX["noise_level"]] = _sparse(rng, 0.5, 1.0, 3.0)
    v[PARAM_INDEX["filter_cutoff"]] = rng.uniform(0.15, 1.0)
    v[PARAM_INDEX["filter_resonance"]] = _sparse(rng, 0.3, 1.2, 2.5)
    v[PARAM_INDEX["filter_type"]] = rng.choice(FILTER_STEPS, p=[0.6, 0.2, 0.2])
    v[PARAM_INDEX["filter_env_amount"]] = 0.5 if rng.random() < 0.3 else rng.uniform()
    v[PARAM_INDEX["filter_env_decay"]] = rng.uniform()
    v[PARAM_INDEX["lfo_rate"]] = rng.uniform()
    v[PARAM_INDEX["lfo_to_pitch"]] = _sparse(rng, 0.75, 1.0, 6.0)
    v[PARAM_INDEX["lfo_to_cutoff"]] = _sparse(rng, 0.6, 1.0, 3.0)
    v[PARAM_INDEX["velocity_to_filter"]] = rng.uniform()
    v[PARAM_INDEX["distortion_mix"]] = _sparse(rng, 0.5, 1.0, 2.5)
    v[PARAM_INDEX["reverb_mix"]] = rng.uniform()
    for name in ("amp_attack", "amp_decay", "amp_sustain", "amp_release"):
        v[PARAM_INDEX[name]] = rng.uniform()
    for name, (lo, hi) in family.items():
        v[PARAM_INDEX[name]] = rng.uniform(lo, hi)
    return v


def sample_uniform(rng: np.random.Generator) -> np.ndarray:
    v = rng.uniform(size=len(PARAM_NAMES))
    for idx in DISCRETE_INDICES:
        steps = FILTER_STEPS if PARAM_NAMES[idx] == "filter_type" else WAVE_STEPS
        v[idx] = rng.choice(steps)
    return v


def _concept_sampler(seed: int):
    from scripts.synthesize_dataset import CONCEPTS, generate_variation

    py_rng = random.Random(seed)  # nosec B311 - seeded RNG for reproducible data, not security-sensitive

    def sample() -> Tuple[np.ndarray, str]:
        concept = CONCEPTS[py_rng.randrange(len(CONCEPTS))]
        prompt = py_rng.choice(concept["prompts"])
        record = generate_variation(concept, prompt, py_rng)
        values = {**record["parameters"]["continuous"], **record["parameters"]["categorical"]}
        return np.array([values[name] for name in PARAM_NAMES]), concept["label"]

    return sample


def sample_bank(n: int, seed: int = 42, mix: Dict[str, float] = DEFAULT_MIX) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Returns (params float32 (n, 20), source index int8 (n,), concept label or "" per row)."""
    rng = np.random.default_rng(seed)
    weights = np.array([mix[s] for s in SOURCES], dtype=np.float64)
    sources = rng.choice(len(SOURCES), size=n, p=weights / weights.sum())
    concept = _concept_sampler(seed)
    params = np.empty((n, len(PARAM_NAMES)), dtype=np.float32)
    labels: List[str] = []
    for i, src in enumerate(sources):
        label = ""
        if SOURCES[src] == "concept":
            vec, label = concept()
        elif SOURCES[src] == "archetype":
            vec = sample_archetype(rng)
        else:
            vec = sample_uniform(rng)
        params[i] = np.clip(vec, 0.0, 1.0)
        labels.append(label)
    return params, sources.astype(np.int8), labels
