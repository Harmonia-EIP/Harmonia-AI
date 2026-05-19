#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import BIPOLAR_INDICES, DISCRETE_INDICES, DISCRETE_STEPS, PARAM_NAMES

# Waveform discrete encoding (4 steps): Sine, Triangle, Saw, Square
SINE, TRI, SAW, SQR = 0.0, 0.333, 0.667, 1.0
# Filter type (3 steps): Lowpass, Bandpass, Highpass
LP, BP, HP = 0.0, 0.5, 1.0


# Modifier words found in prompts that nudge params in a predictable direction.
# Kept conservative on purpose: too-strong deltas drown the concept itself.
MODIFIERS = {
    "bright": {"filter_cutoff": +0.15, "lfo_to_cutoff": +0.03},
    "dark": {"filter_cutoff": -0.18, "reverb_mix": +0.05},
    "soft": {"distortion_mix": -0.18, "amp_attack": +0.04, "filter_resonance": -0.05},
    "gentle": {"distortion_mix": -0.10, "amp_attack": +0.05, "reverb_mix": +0.05},
    "hard": {"distortion_mix": +0.22, "filter_resonance": +0.10},
    "aggressive": {"distortion_mix": +0.30, "filter_resonance": +0.12, "filter_cutoff": +0.05},
    "warm": {"filter_cutoff": -0.07, "reverb_mix": +0.08, "osc_2_detune": +0.04},
    "cold": {"filter_cutoff": +0.05, "reverb_mix": -0.05, "osc_2_detune": -0.03},
    "ambient": {"reverb_mix": +0.30, "amp_release": +0.20, "amp_attack": +0.10},
    "dreamy": {"reverb_mix": +0.25, "amp_release": +0.20, "lfo_to_cutoff": +0.05},
    "ethereal": {"reverb_mix": +0.30, "amp_release": +0.20, "lfo_to_pitch": +0.04},
    "lo-fi": {"filter_cutoff": -0.20, "reverb_mix": +0.05, "noise_level": +0.04},
    "lofi": {"filter_cutoff": -0.20, "reverb_mix": +0.05, "noise_level": +0.04},
    "punchy": {"amp_attack": -0.03, "amp_decay": -0.08, "amp_release": -0.10},
    "vintage": {"filter_cutoff": -0.05, "reverb_mix": +0.05, "osc_2_detune": +0.05},
    "retro": {"filter_cutoff": -0.07, "reverb_mix": +0.05, "osc_2_detune": +0.05},
    "modern": {"reverb_mix": -0.05, "distortion_mix": -0.05},
    "deep": {"filter_cutoff": -0.10, "amp_release": +0.10},
    "wide": {"osc_2_detune": +0.10, "reverb_mix": +0.05},
    "thin": {"osc_2_detune": -0.05, "filter_resonance": +0.05},
    "fat": {"osc_mix": +0.10, "osc_2_detune": +0.05, "filter_resonance": +0.03},
    "detuned": {"osc_2_detune": +0.15},
    "cinematic": {"reverb_mix": +0.20, "amp_release": +0.15, "amp_attack": +0.05},
    "intimate": {"reverb_mix": -0.10, "distortion_mix": -0.10},
    "clean": {"distortion_mix": -0.18},
    "dirty": {"distortion_mix": +0.20, "filter_resonance": +0.05},
    "fast": {"amp_attack": -0.03, "amp_release": -0.06, "lfo_rate": +0.15},
    "slow": {"amp_attack": +0.10, "amp_release": +0.15, "lfo_rate": -0.12},
    "screaming": {"distortion_mix": +0.30, "filter_cutoff": +0.10, "filter_resonance": +0.15},
    "rich": {"osc_2_detune": +0.08, "reverb_mix": +0.05, "osc_mix": +0.05},
    "lush": {"reverb_mix": +0.15, "osc_2_detune": +0.10, "amp_release": +0.10},
    "subby": {"filter_cutoff": -0.20, "osc_1_waveform": SINE - SINE},  # 0 delta on discrete handled separately
    "sub": {"filter_cutoff": -0.20},
    "stab": {"amp_decay": -0.15, "amp_sustain": -0.30, "amp_release": -0.20},
    "epic": {"reverb_mix": +0.15, "amp_release": +0.15, "amp_sustain": +0.10},
    "glassy": {"filter_cutoff": +0.10, "filter_resonance": +0.10, "osc_1_waveform_force": SINE},
    "metallic": {"filter_resonance": +0.20, "distortion_mix": +0.05},
}

# Helper: build a single concept entry with default jitter.
def concept(label, prompts, base, jitter=None):
    j = {"default": 0.04, "filter_cutoff": 0.06, "reverb_mix": 0.06, "amp_attack": 0.03,
         "amp_release": 0.06, "distortion_mix": 0.05, "filter_resonance": 0.05,
         "osc_2_detune": 0.05, "lfo_rate": 0.06, "amp_decay": 0.05, "amp_sustain": 0.06,
         "discrete_swap_prob": 0.05}
    if jitter:
        j.update(jitter)
    return {"label": label, "prompts": prompts, "base": base, "jitter": j}


CONCEPTS = [
    # =========================================================
    # KEYS / PIANOS / ORGANS  (12)
    # =========================================================
    concept(
        "Grand Piano",
        ["Grand Piano", "classical piano", "acoustic grand", "concert piano", "rich piano"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.04,
         "noise_level": 0.0, "filter_cutoff": 0.72, "filter_resonance": 0.03, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.65, "amp_sustain": 0.0, "amp_release": 0.50,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.85, "distortion_mix": 0.0, "reverb_mix": 0.30},
    ),
    concept(
        "Soft Piano",
        ["Soft Piano", "gentle piano", "warm piano", "felt piano", "intimate piano", "delicate piano"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.30, "osc_2_detune": 0.05,
         "noise_level": 0.02, "filter_cutoff": 0.55, "filter_resonance": 0.04, "filter_type": LP,
         "amp_attack": 0.01, "amp_decay": 0.60, "amp_sustain": 0.05, "amp_release": 0.55,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.85, "distortion_mix": 0.0, "reverb_mix": 0.40},
    ),
    concept(
        "Bright Piano",
        ["Bright Piano", "sparkling piano", "crisp piano", "pop piano", "modern piano"],
        {"osc_1_waveform": TRI, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.03,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.06, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.0, "amp_release": 0.40,
         "filter_env_amount": 0.60, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.80, "distortion_mix": 0.02, "reverb_mix": 0.25},
    ),
    concept(
        "Lo-Fi Piano",
        ["Lo-Fi Piano", "lofi piano", "tape piano", "dusty piano", "vinyl piano"],
        {"osc_1_waveform": TRI, "osc_2_waveform": SINE, "osc_mix": 0.35, "osc_2_detune": 0.10,
         "noise_level": 0.10, "filter_cutoff": 0.40, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.60, "amp_sustain": 0.10, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.02, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.70, "distortion_mix": 0.10, "reverb_mix": 0.45},
    ),
    concept(
        "Cinematic Piano",
        ["Cinematic Piano", "epic piano", "film score piano", "emotional piano"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.06,
         "noise_level": 0.01, "filter_cutoff": 0.65, "filter_resonance": 0.04, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.65, "amp_sustain": 0.05, "amp_release": 0.65,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.85, "distortion_mix": 0.0, "reverb_mix": 0.65},
    ),
    concept(
        "Rhodes EPiano",
        ["Rhodes EPiano", "Rhodes electric piano", "rhodes", "vintage rhodes", "jazz rhodes"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.35, "osc_2_detune": 0.04,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.20, "amp_release": 0.55,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.65, "distortion_mix": 0.05, "reverb_mix": 0.30},
    ),
    concept(
        "Wurlitzer EPiano",
        ["Wurlitzer", "Wurli", "wurlitzer epiano", "60s wurli"],
        {"osc_1_waveform": TRI, "osc_2_waveform": SINE, "osc_mix": 0.45, "osc_2_detune": 0.05,
         "noise_level": 0.02, "filter_cutoff": 0.55, "filter_resonance": 0.18, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.15, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.60, "distortion_mix": 0.15, "reverb_mix": 0.30},
    ),
    concept(
        "FM EPiano",
        ["FM EPiano", "DX7 piano", "FM electric piano", "80s epiano"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.50, "osc_2_detune": 0.02,
         "noise_level": 0.0, "filter_cutoff": 0.65, "filter_resonance": 0.08, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.15, "amp_release": 0.50,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.02, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.65, "distortion_mix": 0.05, "reverb_mix": 0.35},
    ),
    concept(
        "Hammond Organ",
        ["Hammond Organ", "B3 organ", "rock organ", "blues organ"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.55, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.10, "amp_sustain": 0.95, "amp_release": 0.10,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.50, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.20, "distortion_mix": 0.25, "reverb_mix": 0.35},
    ),
    concept(
        "Pipe Organ",
        ["Pipe Organ", "church organ", "cathedral organ", "sacred organ"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.65, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.05, "amp_decay": 0.10, "amp_sustain": 0.95, "amp_release": 0.25,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.20, "distortion_mix": 0.0, "reverb_mix": 0.85},
    ),
    concept(
        "Harpsichord",
        ["Harpsichord", "baroque harpsichord", "plucked keys"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.02,
         "noise_level": 0.05, "filter_cutoff": 0.85, "filter_resonance": 0.10, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.35, "amp_sustain": 0.0, "amp_release": 0.25,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.0, "reverb_mix": 0.35},
    ),
    concept(
        "Clavinet",
        ["Clavinet", "clav", "funky clavinet", "70s clav"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.45, "osc_2_detune": 0.03,
         "noise_level": 0.03, "filter_cutoff": 0.70, "filter_resonance": 0.30, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.10, "amp_release": 0.20,
         "filter_env_amount": 0.70, "filter_env_decay": 0.30,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.55, "distortion_mix": 0.15, "reverb_mix": 0.20},
    ),

    # =========================================================
    # BASS  (16)
    # =========================================================
    concept(
        "Sub Bass",
        ["Sub Bass", "subby bass", "808 sub", "deep sub", "low end sub"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.05, "osc_2_detune": 0.0,
         "noise_level": 0.0, "filter_cutoff": 0.20, "filter_resonance": 0.0, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.40, "amp_sustain": 0.90, "amp_release": 0.30,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.05},
    ),
    concept(
        "808 Bass",
        ["808 Bass", "808 trap bass", "trap 808", "rap 808 bass"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.10, "osc_2_detune": 0.0,
         "noise_level": 0.0, "filter_cutoff": 0.25, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.50, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.20, "distortion_mix": 0.25, "reverb_mix": 0.10},
    ),
    concept(
        "Reese Bass",
        ["Reese Bass", "DnB reese", "drum and bass reese", "neuro bass", "rolling reese"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.60, "osc_2_detune": 0.30,
         "noise_level": 0.0, "filter_cutoff": 0.40, "filter_resonance": 0.30, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.85, "amp_release": 0.30,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.25,
         "velocity_to_filter": 0.25, "distortion_mix": 0.45, "reverb_mix": 0.15},
    ),
    concept(
        "Wobble Bass",
        ["Wobble Bass", "dubstep wobble", "wobble bass dubstep", "growl wobble", "yoy wobble"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.55, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.50, "filter_resonance": 0.55, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.85, "amp_release": 0.25,
         "filter_env_amount": 0.65, "filter_env_decay": 0.30,
         "lfo_rate": 0.55, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.80,
         "velocity_to_filter": 0.10, "distortion_mix": 0.55, "reverb_mix": 0.10},
    ),
    concept(
        "Acid Bass",
        ["Acid Bass", "303 acid", "TB-303 bass", "squelchy acid", "psy acid"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.10, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.35, "filter_resonance": 0.65, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.20, "amp_release": 0.15,
         "filter_env_amount": 0.85, "filter_env_decay": 0.25,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.45, "distortion_mix": 0.45, "reverb_mix": 0.10},
    ),
    concept(
        "FM Bass",
        ["FM Bass", "modern FM bass", "FM growl", "FM synth bass"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.40, "osc_2_detune": 0.02,
         "noise_level": 0.0, "filter_cutoff": 0.35, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.35, "amp_sustain": 0.65, "amp_release": 0.25,
         "filter_env_amount": 0.60, "filter_env_decay": 0.35,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.30, "distortion_mix": 0.35, "reverb_mix": 0.10},
    ),
    concept(
        "Pluck Bass",
        ["Pluck Bass", "plucky bass", "funk pluck bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.35, "osc_2_detune": 0.03,
         "noise_level": 0.0, "filter_cutoff": 0.45, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.05, "amp_release": 0.20,
         "filter_env_amount": 0.65, "filter_env_decay": 0.30,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.10, "reverb_mix": 0.15},
    ),
    concept(
        "Hard Bass",
        ["Hard Bass", "hardstyle bass", "raw hard bass", "distorted bass"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SAW, "osc_mix": 0.60, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.60, "filter_resonance": 0.40, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.85, "amp_release": 0.20,
         "filter_env_amount": 0.65, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.20, "distortion_mix": 0.75, "reverb_mix": 0.10},
    ),
    concept(
        "House Bass",
        ["House Bass", "deep house bass", "tech house bass", "rolling house bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.30, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.45, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.55, "amp_release": 0.20,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.30, "distortion_mix": 0.20, "reverb_mix": 0.15},
    ),
    concept(
        "Detuned Bass",
        ["Detuned Bass", "wide bass", "chorus bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.35,
         "noise_level": 0.0, "filter_cutoff": 0.40, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.75, "amp_release": 0.25,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.25, "distortion_mix": 0.15, "reverb_mix": 0.15},
    ),
    concept(
        "Punchy Bass",
        ["Punchy Bass", "tight bass", "snappy bass", "techno bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.45, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.50, "filter_resonance": 0.25, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.30, "amp_release": 0.15,
         "filter_env_amount": 0.70, "filter_env_decay": 0.25,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.30, "reverb_mix": 0.10},
    ),
    concept(
        "Synthwave Bass",
        ["Synthwave Bass", "80s synth bass", "retrowave bass", "outrun bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.12,
         "noise_level": 0.0, "filter_cutoff": 0.50, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.80, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.25, "distortion_mix": 0.15, "reverb_mix": 0.20},
    ),
    concept(
        "Square Bass",
        ["Square Bass", "chiptune bass", "8-bit bass", "retro game bass"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.80, "amp_release": 0.15,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.10, "reverb_mix": 0.10},
    ),
    concept(
        "Growl Bass",
        ["Growl Bass", "growling bass", "monster bass", "dubstep growl"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.60, "osc_2_detune": 0.15,
         "noise_level": 0.05, "filter_cutoff": 0.45, "filter_resonance": 0.60, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.85, "amp_release": 0.20,
         "filter_env_amount": 0.70, "filter_env_decay": 0.30,
         "lfo_rate": 0.45, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.55,
         "velocity_to_filter": 0.20, "distortion_mix": 0.65, "reverb_mix": 0.10},
    ),
    concept(
        "Soft Bass",
        ["Soft Bass", "smooth bass", "warm bass"],
        {"osc_1_waveform": TRI, "osc_2_waveform": SINE, "osc_mix": 0.30, "osc_2_detune": 0.02,
         "noise_level": 0.0, "filter_cutoff": 0.30, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.35, "amp_sustain": 0.80, "amp_release": 0.35,
         "filter_env_amount": 0.50, "filter_env_decay": 0.35,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.05, "reverb_mix": 0.20},
    ),
    concept(
        "Slap Bass",
        ["Slap Bass", "funk bass", "popping bass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.40, "osc_2_detune": 0.03,
         "noise_level": 0.05, "filter_cutoff": 0.65, "filter_resonance": 0.30, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.10, "amp_release": 0.15,
         "filter_env_amount": 0.75, "filter_env_decay": 0.25,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.55, "distortion_mix": 0.15, "reverb_mix": 0.15},
    ),

    # =========================================================
    # LEADS  (14)
    # =========================================================
    concept(
        "Bright Lead",
        ["Bright Lead", "shiny lead", "open lead", "uplifting lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.50, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.02, "amp_decay": 0.35, "amp_sustain": 0.85, "amp_release": 0.35,
         "filter_env_amount": 0.60, "filter_env_decay": 0.40,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.40, "distortion_mix": 0.10, "reverb_mix": 0.25},
    ),
    concept(
        "Supersaw Lead",
        ["Supersaw Lead", "supersaw", "trance supersaw", "EDM supersaw", "anthemic lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.40,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.02, "amp_decay": 0.30, "amp_sustain": 0.90, "amp_release": 0.40,
         "filter_env_amount": 0.55, "filter_env_decay": 0.35,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.02, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.20, "distortion_mix": 0.15, "reverb_mix": 0.35},
    ),
    concept(
        "Hard Lead",
        ["Hard Lead", "screaming lead", "aggressive lead", "shred lead"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SAW, "osc_mix": 0.60, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.80, "filter_resonance": 0.45, "filter_type": LP,
         "amp_attack": 0.02, "amp_decay": 0.30, "amp_sustain": 0.85, "amp_release": 0.30,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.45, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.15,
         "velocity_to_filter": 0.40, "distortion_mix": 0.60, "reverb_mix": 0.15},
    ),
    concept(
        "Vintage Lead",
        ["Vintage Lead", "moog lead", "analog lead", "70s lead", "minimoog lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.45, "osc_2_detune": 0.08,
         "noise_level": 0.0, "filter_cutoff": 0.60, "filter_resonance": 0.30, "filter_type": LP,
         "amp_attack": 0.03, "amp_decay": 0.40, "amp_sustain": 0.75, "amp_release": 0.40,
         "filter_env_amount": 0.60, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.45, "distortion_mix": 0.20, "reverb_mix": 0.30},
    ),
    concept(
        "Square Lead",
        ["Square Lead", "chip lead", "8bit lead", "pulse lead"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.02, "amp_decay": 0.30, "amp_sustain": 0.85, "amp_release": 0.25,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.20, "distortion_mix": 0.10, "reverb_mix": 0.20},
    ),
    concept(
        "Sine Lead",
        ["Sine Lead", "pure sine", "flute-like lead", "sub lead"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.20, "osc_2_detune": 0.0,
         "noise_level": 0.0, "filter_cutoff": 0.65, "filter_resonance": 0.0, "filter_type": LP,
         "amp_attack": 0.05, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.45,
         "filter_env_amount": 0.50, "filter_env_decay": 0.40,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.0, "reverb_mix": 0.30},
    ),
    concept(
        "Wah Lead",
        ["Wah Lead", "wah-wah lead", "auto-wah lead", "funky wah"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.50, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.50, "filter_type": BP,
         "amp_attack": 0.02, "amp_decay": 0.35, "amp_sustain": 0.80, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.55, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.60,
         "velocity_to_filter": 0.55, "distortion_mix": 0.10, "reverb_mix": 0.20},
    ),
    concept(
        "Acid Lead",
        ["Acid Lead", "303 acid lead", "squelchy lead", "acid house lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.15, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.45, "filter_resonance": 0.70, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.50, "amp_release": 0.20,
         "filter_env_amount": 0.80, "filter_env_decay": 0.25,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.40, "distortion_mix": 0.35, "reverb_mix": 0.15},
    ),
    concept(
        "Plucky Lead",
        ["Plucky Lead", "plucked lead", "short lead", "staccato lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.70, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.20, "amp_release": 0.25,
         "filter_env_amount": 0.65, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.45, "distortion_mix": 0.05, "reverb_mix": 0.25},
    ),
    concept(
        "Detuned Lead",
        ["Detuned Lead", "wide lead", "chorused lead", "fat lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.45,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.04, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.40,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.25, "distortion_mix": 0.10, "reverb_mix": 0.30},
    ),
    concept(
        "80s Lead",
        ["80s Lead", "synthwave lead", "retro lead", "VHS lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.45, "osc_2_detune": 0.15,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.25, "filter_type": LP,
         "amp_attack": 0.03, "amp_decay": 0.35, "amp_sustain": 0.80, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.35, "distortion_mix": 0.15, "reverb_mix": 0.40},
    ),
    concept(
        "Vocal Lead",
        ["Vocal Lead", "ooh lead", "vox lead", "vowel lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.06,
         "noise_level": 0.02, "filter_cutoff": 0.55, "filter_resonance": 0.40, "filter_type": BP,
         "amp_attack": 0.05, "amp_decay": 0.40, "amp_sustain": 0.80, "amp_release": 0.40,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.03, "lfo_to_cutoff": 0.20,
         "velocity_to_filter": 0.30, "distortion_mix": 0.05, "reverb_mix": 0.30},
    ),
    concept(
        "Stab Lead",
        ["Stab Lead", "house stab", "synth stab", "chord stab"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.55, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.70, "filter_resonance": 0.30, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.15, "amp_sustain": 0.0, "amp_release": 0.10,
         "filter_env_amount": 0.75, "filter_env_decay": 0.20,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.10, "reverb_mix": 0.25},
    ),
    concept(
        "Trance Lead",
        ["Trance Lead", "uplifting trance lead", "anthem lead", "euphoric lead"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.25,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.04, "amp_decay": 0.35, "amp_sustain": 0.85, "amp_release": 0.45,
         "filter_env_amount": 0.60, "filter_env_decay": 0.40,
         "lfo_rate": 0.45, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.30, "distortion_mix": 0.15, "reverb_mix": 0.45},
    ),

    # =========================================================
    # PADS  (14)
    # =========================================================
    concept(
        "Warm Pad",
        ["Warm Pad", "soft warm pad", "cozy pad", "vintage pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.55, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.80,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.30, "distortion_mix": 0.0, "reverb_mix": 0.60},
    ),
    concept(
        "Ambient Pad",
        ["Ambient Pad", "atmospheric pad", "ambient texture", "ethereal pad"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.02, "filter_cutoff": 0.50, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.70, "amp_decay": 0.60, "amp_sustain": 0.85, "amp_release": 0.90,
         "filter_env_amount": 0.55, "filter_env_decay": 0.55,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.15,
         "velocity_to_filter": 0.20, "distortion_mix": 0.0, "reverb_mix": 0.85},
    ),
    concept(
        "String Pad",
        ["String Pad", "synthetic strings pad", "string ensemble pad", "lush strings pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.20,
         "noise_level": 0.02, "filter_cutoff": 0.60, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.45, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.75,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.02, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.70},
    ),
    concept(
        "Bright Pad",
        ["Bright Pad", "shimmering pad", "open pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.0, "filter_cutoff": 0.80, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.45, "amp_decay": 0.50, "amp_sustain": 0.85, "amp_release": 0.75,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.20, "distortion_mix": 0.0, "reverb_mix": 0.55},
    ),
    concept(
        "Dark Pad",
        ["Dark Pad", "horror pad", "dystopian pad", "shadowy pad"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.10,
         "noise_level": 0.05, "filter_cutoff": 0.25, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.55, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.80,
         "filter_env_amount": 0.50, "filter_env_decay": 0.55,
         "lfo_rate": 0.15, "lfo_to_pitch": 0.06, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.20, "distortion_mix": 0.05, "reverb_mix": 0.75},
    ),
    concept(
        "Choir Pad",
        ["Choir Pad", "vocal choir pad", "aah choir", "human voices pad"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.55, "osc_2_detune": 0.08,
         "noise_level": 0.02, "filter_cutoff": 0.55, "filter_resonance": 0.35, "filter_type": BP,
         "amp_attack": 0.40, "amp_decay": 0.50, "amp_sustain": 0.85, "amp_release": 0.70,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.80},
    ),
    concept(
        "Detuned Pad",
        ["Detuned Pad", "chorused pad", "wide pad", "spread pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.50,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.50, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.75,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.20, "distortion_mix": 0.0, "reverb_mix": 0.65},
    ),
    concept(
        "Evolving Pad",
        ["Evolving Pad", "moving pad", "morphing pad", "filter sweep pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.0, "filter_cutoff": 0.50, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.60, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.85,
         "filter_env_amount": 0.55, "filter_env_decay": 0.60,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.45,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.70},
    ),
    concept(
        "Glassy Pad",
        ["Glassy Pad", "crystal pad", "ice pad", "shimmer pad"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.45, "osc_2_detune": 0.04,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.30, "filter_type": HP,
         "amp_attack": 0.30, "amp_decay": 0.55, "amp_sustain": 0.75, "amp_release": 0.65,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.15,
         "velocity_to_filter": 0.20, "distortion_mix": 0.0, "reverb_mix": 0.70},
    ),
    concept(
        "Cinematic Pad",
        ["Cinematic Pad", "film pad", "score pad", "epic pad", "trailer pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.55, "osc_2_detune": 0.20,
         "noise_level": 0.02, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.65, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.85,
         "filter_env_amount": 0.55, "filter_env_decay": 0.55,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.85},
    ),
    concept(
        "Drone Pad",
        ["Drone Pad", "drone", "deep drone", "ambient drone"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SAW, "osc_mix": 0.50, "osc_2_detune": 0.05,
         "noise_level": 0.05, "filter_cutoff": 0.30, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.55, "amp_decay": 0.20, "amp_sustain": 0.95, "amp_release": 0.85,
         "filter_env_amount": 0.50, "filter_env_decay": 0.30,
         "lfo_rate": 0.10, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.75},
    ),
    concept(
        "Vocal Pad",
        ["Vocal Pad", "voxy pad", "singing pad"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.45, "osc_2_detune": 0.08,
         "noise_level": 0.02, "filter_cutoff": 0.60, "filter_resonance": 0.40, "filter_type": BP,
         "amp_attack": 0.45, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.70,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.15,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.70},
    ),
    concept(
        "Hybrid Pad",
        ["Hybrid Pad", "modern hybrid pad", "tex-pad"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.20,
         "noise_level": 0.05, "filter_cutoff": 0.55, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.45, "amp_decay": 0.55, "amp_sustain": 0.80, "amp_release": 0.70,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.03, "lfo_to_cutoff": 0.20,
         "velocity_to_filter": 0.25, "distortion_mix": 0.10, "reverb_mix": 0.65},
    ),
    concept(
        "Synth Brass Pad",
        ["Synth Brass Pad", "brass pad", "80s brass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.55, "filter_resonance": 0.25, "filter_type": LP,
         "amp_attack": 0.10, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.45,
         "filter_env_amount": 0.65, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.45, "distortion_mix": 0.10, "reverb_mix": 0.40},
    ),

    # =========================================================
    # PLUCKS / BELLS / MALLETS  (12)
    # =========================================================
    concept(
        "Soft Pluck",
        ["Soft Pluck", "gentle pluck", "smooth pluck", "warm pluck"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.03,
         "noise_level": 0.0, "filter_cutoff": 0.60, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.35, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.55, "distortion_mix": 0.0, "reverb_mix": 0.30},
    ),
    concept(
        "Hard Pluck",
        ["Hard Pluck", "tight pluck", "snappy pluck", "EDM pluck"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.30, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.70, "filter_env_decay": 0.20,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.50, "distortion_mix": 0.20, "reverb_mix": 0.20},
    ),
    concept(
        "Bell Pluck",
        ["Bell Pluck", "synth bell", "FM bell", "tubular bell"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.50, "osc_2_detune": 0.06,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.20, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.0, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.50,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.0, "reverb_mix": 0.55},
    ),
    concept(
        "Music Box",
        ["Music Box", "lullaby music box", "child music box"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.30, "osc_2_detune": 0.02,
         "noise_level": 0.0, "filter_cutoff": 0.80, "filter_resonance": 0.15, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.45, "amp_sustain": 0.0, "amp_release": 0.35,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.30, "distortion_mix": 0.0, "reverb_mix": 0.50},
    ),
    concept(
        "Marimba",
        ["Marimba", "marimba pluck", "wood mallet"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.35, "osc_2_detune": 0.03,
         "noise_level": 0.02, "filter_cutoff": 0.70, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.35, "amp_sustain": 0.0, "amp_release": 0.25,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.50, "distortion_mix": 0.0, "reverb_mix": 0.35},
    ),
    concept(
        "Kalimba",
        ["Kalimba", "thumb piano", "african kalimba"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.02,
         "noise_level": 0.03, "filter_cutoff": 0.65, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.40, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.35,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.45, "distortion_mix": 0.0, "reverb_mix": 0.40},
    ),
    concept(
        "Glass Pluck",
        ["Glass Pluck", "crystal pluck", "ice pluck"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.0, "filter_cutoff": 0.80, "filter_resonance": 0.35, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.0, "reverb_mix": 0.60},
    ),
    concept(
        "Plucky Synth",
        ["Plucky Synth", "EDM pluck synth", "trance pluck"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.50, "osc_2_detune": 0.10,
         "noise_level": 0.0, "filter_cutoff": 0.75, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.0, "amp_release": 0.25,
         "filter_env_amount": 0.70, "filter_env_decay": 0.25,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.45, "distortion_mix": 0.10, "reverb_mix": 0.30},
    ),
    concept(
        "Acoustic Pluck",
        ["Acoustic Pluck", "guitar-like pluck", "nylon pluck"],
        {"osc_1_waveform": TRI, "osc_2_waveform": SINE, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.04, "filter_cutoff": 0.55, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.40, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.50, "distortion_mix": 0.05, "reverb_mix": 0.30},
    ),
    concept(
        "Mallet Pluck",
        ["Mallet Pluck", "mallet hit", "soft mallet"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.03,
         "noise_level": 0.02, "filter_cutoff": 0.65, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.40, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.45, "distortion_mix": 0.0, "reverb_mix": 0.35},
    ),
    concept(
        "Steel Drum",
        ["Steel Drum", "caribbean steel drum", "calypso drum"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.45, "osc_2_detune": 0.10,
         "noise_level": 0.05, "filter_cutoff": 0.70, "filter_resonance": 0.25, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.40, "amp_sustain": 0.0, "amp_release": 0.30,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.45, "distortion_mix": 0.0, "reverb_mix": 0.35},
    ),
    concept(
        "Tine Bell",
        ["Tine Bell", "metal bell", "chime bell"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.55, "osc_2_detune": 0.07,
         "noise_level": 0.0, "filter_cutoff": 0.85, "filter_resonance": 0.20, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.50, "amp_sustain": 0.0, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.35, "distortion_mix": 0.0, "reverb_mix": 0.60},
    ),

    # =========================================================
    # STRINGS / BRASS / WIND  (11)
    # =========================================================
    concept(
        "Ensemble Strings",
        ["Ensemble Strings", "orchestral strings", "string section", "symphonic strings"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.18,
         "noise_level": 0.02, "filter_cutoff": 0.60, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.20, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.55,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.35, "distortion_mix": 0.0, "reverb_mix": 0.65},
    ),
    concept(
        "Solo Violin",
        ["Solo Violin", "violin", "solo strings", "fiddle"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.05, "filter_cutoff": 0.65, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.10, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.45, "lfo_to_pitch": 0.08, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.45, "distortion_mix": 0.0, "reverb_mix": 0.55},
    ),
    concept(
        "Cello",
        ["Cello", "deep cello", "bowed cello"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.05, "filter_cutoff": 0.45, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.15, "amp_decay": 0.40, "amp_sustain": 0.85, "amp_release": 0.55,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.06, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.35, "distortion_mix": 0.0, "reverb_mix": 0.55},
    ),
    concept(
        "Pizzicato",
        ["Pizzicato", "pizz strings", "plucked strings"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.03, "filter_cutoff": 0.65, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.55, "filter_env_decay": 0.25,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.0, "reverb_mix": 0.35},
    ),
    concept(
        "Brass Section",
        ["Brass Section", "synth brass", "horn section"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.08,
         "noise_level": 0.02, "filter_cutoff": 0.60, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.05, "amp_decay": 0.30, "amp_sustain": 0.85, "amp_release": 0.40,
         "filter_env_amount": 0.65, "filter_env_decay": 0.40,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.05,
         "velocity_to_filter": 0.55, "distortion_mix": 0.10, "reverb_mix": 0.45},
    ),
    concept(
        "Trumpet",
        ["Trumpet", "solo trumpet", "fanfare trumpet"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.45, "osc_2_detune": 0.04,
         "noise_level": 0.04, "filter_cutoff": 0.65, "filter_resonance": 0.30, "filter_type": BP,
         "amp_attack": 0.05, "amp_decay": 0.30, "amp_sustain": 0.85, "amp_release": 0.35,
         "filter_env_amount": 0.65, "filter_env_decay": 0.35,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.50, "distortion_mix": 0.05, "reverb_mix": 0.40},
    ),
    concept(
        "French Horn",
        ["French Horn", "horn", "orchestral horn"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.45, "osc_2_detune": 0.05,
         "noise_level": 0.03, "filter_cutoff": 0.50, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.10, "amp_decay": 0.35, "amp_sustain": 0.85, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.0, "reverb_mix": 0.50},
    ),
    concept(
        "Synth Flute",
        ["Synth Flute", "flute", "soft flute", "breathy flute"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.03,
         "noise_level": 0.20, "filter_cutoff": 0.65, "filter_resonance": 0.05, "filter_type": LP,
         "amp_attack": 0.08, "amp_decay": 0.35, "amp_sustain": 0.85, "amp_release": 0.40,
         "filter_env_amount": 0.55, "filter_env_decay": 0.35,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.06, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.25, "distortion_mix": 0.0, "reverb_mix": 0.45},
    ),
    concept(
        "Pan Flute",
        ["Pan Flute", "andean pan flute", "wood flute"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.25, "osc_2_detune": 0.03,
         "noise_level": 0.18, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.06, "amp_decay": 0.35, "amp_sustain": 0.80, "amp_release": 0.35,
         "filter_env_amount": 0.55, "filter_env_decay": 0.30,
         "lfo_rate": 0.40, "lfo_to_pitch": 0.07, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.30, "distortion_mix": 0.0, "reverb_mix": 0.55},
    ),
    concept(
        "Sax",
        ["Sax", "saxophone", "jazz sax"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.45, "osc_2_detune": 0.05,
         "noise_level": 0.06, "filter_cutoff": 0.55, "filter_resonance": 0.30, "filter_type": BP,
         "amp_attack": 0.04, "amp_decay": 0.35, "amp_sustain": 0.80, "amp_release": 0.40,
         "filter_env_amount": 0.55, "filter_env_decay": 0.35,
         "lfo_rate": 0.35, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.55, "distortion_mix": 0.05, "reverb_mix": 0.40},
    ),
    concept(
        "Brass Stab",
        ["Brass Stab", "horn stab", "punchy brass"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.08,
         "noise_level": 0.02, "filter_cutoff": 0.65, "filter_resonance": 0.20, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.70, "filter_env_decay": 0.20,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.50, "distortion_mix": 0.10, "reverb_mix": 0.30},
    ),

    # =========================================================
    # FX / TEXTURES  (9)
    # =========================================================
    concept(
        "Riser FX",
        ["Riser FX", "uplifter", "riser sweep", "build-up riser"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.55, "osc_2_detune": 0.20,
         "noise_level": 0.30, "filter_cutoff": 0.80, "filter_resonance": 0.40, "filter_type": HP,
         "amp_attack": 0.70, "amp_decay": 0.30, "amp_sustain": 0.95, "amp_release": 0.20,
         "filter_env_amount": 0.85, "filter_env_decay": 0.85,
         "lfo_rate": 0.50, "lfo_to_pitch": 0.40, "lfo_to_cutoff": 0.50,
         "velocity_to_filter": 0.10, "distortion_mix": 0.15, "reverb_mix": 0.65},
    ),
    concept(
        "Downer FX",
        ["Downer FX", "downlifter", "falling sweep"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.25, "filter_cutoff": 0.60, "filter_resonance": 0.30, "filter_type": LP,
         "amp_attack": 0.40, "amp_decay": 0.55, "amp_sustain": 0.65, "amp_release": 0.85,
         "filter_env_amount": 0.45, "filter_env_decay": 0.80,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.30, "lfo_to_cutoff": 0.50,
         "velocity_to_filter": 0.10, "distortion_mix": 0.10, "reverb_mix": 0.70},
    ),
    concept(
        "Sweep FX",
        ["Sweep FX", "filter sweep", "noise sweep", "wash sweep"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.15,
         "noise_level": 0.40, "filter_cutoff": 0.50, "filter_resonance": 0.45, "filter_type": BP,
         "amp_attack": 0.40, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.50,
         "filter_env_amount": 0.55, "filter_env_decay": 0.55,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.65,
         "velocity_to_filter": 0.15, "distortion_mix": 0.05, "reverb_mix": 0.65},
    ),
    concept(
        "Whoosh FX",
        ["Whoosh FX", "whoosh transition", "white noise whoosh"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.30, "osc_2_detune": 0.10,
         "noise_level": 0.85, "filter_cutoff": 0.75, "filter_resonance": 0.10, "filter_type": HP,
         "amp_attack": 0.30, "amp_decay": 0.35, "amp_sustain": 0.50, "amp_release": 0.50,
         "filter_env_amount": 0.75, "filter_env_decay": 0.65,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.40,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.70},
    ),
    concept(
        "Hit FX",
        ["Hit FX", "impact hit", "trailer hit", "boom hit"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SAW, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.25, "filter_cutoff": 0.55, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.10, "amp_release": 0.55,
         "filter_env_amount": 0.70, "filter_env_decay": 0.45,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.10, "distortion_mix": 0.35, "reverb_mix": 0.65},
    ),
    concept(
        "Sparkle FX",
        ["Sparkle FX", "shimmer fx", "twinkle fx"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.45, "osc_2_detune": 0.05,
         "noise_level": 0.05, "filter_cutoff": 0.85, "filter_resonance": 0.40, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.25,
         "filter_env_amount": 0.60, "filter_env_decay": 0.30,
         "lfo_rate": 0.65, "lfo_to_pitch": 0.10, "lfo_to_cutoff": 0.20,
         "velocity_to_filter": 0.30, "distortion_mix": 0.0, "reverb_mix": 0.70},
    ),
    concept(
        "Reverse FX",
        ["Reverse FX", "reverse cymbal", "backwards fx"],
        {"osc_1_waveform": SAW, "osc_2_waveform": TRI, "osc_mix": 0.40, "osc_2_detune": 0.10,
         "noise_level": 0.30, "filter_cutoff": 0.65, "filter_resonance": 0.25, "filter_type": HP,
         "amp_attack": 0.55, "amp_decay": 0.50, "amp_sustain": 0.85, "amp_release": 0.10,
         "filter_env_amount": 0.55, "filter_env_decay": 0.40,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.15,
         "velocity_to_filter": 0.15, "distortion_mix": 0.05, "reverb_mix": 0.65},
    ),
    concept(
        "Atmosphere FX",
        ["Atmosphere FX", "atmosphere texture", "soundscape fx", "ambient texture"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SAW, "osc_mix": 0.45, "osc_2_detune": 0.15,
         "noise_level": 0.20, "filter_cutoff": 0.40, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.65, "amp_decay": 0.55, "amp_sustain": 0.85, "amp_release": 0.85,
         "filter_env_amount": 0.55, "filter_env_decay": 0.65,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.07, "lfo_to_cutoff": 0.20,
         "velocity_to_filter": 0.15, "distortion_mix": 0.05, "reverb_mix": 0.85},
    ),
    concept(
        "Sci-Fi FX",
        ["Sci-Fi FX", "alien fx", "robotic fx", "space fx"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SAW, "osc_mix": 0.50, "osc_2_detune": 0.20,
         "noise_level": 0.15, "filter_cutoff": 0.65, "filter_resonance": 0.45, "filter_type": BP,
         "amp_attack": 0.10, "amp_decay": 0.40, "amp_sustain": 0.55, "amp_release": 0.45,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.70, "lfo_to_pitch": 0.30, "lfo_to_cutoff": 0.45,
         "velocity_to_filter": 0.25, "distortion_mix": 0.20, "reverb_mix": 0.60},
    ),

    # =========================================================
    # DRUMS (synthesized)  (7)
    # =========================================================
    concept(
        "Synth Kick",
        ["Synth Kick", "808 kick", "techno kick", "EDM kick"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.10, "osc_2_detune": 0.0,
         "noise_level": 0.05, "filter_cutoff": 0.25, "filter_resonance": 0.0, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.65, "filter_env_decay": 0.20,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.25, "reverb_mix": 0.05},
    ),
    concept(
        "Synth Snare",
        ["Synth Snare", "EDM snare", "trap snare"],
        {"osc_1_waveform": TRI, "osc_2_waveform": SINE, "osc_mix": 0.35, "osc_2_detune": 0.05,
         "noise_level": 0.65, "filter_cutoff": 0.55, "filter_resonance": 0.20, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.25, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.55, "filter_env_decay": 0.25,
         "lfo_rate": 0.30, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.40, "distortion_mix": 0.20, "reverb_mix": 0.20},
    ),
    concept(
        "Synth Hi-Hat",
        ["Synth Hi-Hat", "closed hat", "hi-hat", "tch hi-hat"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SQR, "osc_mix": 0.35, "osc_2_detune": 0.10,
         "noise_level": 0.85, "filter_cutoff": 0.85, "filter_resonance": 0.15, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.15, "amp_sustain": 0.0, "amp_release": 0.10,
         "filter_env_amount": 0.50, "filter_env_decay": 0.15,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.10},
    ),
    concept(
        "Synth Clap",
        ["Synth Clap", "EDM clap", "house clap"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SAW, "osc_mix": 0.40, "osc_2_detune": 0.05,
         "noise_level": 0.75, "filter_cutoff": 0.75, "filter_resonance": 0.20, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.55, "filter_env_decay": 0.20,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.10, "reverb_mix": 0.25},
    ),
    concept(
        "Synth Tom",
        ["Synth Tom", "tom drum", "low tom"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SINE, "osc_mix": 0.30, "osc_2_detune": 0.0,
         "noise_level": 0.10, "filter_cutoff": 0.35, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.25,
         "filter_env_amount": 0.60, "filter_env_decay": 0.25,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.25, "distortion_mix": 0.10, "reverb_mix": 0.15},
    ),
    concept(
        "Rim Shot",
        ["Rim Shot", "rim click", "side stick"],
        {"osc_1_waveform": SQR, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.05,
         "noise_level": 0.30, "filter_cutoff": 0.70, "filter_resonance": 0.40, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.15, "amp_sustain": 0.0, "amp_release": 0.10,
         "filter_env_amount": 0.55, "filter_env_decay": 0.15,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.35, "distortion_mix": 0.05, "reverb_mix": 0.15},
    ),
    concept(
        "Cymbal",
        ["Cymbal", "synth cymbal", "crash cymbal"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SQR, "osc_mix": 0.40, "osc_2_detune": 0.20,
         "noise_level": 0.90, "filter_cutoff": 0.85, "filter_resonance": 0.05, "filter_type": HP,
         "amp_attack": 0.0, "amp_decay": 0.55, "amp_sustain": 0.0, "amp_release": 0.55,
         "filter_env_amount": 0.50, "filter_env_decay": 0.45,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.15, "distortion_mix": 0.10, "reverb_mix": 0.30},
    ),

    # =========================================================
    # WORLD / EXOTIC  (5)
    # =========================================================
    concept(
        "Sitar",
        ["Sitar", "indian sitar", "raga sitar"],
        {"osc_1_waveform": SAW, "osc_2_waveform": SAW, "osc_mix": 0.45, "osc_2_detune": 0.20,
         "noise_level": 0.05, "filter_cutoff": 0.60, "filter_resonance": 0.45, "filter_type": BP,
         "amp_attack": 0.0, "amp_decay": 0.45, "amp_sustain": 0.10, "amp_release": 0.35,
         "filter_env_amount": 0.65, "filter_env_decay": 0.40,
         "lfo_rate": 0.45, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.20,
         "velocity_to_filter": 0.40, "distortion_mix": 0.05, "reverb_mix": 0.45},
    ),
    concept(
        "Tribal Drum",
        ["Tribal Drum", "djembe", "ethnic drum"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.30, "osc_2_detune": 0.05,
         "noise_level": 0.25, "filter_cutoff": 0.45, "filter_resonance": 0.15, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.30, "amp_sustain": 0.0, "amp_release": 0.20,
         "filter_env_amount": 0.55, "filter_env_decay": 0.25,
         "lfo_rate": 0.25, "lfo_to_pitch": 0.0, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.30, "distortion_mix": 0.05, "reverb_mix": 0.30},
    ),
    concept(
        "Choir Ah",
        ["Choir Ah", "choir aah", "human aah", "vocal choir"],
        {"osc_1_waveform": SINE, "osc_2_waveform": TRI, "osc_mix": 0.55, "osc_2_detune": 0.10,
         "noise_level": 0.04, "filter_cutoff": 0.60, "filter_resonance": 0.35, "filter_type": BP,
         "amp_attack": 0.30, "amp_decay": 0.50, "amp_sustain": 0.85, "amp_release": 0.60,
         "filter_env_amount": 0.55, "filter_env_decay": 0.45,
         "lfo_rate": 0.20, "lfo_to_pitch": 0.04, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.30, "distortion_mix": 0.0, "reverb_mix": 0.75},
    ),
    concept(
        "Spiritual Drone",
        ["Spiritual Drone", "om drone", "meditation drone"],
        {"osc_1_waveform": SINE, "osc_2_waveform": SAW, "osc_mix": 0.40, "osc_2_detune": 0.10,
         "noise_level": 0.05, "filter_cutoff": 0.35, "filter_resonance": 0.10, "filter_type": LP,
         "amp_attack": 0.65, "amp_decay": 0.30, "amp_sustain": 0.95, "amp_release": 0.85,
         "filter_env_amount": 0.50, "filter_env_decay": 0.40,
         "lfo_rate": 0.15, "lfo_to_pitch": 0.05, "lfo_to_cutoff": 0.10,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.80},
    ),
    concept(
        "Chiptune Lead",
        ["Chiptune Lead", "8-bit lead", "game boy lead", "NES lead"],
        {"osc_1_waveform": SQR, "osc_2_waveform": SQR, "osc_mix": 0.50, "osc_2_detune": 0.0,
         "noise_level": 0.0, "filter_cutoff": 0.95, "filter_resonance": 0.0, "filter_type": LP,
         "amp_attack": 0.0, "amp_decay": 0.20, "amp_sustain": 0.85, "amp_release": 0.10,
         "filter_env_amount": 0.50, "filter_env_decay": 0.20,
         "lfo_rate": 0.55, "lfo_to_pitch": 0.06, "lfo_to_cutoff": 0.0,
         "velocity_to_filter": 0.10, "distortion_mix": 0.05, "reverb_mix": 0.05},
    ),
]


def _apply_modifiers(params, prompt):
    """Mutate params in-place using known modifier words present in the prompt."""
    low = prompt.lower()
    for word, deltas in MODIFIERS.items():
        if word in low:
            for k, delta in deltas.items():
                if k.endswith("_force"):
                    real_key = k[: -len("_force")]
                    params[real_key] = delta
                else:
                    params[k] = max(0.0, min(1.0, params.get(k, 0.0) + delta))


def _waveform_neighbors(value):
    steps = [SINE, TRI, SAW, SQR]
    if value not in steps:
        # snap to nearest
        value = min(steps, key=lambda x: abs(x - value))
    idx = steps.index(value)
    neighbors = []
    if idx > 0:
        neighbors.append(steps[idx - 1])
    if idx < len(steps) - 1:
        neighbors.append(steps[idx + 1])
    return neighbors or [value]


def _filter_neighbors(value):
    steps = [LP, BP, HP]
    if value not in steps:
        value = min(steps, key=lambda x: abs(x - value))
    idx = steps.index(value)
    neighbors = []
    if idx > 0:
        neighbors.append(steps[idx - 1])
    if idx < len(steps) - 1:
        neighbors.append(steps[idx + 1])
    return neighbors or [value]


def _jitter_continuous(value, std, rng):
    return max(0.0, min(1.0, value + rng.gauss(0.0, std)))


def generate_variation(concept_def, prompt, rng):
    base = dict(concept_def["base"])
    jitter = concept_def["jitter"]
    _apply_modifiers(base, prompt)

    for name in PARAM_NAMES:
        idx = PARAM_NAMES.index(name)
        if idx in DISCRETE_INDICES:
            if rng.random() < jitter.get("discrete_swap_prob", 0.05):
                if name == "filter_type":
                    base[name] = rng.choice(_filter_neighbors(base[name]))
                else:
                    base[name] = rng.choice(_waveform_neighbors(base[name]))
            continue
        std = jitter.get(name, jitter.get("default", 0.04))
        base[name] = _jitter_continuous(base[name], std, rng)

    # Charter record shape expected by PresetDataset.
    continuous = {}
    categorical = {}
    for name in PARAM_NAMES:
        idx = PARAM_NAMES.index(name)
        if idx in DISCRETE_INDICES:
            categorical[name] = float(base[name])
        else:
            continuous[name] = float(base[name])
    return {
        "raw_text": prompt,
        "description": prompt,
        "name": concept_def["label"],
        "parameters": {"continuous": continuous, "binary": {}, "categorical": categorical},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "processed" / "synthetic_dataset.charter.npy")
    parser.add_argument("--per-prompt", type=int, default=50, help="Variations per (concept, prompt) pair.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix-anchors", action="store_true", help="Also include anchor_presets entries verbatim (no jitter), oversampled 100x.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    records = []
    family_counts = {}
    for concept_def in CONCEPTS:
        for prompt in concept_def["prompts"]:
            for _ in range(args.per_prompt):
                records.append(generate_variation(concept_def, prompt, rng))
        family_counts[concept_def["label"]] = len(concept_def["prompts"]) * args.per_prompt

    if args.mix_anchors:
        from scripts.prepare_dataset import _iter_anchor_records
        anchors = list(_iter_anchor_records(BASE_DIR / "data" / "raw" / "anchor_presets.json"))
        for _ in range(100):
            records.extend(anchors)
        print(f"Mixed in {len(anchors)*100} anchor records.")

    rng.shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.array(records, dtype=object), allow_pickle=True)

    total_prompts = sum(len(c["prompts"]) for c in CONCEPTS)
    print(f"Concepts          : {len(CONCEPTS)}")
    print(f"Unique prompts    : {total_prompts}")
    print(f"Variations/prompt : {args.per_prompt}")
    print(f"Total records     : {len(records)}")
    print(f"Output            : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
