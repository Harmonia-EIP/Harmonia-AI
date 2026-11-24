#!/usr/bin/env python3
"""Presets that imitate real recordings (FSD50K): everyday sounds the preset banks do not hold.

The banks are musical (DX7, OB-Xf, Surge): no door slam, no glass, no footsteps, so the generators had
never seen such a preset. Here each recording is imitated on the analog engine by a search that plays a
preset, listens, and keeps what gets closer by two measures at once:
- CLAP (the generators' ear): similarity of the preset's sound to the recording;
- the perceptual distance of src/presets/perceptual.py (timbre per 1/3 octave, envelope, movement).
A search on CLAP alone finds sounds CLAP likes and an independent judge does not; both together hold up
with the independent judge (CHANGELOG, v3.3). The search starts from what the analog generator draws for
the recording's sound.

Only recordings under CC0 or CC BY are used, speech and other human voice sounds are left out, and the
recordings themselves are never copied: the output is one line of synthesizer settings per recording, with
its title, labels, licence and author (data/v3/matched_fsd/, not in git).

With --samples (#43), the search may also play a recorded instrument (the sample source of charter 3.1, from the
"listen" pack the server plays): half of the first presets get one of the instruments that sound most like the
recording to CLAP, and the search can swap it for another of them, switch it off or on, or move it by an octave.
--classes instruments limits the recordings to the classes of musical instruments the packs hold.

    python scripts/v3/match_recordings.py --per-class 8 --workers 10
    python scripts/v3/match_recordings.py --samples --classes instruments --per-class 60 \
        --generator data/v4/generator/analog.pt --out data/samples/matched_fsd
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

OUT_DIR = BASE_DIR / "data" / "v3" / "matched_fsd"
SR = 48000
NOTE = 60
HOLD_SECONDS, TOTAL_SECONDS = 1.5, 3.5
LICENSES = ("creativecommons.org/publicdomain/zero", "creativecommons.org/licenses/by/")
# Human voice and body sounds: a synthesizer preset cannot stand for them.
VOICE = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
         "Conversation", "Whispering", "Human_voice", "Singing", "Male_singing", "Female_singing", "Chatter",
         "Crowd", "Cheering", "Shout", "Yell", "Screaming", "Laughter", "Giggle", "Chuckle_and_chortle",
         "Crying_and_sobbing", "Sigh", "Gasp", "Breathing", "Respiratory_sounds", "Cough", "Sneeze",
         "Burping_and_eructation", "Fart", "Chewing_and_mastication", "Speech_synthesizer", "Human_group_actions"}
PERCEPTUAL_WEIGHT = 0.006  # CLAP similarity given up per point of perceptual distance
# FSD50K classes (most specific label) of the pitched instruments the sample packs hold (samples/README.md):
# no drums (the packs have no unpitched percussion), no guitars (no guitar in the packs)
INSTRUMENT_CLASSES = ("Wind_instrument_and_woodwind_instrument", "Bowed_string_instrument", "Piano", "Trumpet",
                      "Brass_instrument", "Organ", "Keyboard_(musical)", "Harp", "Marimba_and_xylophone", "Harmonica",
                      "Plucked_string_instrument", "Glockenspiel", "Mallet_percussion", "Accordion", "Church_bell",
                      "Bell")
INSTRUMENTS_CLAP = BASE_DIR / "data" / "samples" / "instruments_clap.npz"  # scripts/samples/instrument_descriptors.py
CANDIDATES = 6  # instruments tried per recording
SILENT_LEVEL = 0.02  # below this level the recorded instrument is written as off
_targets: dict = {}


def load_clip(fsd50k: Path, fname: str) -> np.ndarray:
    """The recording as the engine will be compared to it: mono 48 kHz, from its onset, TOTAL_SECONDS long."""
    import soundfile as sf
    import soxr

    audio, sr = sf.read(str(fsd50k / "FSD50K.dev_audio" / f"{fname}.wav"), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="HQ")
    peak = float(np.abs(audio).max()) + 1e-9
    onset = int(np.argmax(np.abs(audio) > 0.01 * peak))
    audio = audio[max(0, onset - 48):]
    n = int(TOTAL_SECONDS * SR)
    return np.pad(audio, (0, max(0, n - len(audio))))[:n].astype(np.float64)


def _play(task):
    """Worker: play one normalized preset, return its audio and its perceptual distance to the recording."""
    from src.presets import perceptual
    from src.synth import v3_params as P
    from src.synth.engine_v3 import render_physical

    normalized, fname, fsd50k = task
    if fname not in _targets:
        _targets[fname] = perceptual.describe([load_clip(Path(fsd50k), fname)])
    stereo = render_physical(P.denormalize(normalized), NOTE, 100 / 127, HOLD_SECONDS, TOTAL_SECONDS, SR, 7)
    mono = np.nan_to_num(stereo.mean(axis=0))
    return mono.astype(np.float32), perceptual.distance(_targets[fname], perceptual.describe([mono]))


def pick_clips(fsd50k: Path, embeddings: Path, per_class: int, seed: int, cache: Path | None = None,
               classes: tuple | None = None):
    """Up to per_class recordings of every non-voice class (its most specific label), class by class in turn;
    only `classes` when given. The choice is kept in `cache` (reading every file's length takes minutes), so a
    resumed run starts at once."""
    import soundfile as sf

    z = np.load(embeddings)
    wanted = sorted(classes) if classes else None
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("per_class") == per_class and saved.get("seed") == seed and saved.get("classes") == wanted:
            index = {str(f): i for i, f in enumerate(z["fname"])}
            return [{**clip, "emb": z["emb"][index[clip["fname"]]].astype(np.float32)} for clip in saved["clips"]]
    info = json.loads((fsd50k / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for i in rng.permutation(len(z["fname"])):
        fname, labels = str(z["fname"][i]), str(z["labels"][i])
        leaf = labels.split(",")[0]
        meta = info.get(fname, {})
        if leaf in VOICE or not any(k in meta.get("license", "") for k in LICENSES) or (classes and leaf not in classes):
            continue
        chosen = by_class.setdefault(leaf, [])
        if len(chosen) >= per_class:
            continue
        duration = sf.info(str(fsd50k / "FSD50K.dev_audio" / f"{fname}.wav")).duration
        if not 0.25 <= duration <= 8.0:
            continue
        chosen.append({"fname": fname, "labels": labels, "title": str(z["title"][i]), "license": meta["license"],
                       "uploader": meta.get("uploader", ""), "emb": z["emb"][i].astype(np.float32)})
    rounds = [[c[k] for c in by_class.values() if len(c) > k] for k in range(per_class)]
    clips = [clip for group in rounds for clip in group]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"per_class": per_class, "seed": seed, "classes": wanted,
                                     "clips": [{k: v for k, v in c.items() if k != "emb"} for c in clips]},
                                    ensure_ascii=False), encoding="utf-8")
    return clips


def candidate_instruments(targets: np.ndarray, clap: Path = INSTRUMENTS_CLAP, count: int = CANDIDATES) -> np.ndarray:
    """(clips, count) charter choices of the instruments whose sound is closest to each recording's (CLAP)."""
    from src.samples.player import instrument_ids

    z = np.load(clap)
    choice_of = {name: c for c, name in instrument_ids().items()}
    choices = np.array([choice_of[str(i)] for i in z["ids"]])
    emb = z["emb"].astype(np.float32)
    order = np.argsort(-(targets @ emb.T), axis=1)[:, :count]
    return choices[order]


def _set_instrument(presets: np.ndarray, where: np.ndarray, choices: np.ndarray, levels: np.ndarray) -> None:
    """Normalized presets (..., N_PARAMS): those in `where` play `choices` at `levels`."""
    from src.synth import v3_params as P

    spec = P.BY_NAME["sample_instrument"]
    presets[..., P.INDEX["sample_instrument"]] = np.where(where, (choices - spec.start) / (spec.end - spec.start),
                                                           presets[..., P.INDEX["sample_instrument"]])
    presets[..., P.INDEX["sample_level"]] = np.where(where, levels, presets[..., P.INDEX["sample_level"]])


def add_instruments(first: np.ndarray, candidates: np.ndarray, rng) -> np.ndarray:
    """The second half of each clip's first presets (clips, pop, N_PARAMS) play its candidate instruments in turn,
    at a level between half and full."""
    out = first.copy()
    clips, pop, _ = out.shape
    where = np.zeros((clips, pop), dtype=bool)
    where[:, pop // 2:] = True
    choices = candidates[:, np.arange(pop) % candidates.shape[1]]
    _set_instrument(out, where, choices, rng.uniform(0.5, 1.0, (clips, pop)))
    return out


def mutate_instruments(kids: np.ndarray, candidates: np.ndarray, rng) -> np.ndarray:
    """The sample source's own moves: another candidate instrument (10 %), switched on with a candidate or off
    (5 %), an octave up or down (4 %). Level and start move with the continuous settings."""
    from src.synth import v3_params as P

    out = kids.copy()
    clips, pop, _ = out.shape
    pick = candidates[np.arange(clips)[:, None], rng.integers(0, candidates.shape[1], (clips, pop))]
    instrument, level = P.INDEX["sample_instrument"], P.INDEX["sample_level"]
    playing = (out[..., instrument] > 0) & (out[..., level] > 0)
    swap = playing & (rng.random((clips, pop)) < 0.10)
    _set_instrument(out, swap, pick, out[..., level])
    toggle = rng.random((clips, pop)) < 0.05
    _set_instrument(out, toggle & ~playing, pick, rng.uniform(0.5, 1.0, (clips, pop)))
    out[..., level] = np.where(toggle & playing, 0.0, out[..., level])
    tune = P.INDEX["sample_tune"]
    spec = P.SPECS[tune]
    step = 12.0 / (spec.end - spec.start)
    octave = rng.random((clips, pop)) < 0.04
    moved = out[..., tune] + rng.choice((-step, step), (clips, pop))
    out[..., tune] = np.where(octave & (moved >= -1e-9) & (moved <= 1 + 1e-9), np.clip(moved, 0.0, 1.0), out[..., tune])
    return out


def written(normalized: np.ndarray) -> np.ndarray:
    """Physical settings of a found preset; a sample source that plays nothing (no instrument, or a level too
    low to hear) is written off, with its other settings at their defaults."""
    from src.synth import v3_params as P

    physical = P.complete(P.denormalize(normalized) if len(normalized) == P.N_PARAMS else
                          np.array([P.from_normalized(s, v) for s, v in zip(P.SPECS, normalized)]))
    sample = [P.INDEX[name] for name in P.SAMPLE_PARAMS]
    if round(physical[sample[0]]) <= 0 or physical[sample[1]] < SILENT_LEVEL:
        physical[sample] = P.DEFAULTS[sample]
    return physical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--embeddings", type=Path, default=BASE_DIR / "data" / "v2" / "fsd50k_dev.npz",
                        help="CLAP embeddings of the recordings (scripts/v2/embed_fsd50k.py)")
    parser.add_argument("--generator", type=Path, default=BASE_DIR / "data" / "v3" / "generator" / "analog.pt")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--batch", type=int, default=24, help="recordings searched together")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=0, help="stop after this many new recordings (0: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", action="store_true", help="the search may also play a recorded instrument")
    parser.add_argument("--classes", default="", help="'instruments' or a comma-separated list of FSD50K classes")
    args = parser.parse_args()
    classes = (INSTRUMENT_CLASSES if args.classes == "instruments" else
               tuple(c for c in args.classes.split(",") if c) or None)

    # One thread per rendering worker (they inherit the environment): numpy/Accelerate and numba would
    # otherwise each start a thread per core in every worker. Set here, not at import, so that importing this
    # module (tests) leaves the process alone.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    import torch

    from src.synth import v3_params as P
    from src.v2.clap_audio import ClapEmbedder
    from src.v3 import codecs
    from src.v3.diffusion import Diffusion

    args.out.mkdir(parents=True, exist_ok=True)
    clips = pick_clips(args.fsd50k, args.embeddings, args.per_class, args.seed, args.out / "picked.json", classes)
    if args.limit:
        clips = clips[:args.limit]
    out_path = args.out / "fsd.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["fname"] for line in open(out_path, encoding="utf-8")}
    clips = [c for c in clips if c["fname"] not in done]
    print(f"{len(clips)} recordings to imitate ({len(done)} already done)", flush=True)
    if args.max_new:
        clips = clips[:args.max_new]

    diffusion = Diffusion.from_weights(torch.load(args.generator, map_location="cpu", weights_only=True))
    clap = ClapEmbedder(half=True)
    # the 46 parameters of charter 3.0, and with --samples the sample source (its discrete settings move on
    # their own: mutate_instruments)
    n_params = P.N_PARAMS if args.samples else P.CORE_PARAMS
    searched = P.SPECS[:n_params]
    continuous = np.array([spec.interval == 0.0 for spec in searched])
    discrete = [(i, int(round(spec.end - spec.start)) + 1) for i, spec in enumerate(searched)
                if spec.interval > 0 and spec.name not in P.SAMPLE_PARAMS]
    pop, mu = args.population, max(2, args.population // 4)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, open(out_path, "a", encoding="utf-8") as f:
        for b in range(0, len(clips), args.batch):
            group = clips[b:b + args.batch]
            target = np.stack([c["emb"] for c in group])
            target /= np.linalg.norm(target, axis=1, keepdims=True)
            names = [c["fname"] for c in group for _ in range(pop)]

            def score(presets: np.ndarray):
                flat = presets.reshape(-1, presets.shape[-1])
                played = pool.map(_play, [(x, n, str(args.fsd50k)) for x, n in zip(flat, names)], chunksize=8)
                emb = clap.embed_audio([p[0] for p in played], batch_size=64)
                emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
                cos = np.einsum("cnd,cd->cn", emb.reshape(len(group), pop, -1), target)
                dist = np.array([p[1] for p in played]).reshape(len(group), pop)
                return cos - PERCEPTUAL_WEIGHT * dist, cos, dist

            first = np.zeros((len(group), pop, n_params))
            for c in range(len(group)):
                cond = torch.from_numpy(np.repeat(target[c][None], pop, axis=0))
                drawn = diffusion.sample(cond, guidance=2.0, generator=torch.Generator().manual_seed(b + c)).numpy()
                first[c] = np.stack([P.normalize(codecs.analog_decode(v))[:n_params] for v in drawn])
            candidates = candidate_instruments(target) if args.samples else None
            if args.samples:
                first = add_instruments(first, candidates, rng)
            s, cos, dist = score(first)
            order = np.argsort(-s, axis=1)[:, :mu]
            elite = np.take_along_axis(first, order[:, :, None], axis=1)
            elite_s, elite_cos, elite_dist = (np.take_along_axis(a, order, axis=1) for a in (s, cos, dist))
            sigma = 0.15
            rows = np.arange(len(group))[:, None]
            for _ in range(args.generations):
                kids = np.where(rng.random((len(group), pop, n_params)) < 0.3,
                                elite[rows, rng.integers(0, mu, (len(group), pop))],
                                elite[rows, rng.integers(0, mu, (len(group), pop))])
                moves = sigma * rng.standard_normal(kids.shape) * (rng.random(kids.shape) < 0.5)
                kids = np.where(continuous, np.clip(kids + moves, 0.0, 1.0), kids)
                for i, n in discrete:
                    flip = rng.random((len(group), pop)) < 0.08
                    kids[..., i] = np.where(flip, rng.integers(0, n, (len(group), pop)) / max(1, n - 1), kids[..., i])
                if args.samples:
                    kids = mutate_instruments(kids, candidates, rng)
                s, cos, dist = score(kids)
                both = [np.concatenate([a, k], axis=1) for a, k in ((elite_s, s), (elite_cos, cos), (elite_dist, dist))]
                presets = np.concatenate([elite, kids], axis=1)
                order = np.argsort(-both[0], axis=1)[:, :mu]
                elite = np.take_along_axis(presets, order[:, :, None], axis=1)
                elite_s, elite_cos, elite_dist = (np.take_along_axis(a, order, axis=1) for a in both)
                sigma *= 0.96
            for c, clip in enumerate(group):
                row = {k: clip[k] for k in ("fname", "labels", "title", "license", "uploader")}
                row.update(cos=round(float(elite_cos[c, 0]), 4), distance=round(float(elite_dist[c, 0]), 3),
                           physical=[round(float(x), 6) for x in written(elite[c, 0])[:n_params]])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n = min(b + args.batch, len(clips))
            print(f"  {n}/{len(clips)} ({time.time() - t0:.0f}s) cos {elite_cos[:, 0].mean():.3f} "
                  f"distance {elite_dist[:, 0].mean():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
