#!/usr/bin/env python3
"""Benchmark step 2: score the generations with an independent judge (Microsoft CLAP 2023).

The judge never took part in training (the teacher is LAION CLAP). For every generation:
- text score: cosine between the judge's embedding of the rendered preset and of the English prompt;
- real-sound score: cosine to the centroid of real FSD50K *eval* recordings of the mapped class.

Runs in a separate environment (msclap needs transformers<5):
    python -m venv .venv-judge && .venv-judge/bin/pip install msclap==1.3.4 numba soundfile soxr
    .venv-judge/bin/python scripts/v2/benchmark_judge.py --generations data/v2/bench/generations.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.synth.engine import render_preset  # noqa: E402

JUDGE_SR = 44100
JUDGE_SECONDS = 7
NOTE, HOLD_S, TOTAL_S = 57, 1.5, 4.0  # A3: a different note from the bank's A2/A4 renders


def loudest_window(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) <= length:
        return audio
    energy = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    starts = np.arange(0, len(audio) - length + 1, JUDGE_SR // 10)
    start = int(starts[np.argmax(energy[starts + length] - energy[starts])])
    return audio[start : start + length]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def class_centroids(fsd50k: Path, classes, tmp: Path, judge, per_class: int, seed: int):
    rows = list(csv.DictReader(open(fsd50k / "FSD50K.ground_truth" / "eval.csv", newline="", encoding="utf-8")))
    rng = np.random.default_rng(seed)
    centroids = {}
    for cls in sorted(classes):
        # Prefer clips where the class is the most specific (first) label.
        leaf = [r["fname"] for r in rows if r["labels"].split(",")[0] == cls]
        anywhere = [r["fname"] for r in rows if cls in r["labels"].split(",") and r["fname"] not in leaf]
        picked = list(rng.permutation(leaf))[:per_class]
        picked += list(rng.permutation(anywhere))[: max(0, per_class - len(picked))]
        paths = []
        for fname in picked:
            audio, sr = sf.read(str(fsd50k / "FSD50K.eval_audio" / f"{fname}.wav"), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != JUDGE_SR:
                audio = soxr.resample(audio, sr, JUDGE_SR)
            path = tmp / f"fsd_{fname}.wav"
            sf.write(path, loudest_window(audio, JUDGE_SR * JUDGE_SECONDS), JUDGE_SR)
            paths.append(str(path))
        emb = normalize(judge.get_audio_embeddings(paths).detach().cpu().numpy())
        centroids[cls] = {"centroid": normalize(emb.mean(axis=0)), "clips": len(paths)}
    return centroids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=BASE_DIR / "data" / "v2" / "bench" / "generations.json")
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "bench" / "scores.json")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from msclap import CLAP

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generations = json.loads(args.generations.read_text(encoding="utf-8"))
    judge = CLAP(version="2023", use_cuda=False)

    def read_audio(audio_path, resample=True):
        # Same contract as CLAPWrapper.read_audio, without torchaudio.load (which now needs torchcodec + FFmpeg).
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        target = judge.args.sampling_rate
        if resample and sr != target:
            audio, sr = soxr.resample(audio, sr, target).astype(np.float32), target
        return torch.from_numpy(audio)[None, :], sr

    judge.read_audio = read_audio

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        paths, rms = [], []
        for i, g in enumerate(generations):
            audio = render_preset(g["values"], note=NOTE, hold_seconds=HOLD_S, total_seconds=TOTAL_S, sr=JUDGE_SR, q_mode=g["engine"])
            audio = np.nan_to_num(audio)
            rms.append(float(np.sqrt((audio.astype(np.float64) ** 2).mean())))
            path = tmp / f"gen_{i:05d}.wav"
            sf.write(path, audio, JUDGE_SR)
            paths.append(str(path))
        audio_emb = []
        for i in range(0, len(paths), 32):
            audio_emb.append(judge.get_audio_embeddings(paths[i : i + 32]).detach().cpu().numpy())
        audio_emb = normalize(np.concatenate(audio_emb))
        prompts = sorted({g["prompt_en"] for g in generations})
        text_emb = dict(zip(prompts, normalize(judge.get_text_embeddings(prompts).detach().cpu().numpy())))
        centroids = class_centroids(args.fsd50k, {g["fsd50k"] for g in generations if g["fsd50k"]}, tmp, judge, args.per_class, args.seed)

    for g, emb, level in zip(generations, audio_emb, rms):
        g["rms"] = level
        g["text_score"] = float(emb @ text_emb[g["prompt_en"]])
        g["real_score"] = float(emb @ centroids[g["fsd50k"]]["centroid"]) if g["fsd50k"] else None
        g["judge_emb"] = emb

    # Aggregate per system / engine / language.
    summary = defaultdict(lambda: defaultdict(list))
    for g in generations:
        key = f"{g['system']}@app1.{g['engine']}"
        summary[key][f"text_{g['lang']}"].append(g["text_score"])
        summary[key][f"text_{g['lang']}_{g['category']}"].append(g["text_score"])
        if g["real_score"] is not None:
            summary[key][f"real_{g['lang']}"].append(g["real_score"])
        summary[key][f"silent_{g['lang']}"].append(float(g["rms"] < 1e-4))
    by_pair = defaultdict(dict)
    for g in generations:
        by_pair[(g["system"], g["engine"], g["prompt_en"])][g["lang"]] = g["judge_emb"]
    for (system, engine, _), pair in by_pair.items():
        if "en" in pair and "fr" in pair:
            summary[f"{system}@app1.{engine}"]["fr_en_consistency"].append(float(pair["en"] @ pair["fr"]))

    table = {key: {metric: round(float(np.mean(v)), 4) for metric, v in sorted(metrics.items())} for key, metrics in summary.items()}
    for g in generations:
        del g["judge_emb"]
    args.out.write_text(json.dumps({"summary": table, "class_clips": {k: v["clips"] for k, v in centroids.items()}, "generations": generations},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    columns = ["text_en", "text_fr", "real_en", "real_fr", "fr_en_consistency", "silent_en"]
    print(f"{'system':36s}" + "".join(f"{c:>19s}" for c in columns))
    for key in sorted(table, key=lambda k: -table[k].get("text_en", 0)):
        print(f"{key:36s}" + "".join(f"{table[key].get(c, float('nan')):19.4f}" for c in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
