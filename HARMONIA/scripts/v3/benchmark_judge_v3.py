#!/usr/bin/env python3
"""Benchmark of the v3 run time, step 2: an independent judge (Microsoft CLAP 2023) scores the sounds.

The judge took no part in training or in the run time's search (both listen with LAION CLAP). Scores:
- text: cosine between the sound and the prompt (English wording of the prompt for the judge);
- real: for FSD50K class prompts, cosine between the sound and the centroid of real eval recordings of
  that class.
The same scores for the real recordings themselves give the ceiling.

Runs in its own environment (msclap needs transformers<5):
    python -m venv .venv-judge && .venv-judge/bin/pip install msclap==1.3.4 soundfile soxr
    .venv-judge/bin/python scripts/v3/benchmark_judge_v3.py v3.0 polish2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "v3" / "bench"
JUDGE_SR, JUDGE_SECONDS = 44100, 7


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def load_judge():
    import torch
    from msclap import CLAP

    torch.manual_seed(0)
    np.random.seed(0)
    judge = CLAP(version="2023", use_cuda=False)

    def read_audio(audio_path, resample=True):
        # same contract as CLAPWrapper.read_audio, without torchaudio.load (which needs torchcodec + FFmpeg)
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        target = judge.args.sampling_rate
        if resample and sr != target:
            audio, sr = soxr.resample(audio, sr, target).astype(np.float32), target
        return torch.from_numpy(audio)[None, :], sr

    judge.read_audio = read_audio
    return judge


def embed_audio(judge, paths) -> np.ndarray:
    out = [judge.get_audio_embeddings(paths[i:i + 32]).detach().cpu().numpy() for i in range(0, len(paths), 32)]
    return normalize(np.concatenate(out))


def class_centroids(judge, fsd50k: Path, classes, per_class: int, cache: Path):
    """Judge embeddings of real eval recordings per class: {class: (centroid, mean clip-to-text... )}."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if set(classes) <= set(z["classes"]):
            return {c: z["centroids"][i] for i, c in enumerate(z["classes"])}, \
                   {c: float(z["clip_to_centroid"][i]) for i, c in enumerate(z["classes"])}
    with open(fsd50k / "FSD50K.ground_truth" / "eval.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(0)
    centroids, spread = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for cls in sorted(classes):
            names = [r["fname"] for r in rows if r["labels"].split(",")[0] == cls]
            paths = []
            for fname in list(rng.permutation(names))[:per_class]:
                audio, sr = sf.read(str(fsd50k / "FSD50K.eval_audio" / f"{fname}.wav"), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)[: sr * JUDGE_SECONDS]
                path = Path(tmp) / f"{fname}.wav"
                sf.write(path, audio, sr)
                paths.append(str(path))
            emb = embed_audio(judge, paths)
            centroids[cls] = normalize(emb.mean(axis=0))
            # leave-one-out: how close a real recording is to the centroid of the others
            total = emb.sum(axis=0)
            loo = [float(e @ normalize(total - e)) for e in emb] if len(emb) > 1 else [1.0]
            spread[cls] = float(np.mean(loo))
    np.savez(cache, classes=np.array(sorted(classes)), centroids=np.stack([centroids[c] for c in sorted(classes)]),
             clip_to_centroid=np.array([spread[c] for c in sorted(classes)]))
    return centroids, spread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("systems", nargs="+")
    parser.add_argument("--dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    judge = load_judge()
    runs = {name: json.loads((args.dir / name / "generations.json").read_text(encoding="utf-8")) for name in args.systems}
    texts = sorted({r["judge"] for rows in runs.values() for r in rows})
    text_emb = dict(zip(texts, normalize(judge.get_text_embeddings(texts).detach().cpu().numpy())))
    classes = sorted({r["fsd_class"] for rows in runs.values() for r in rows if r.get("fsd_class")})
    centroids, spread = class_centroids(judge, args.fsd50k, classes, args.per_class, args.dir / "judge_classes.npz")
    table = {}
    for name, rows in runs.items():
        emb = embed_audio(judge, [str(args.dir / name / r["wav"]) for r in rows])
        for r, e in zip(rows, emb):
            r["text_score"] = float(e @ text_emb[r["judge"]])
            r["real_score"] = float(e @ centroids[r["fsd_class"]]) if r.get("fsd_class") else None
        (args.dir / name / "scores.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "parameters"} for r in rows], ensure_ascii=False, indent=1),
            encoding="utf-8")
        malo = [r for r in rows if r["set"] == "malo"]
        fsd = [r for r in rows if r["set"] == "fsd"]
        table[name] = {"malo_text": np.mean([r["text_score"] for r in malo]) if malo else float("nan"),
                       "fsd_text": np.mean([r["text_score"] for r in fsd]) if fsd else float("nan"),
                       "fsd_real": np.mean([r["real_score"] for r in fsd]) if fsd else float("nan"),
                       "similarity": np.mean([r["similarity"] for r in rows]),
                       "seconds": np.mean([r["seconds"] for r in rows])}
    ceiling = float(np.mean([spread[c] for c in classes])) if classes else float("nan")
    print(f"{'system':16s}{'malo text':>11s}{'fsd text':>10s}{'fsd real':>10s}{'own sim.':>10s}{'s/prompt':>10s}")
    for name, m in table.items():
        print(f"{name:16s}{m['malo_text']:11.3f}{m['fsd_text']:10.3f}{m['fsd_real']:10.3f}{m['similarity']:10.3f}{m['seconds']:10.1f}")
    print(f"{'real recordings':16s}{'':11s}{'':10s}{ceiling:10.3f}")
    if len(runs) >= 2:
        first, *others = list(runs)
        for other in others:
            for key, subset in (("text_score", None), ("real_score", "fsd")):
                pairs = [(a[key], b[key]) for a, b in zip(runs[first], runs[other])
                         if a[key] is not None and (subset is None or a["set"] == subset)]
                d = np.array([b - a for a, b in pairs])
                boot = np.random.default_rng(0).choice(d, size=(2000, len(d))).mean(axis=1)
                print(f"{other} - {first} on {key}: {d.mean():+.3f} (95% CI {np.percentile(boot, 2.5):+.3f} .. "
                      f"{np.percentile(boot, 97.5):+.3f}), better on {np.mean(d > 0) * 100:.0f}% of {len(d)} prompts")
    (args.dir / "summary.json").write_text(json.dumps({k: {m: float(v) for m, v in t.items()} for k, t in table.items()},
                                                      indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
