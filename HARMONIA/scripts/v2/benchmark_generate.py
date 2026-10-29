#!/usr/bin/env python3
"""Benchmark step 1: generate a preset for every benchmark prompt (English and French) with every system.

Systems: uniform random, the two v1 models (synthetic_v1, charter_v1), v2 in its three modes (from the
exported ONNX model), and an oracle that retrieves with the CLAP teacher itself (English only, upper
bound for the student). Output: data/v2/bench/generations.json, scored by benchmark_judge.py.

    python scripts/v2/benchmark_generate.py --model models/harmonia_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import PARAM_NAMES, normalise_vector  # noqa: E402
from src.synth.engine import ENGINE_APP_1_0, ENGINE_APP_1_1  # noqa: E402
from src.v2.benchmark_set import load_prompts  # noqa: E402
from src.v2.inference import HarmoniaV2  # noqa: E402


def prompt_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def v1_generator(model_key: str):
    import torch
    from scripts import server

    runtime = server._get_runtime(model_key)
    if not runtime.ready:
        raise RuntimeError(f"v1 model {model_key} unavailable: {runtime.error}")

    def generate(prompt: str):
        inputs = runtime.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=runtime.tokenizer_max_length)
        with torch.no_grad():
            values = runtime.model(inputs["input_ids"], inputs["attention_mask"])[0].tolist()
        return normalise_vector(values)

    return generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "models" / "harmonia_v2")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "v2")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "bench" / "generations.json")
    args = parser.parse_args()

    prompts = load_prompts()
    v2 = HarmoniaV2(args.model)
    systems = {
        "random": lambda p: normalise_vector(list(np.random.default_rng(prompt_seed(p)).uniform(size=len(PARAM_NAMES)))),
        "v1_synthetic": v1_generator("default"),
        "v1_charter": v1_generator("charter_v1"),
        "v2_retrieval": lambda p: v2.generate(p, mode="retrieval").values,
        "v2_neural": lambda p: v2.generate(p, mode="neural").values,
        "v2_hybrid": lambda p: v2.generate(p, mode="hybrid").values,
    }
    # (system, engine the preset is played with). v1 is also played on the current app engine (1.0).
    plan = [("random", ENGINE_APP_1_1), ("v1_synthetic", ENGINE_APP_1_0), ("v1_synthetic", ENGINE_APP_1_1),
            ("v1_charter", ENGINE_APP_1_0), ("v1_charter", ENGINE_APP_1_1), ("v2_retrieval", ENGINE_APP_1_1),
            ("v2_neural", ENGINE_APP_1_1), ("v2_hybrid", ENGINE_APP_1_1)]

    results = []
    for system, engine in plan:
        for lang in ("en", "fr"):
            for p in prompts:
                results.append({"system": system, "engine": engine, "lang": lang, "prompt_en": p["en"], "prompt": p[lang],
                                "category": p["category"], "fsd50k": p["fsd50k"], "values": systems[system](p[lang])})

    # Oracle: CLAP teacher text embedding -> full bank retrieval (English prompts only).
    from scripts.v2.train_predictor import load_bank
    from src.v2.clap_audio import ClapEmbedder

    bank_params, bank_emb = load_bank(args.data / "bank" / "bank.npz")
    teacher = ClapEmbedder().embed_text([p["en"] for p in prompts])
    for p, t in zip(prompts, teacher):
        best = int(np.argmax(bank_emb @ t))
        results.append({"system": "oracle_teacher_fullbank", "engine": ENGINE_APP_1_1, "lang": "en", "prompt_en": p["en"],
                        "prompt": p["en"], "category": p["category"], "fsd50k": p["fsd50k"],
                        "values": normalise_vector(bank_params[best].astype(float).tolist())})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(results)} generations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
