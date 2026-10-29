"""Benchmark prompts (scripts/v2/benchmark_prompts.json), kept out of every training set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "benchmark_prompts.json"


def load_prompts() -> List[Dict[str, str]]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def held_out_sentences() -> Set[str]:
    return {p[lang].strip().lower() for p in load_prompts() for lang in ("en", "fr")}
