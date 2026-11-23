"""The recordings the sample source plays: which one answers a note, and its audio (#42).

A sample pack (scripts/samples/build_packs.py) unpacks into one folder: catalog.json, CREDITS.md and one
sub-folder per library. The charter's `sample_instrument` choice i (>= 1) plays the catalogue instrument
`sample_instruments.ids[i - 1]`; 0, or an instrument whose pack is not installed, plays nothing.

The server installs the small "listen" pack (one take per note), the plugin the full packs; both read the
same catalogue format, so a preset plays the same instrument everywhere.

    bank = default_bank()           # $HARMONIA_SAMPLES_DIR, else models/samples
    region = bank.region(1, note=60, velocity=100)
    audio, rate = bank.audio(region)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import charter_v3

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = BASE_DIR / "models" / "samples"
CACHE_BYTES = 768 * 1024 * 1024  # decoded audio kept in memory


class SampleBank:
    def __init__(self, folder: Path, catalog: Optional[Path] = None, cache_bytes: int = CACHE_BYTES):
        self.folder = Path(folder)
        raw = json.loads(Path(catalog or self.folder / "catalog.json").read_text(encoding="utf-8"))
        by_id = {i["id"]: i for i in raw["instruments"]}
        ids = charter_v3.load().raw["sample_instruments"]["ids"]
        # choice index -> instrument (None when this pack does not hold it)
        self.instruments: List[Optional[dict]] = [None] + [by_id.get(i) for i in ids]
        self._attack = [None if i is None else [r for r in i["regions"] if r["trigger"] == "attack"]
                        for i in self.instruments]
        self._cache: "OrderedDict[str, Tuple[np.ndarray, int]]" = OrderedDict()
        self._cached_bytes = 0
        self.cache_bytes = cache_bytes

    def has(self, choice: int) -> bool:
        return 0 < choice < len(self.instruments) and self.instruments[choice] is not None

    def gain(self, choice: int) -> float:
        """Level that brings the instrument's loudest recording to about the oscillators' level."""
        return float(self.instruments[choice].get("gain", 1.0)) if self.has(choice) else 0.0

    def region(self, choice: int, note: int, velocity: int, take: int = 0) -> Optional[dict]:
        """The recording answering `note` at `velocity` (1..127); outside the recorded range, the nearest
        note. `take` picks among round robins (deterministic for a given take)."""
        if not self.has(choice):
            return None
        regions = self._attack[choice]
        velocity = int(min(127, max(1, velocity)))
        inside = [r for r in regions if r["lokey"] <= note <= r["hikey"]]
        if not inside:
            nearest = min({r["key"] for r in regions}, key=lambda k: (abs(k - note), k))
            inside = [r for r in regions if r["key"] == nearest]
        layer = [r for r in inside if r["lovel"] <= velocity <= r["hivel"]]
        if not layer:
            layer = [min(inside, key=lambda r: min(abs(r["lovel"] - velocity), abs(r["hivel"] - velocity)))]
        layer.sort(key=lambda r: (r["round_robin"], r["file"]))
        return layer[take % len(layer)]

    def audio(self, region: dict) -> Tuple[np.ndarray, int]:
        """(channels, frames) float32 audio of a region and its sample rate (decoded once, then cached)."""
        import soundfile as sf

        key = region["file"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data, rate = sf.read(str(self.folder / key), dtype="float32", always_2d=True)
        audio = np.ascontiguousarray(data.T)
        self._cache[key] = (audio, int(rate))
        self._cached_bytes += audio.nbytes
        while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
            _, (old, _) = self._cache.popitem(last=False)
            self._cached_bytes -= old.nbytes
        return audio, int(rate)


@lru_cache(maxsize=1)
def default_bank() -> Optional[SampleBank]:
    """The installed pack: $HARMONIA_SAMPLES_DIR, else models/samples; None when there is none (the sample
    source then plays nothing, everything else works)."""
    folder = Path(os.environ.get("HARMONIA_SAMPLES_DIR", str(DEFAULT_DIR)))
    if not (folder / "catalog.json").exists():
        return None
    return SampleBank(folder)


def instrument_ids() -> Dict[int, str]:
    """Charter choice index -> catalogue id."""
    return {i + 1: name for i, name in enumerate(charter_v3.load().raw["sample_instruments"]["ids"])}
