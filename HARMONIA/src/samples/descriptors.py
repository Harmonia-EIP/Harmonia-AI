"""Timbre coordinates of the sampled instruments: how the generators write the sample source (#46).

A preset names its recorded instrument by a number (`sample_instrument`, 1..125). A generator writes
continuous values: with that number as one value it would blend unrelated instruments (choice 12 is not
"between" 11 and 13), and one value per instrument would be far too sparse for the data there is.

So each instrument gets a few coordinates that say how it sounds. It is played alone through the engine at
the generators' notes, CLAP listens (scripts/samples/instrument_descriptors.py), and the principal axes of
those sounds give its coordinates: instruments that sound alike (viola and violin) lie close together. A
generator writes coordinates; the preset gets the instrument nearest to them (src/v3/codecs.py). The preset
itself still holds an instrument number, so nothing changes for the plugin.

The table is small and lives in git (samples/descriptors.json): a generator is trained against it and only
decodes correctly with it, so it is also copied next to the generator when a model is exported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from src import charter_v3

BASE_DIR = Path(__file__).resolve().parents[2]
PATH = BASE_DIR / "samples" / "descriptors.json"
SPAN = 0.9  # the farthest instrument from the centre, so coordinates stay inside the codec's [-1, 1]


@dataclass(frozen=True)
class Descriptors:
    """Coordinates of every charter instrument choice: row 0 is "no instrument" (the centre), rows of
    instruments the table does not describe are NaN and never chosen."""

    table: np.ndarray  # (choices, dims)

    @property
    def dims(self) -> int:
        return int(self.table.shape[1])

    def coordinates(self, choice: int) -> np.ndarray:
        """The coordinates of an instrument choice (zeros for 0 or an instrument the table does not describe)."""
        if 0 < choice < len(self.table) and not np.isnan(self.table[choice, 0]):
            return self.table[choice].copy()
        return np.zeros(self.dims)

    def nearest(self, coordinates: Sequence[float]) -> int:
        """The described instrument (choice >= 1) closest to `coordinates`."""
        distance = np.linalg.norm(self.table[1:] - np.asarray(coordinates, dtype=np.float64)[None, :], axis=1)
        return 1 + int(np.nanargmin(distance))

    @classmethod
    def from_raw(cls, raw: Mapping[str, object]) -> "Descriptors":
        ids = charter_v3.load().raw["sample_instruments"]["ids"]
        coordinates: Mapping[str, Sequence[float]] = raw["instruments"]  # type: ignore[assignment]
        dims = int(raw["dims"])  # type: ignore[arg-type]
        table = np.full((len(ids) + 1, dims), np.nan)
        table[0] = 0.0
        for choice, instrument in enumerate(ids, start=1):
            if instrument in coordinates:
                table[choice] = coordinates[instrument]
        return cls(table)


@lru_cache(maxsize=4)
def load(path: Path = PATH) -> Descriptors:
    return Descriptors.from_raw(json.loads(Path(path).read_text(encoding="utf-8")))


def fit(embeddings: np.ndarray, ids: Sequence[str], dims: int) -> Dict[str, object]:
    """Coordinates from unit CLAP vectors (one per instrument): the first `dims` principal axes, one scale for
    all of them so distances keep their proportions, the farthest instrument at SPAN."""
    x = np.asarray(embeddings, dtype=np.float64)
    centred = x - x.mean(axis=0, keepdims=True)
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ axes[:dims].T
    projected *= SPAN / np.abs(projected).max()
    explained = float((singular[:dims] ** 2).sum() / (singular ** 2).sum())
    return {"dims": dims, "explained_variance": round(explained, 4),
            "instruments": {i: [round(float(v), 5) for v in row] for i, row in zip(ids, projected)}}


def neighbours_kept(embeddings: np.ndarray, coordinates: np.ndarray) -> float:
    """Share of instruments whose nearest other instrument is the same by their coordinates as by CLAP: how
    faithful the few coordinates are to what CLAP hears."""
    x = np.asarray(embeddings, dtype=np.float64)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(np.mean(similarity.argmax(axis=1) == distance.argmin(axis=1)))
