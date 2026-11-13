"""Bank retrieval scoring shared by training, export, tuning and inference (numpy only)."""

from __future__ import annotations

import numpy as np

AUDIBLE_DB_FLOOR = -40.0


def retrieval_scores(queries: np.ndarray, bank_emb: np.ndarray, hub: np.ndarray, csls_weight: float) -> np.ndarray:
    """Cosine similarity minus a CSLS penalty for presets that are close to every description."""
    return queries @ bank_emb.T - csls_weight * hub[None, :]


def top_indices(queries: np.ndarray, bank_emb: np.ndarray, hub: np.ndarray, csls_weight: float, k: int = 1, batch: int = 512) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), batch):
        scores = retrieval_scores(queries[i : i + batch], bank_emb, hub, csls_weight)
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k > 1 else scores.argmax(axis=1)[:, None]
        if k > 1:
            order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
            part = np.take_along_axis(part, order, axis=1)
        out[i : i + batch] = part
    return out
