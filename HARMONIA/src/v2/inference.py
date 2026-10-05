"""Harmonia v2 runtime: prompt -> charter preset with ONNX Runtime (no torch / transformers).

Model directory layout (written by scripts/v2/export_v2.py):
    manifest.json        versions, sizes, sha256 of every file, retrieval settings
    tokenizer.json       multilingual tokenizer (Hugging Face `tokenizers` format)
    text_encoder.onnx    student text encoder -> L2-normalized 512-d CLAP-space embedding
    predictor.onnx       embedding -> 20 normalized charter values
    bank.npz             preset bank: int8 audio embeddings (+ scale) and float16 parameters
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.charter import PARAM_NAMES, normalise_vector

MODES = ("hybrid", "retrieval", "neural")
DEFAULT_MODE = "hybrid"


@dataclass
class Generation:
    values: List[float]
    mode: str
    bank_index: Optional[int]
    similarity: Optional[float]


class HarmoniaV2:
    def __init__(self, model_dir: Path, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.max_tokens = int(self.manifest.get("max_tokens", 64))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_tokens)
        self.tokenizer.no_padding()

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.text_session = ort.InferenceSession(str(self.model_dir / "text_encoder.onnx"), options, providers=providers)
        self.predictor_session = ort.InferenceSession(str(self.model_dir / "predictor.onnx"), options, providers=providers)

        bank = np.load(self.model_dir / "bank.npz")
        # Stored as int8 (emb ≈ int8 * scale) to keep the download small; dequantized once here.
        self.bank_emb = bank["emb"].astype(np.float32) * float(bank["scale"])
        self.bank_params = bank["params"].astype(np.float32)
        self.bank_hub = bank["hub"].astype(np.float32) if "hub" in bank.files else np.zeros(len(self.bank_params), dtype=np.float32)
        retrieval = self.manifest.get("retrieval", {})
        self.top_k = int(retrieval.get("top_k", 32))
        self.temperature = float(retrieval.get("temperature", 0.02))
        self.hybrid_weight = float(retrieval.get("hybrid_weight", 0.15))
        self.csls_weight = float(retrieval.get("csls_weight", 0.0))
        weights = retrieval.get("param_weights", {})
        self.param_weights = np.array([float(weights.get(n, 1.0)) for n in PARAM_NAMES], dtype=np.float32)

    def embed(self, prompt: str) -> np.ndarray:
        enc = self.tokenizer.encode(prompt)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        return self.text_session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]

    def _candidates(self, emb: np.ndarray):
        sims = self.bank_emb @ emb
        scores = sims - self.csls_weight * self.bank_hub  # CSLS: penalize presets close to everything
        k = min(self.top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, sims[top], scores[top]

    def generate(self, prompt: str, mode: str = DEFAULT_MODE, variation: Optional[int] = None) -> Generation:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        emb = self.embed(prompt).astype(np.float32)
        neural = None
        if mode != "retrieval":
            neural = self.predictor_session.run(None, {"embedding": emb[None, :]})[0][0]
        if mode == "neural":
            return Generation(normalise_vector(neural.tolist()), mode, None, None)

        top, sims, score = self._candidates(emb)
        score = score.astype(np.float64)
        if mode == "hybrid":
            dist = np.sqrt((((self.bank_params[top] - neural) ** 2) * self.param_weights).mean(axis=1))
            score = score - self.hybrid_weight * dist
        if variation is None or int(variation) == 0:
            pick = int(np.argmax(score))
        else:
            # Variations: sample among the best candidates, sharper for higher scores.
            rng = np.random.default_rng(int(variation) % 2**32)
            p = np.exp((score - score.max()) / self.temperature)
            pick = int(rng.choice(len(top), p=p / p.sum()))
        values = self.bank_params[top[pick]].astype(np.float64).tolist()
        return Generation(normalise_vector(values), mode, int(top[pick]), float(sims[pick]))
