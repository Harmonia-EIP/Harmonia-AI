#!/usr/bin/env python3
"""Export the trained v2 model to the ONNX runtime layout read by src/v2/inference.py.

- text_encoder.onnx: student encoder, dynamically quantized to int8;
- predictor.onnx: parameter predictor;
- bank.npz: bank presets that are useful for retrieval (top-k neighbours of the training corpus
  and of the FSD50K recordings, plus a random share for coverage), int8 embeddings;
- tokenizer.json and manifest.json (versions, settings, sha256 of every file).

    python scripts/v2/export_v2.py --data data/v2 --out models/harmonia_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.train_predictor import LOSS_WEIGHTS, load_bank, load_priors  # noqa: E402
from src.v2.predictor import ExportablePredictor, ParamPredictor  # noqa: E402
from src.v2.retrieval import top_indices  # noqa: E402
from src.v2.text_student import MAX_TOKENS, encode_texts, load_trained_student  # noqa: E402

MODEL_VERSION = "harmonia_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_text_encoder(student_dir: Path, out_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, tokenizer = load_trained_student(student_dir, torch.device("cpu"))
    sample = tokenizer(["a door slamming in a cathedral"], return_tensors="pt")
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "text_encoder.fp32.onnx"
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        # Per-channel weights: same size, but the same top-1 preset as fp32 for 93% of prompts (80% per-tensor).
        quantize_dynamic(str(fp32), str(out_path), weight_type=QuantType.QInt8, per_channel=True)


def export_predictor(predictor_path: Path, out_path: Path) -> None:
    predictor = ParamPredictor()
    predictor.load_state_dict(torch.load(predictor_path, map_location="cpu", weights_only=True))
    torch.onnx.export(
        ExportablePredictor(predictor),
        (torch.zeros(1, 512),),
        str(out_path),
        input_names=["embedding"],
        output_names=["values"],
        dynamic_axes={"embedding": {0: "batch"}, "values": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def select_bank(data: Path, bank_size: int, top_k: int, seed: int) -> np.ndarray:
    """Indices of the audible presets to ship: neighbours of real usage first, then random coverage."""
    _, bank_emb = load_bank(data / "bank" / "bank.npz")
    usable, hub, tuned = load_priors(data)
    csls = float(tuned.get("csls_weight", 0.5))
    queries = [np.load(data / "teacher_text.npz")["emb"].astype(np.float32), np.load(data / "fsd50k_dev.npz")["emb"].astype(np.float32)]
    counts = np.zeros(len(usable), dtype=np.int64)
    for q in queries:
        np.add.at(counts, top_indices(q, bank_emb[usable], hub[usable], csls, k=top_k).ravel(), 1)
    used = np.argsort(-counts)
    used = used[counts[used] > 0][: int(bank_size * 0.85)]
    rest = np.setdiff1d(np.arange(len(usable)), used)
    fill = np.random.default_rng(seed).choice(rest, size=min(len(rest), bank_size - len(used)), replace=False)
    return np.sort(usable[np.concatenate([used, fill])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "v2")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "models" / MODEL_VERSION)
    parser.add_argument("--bank-size", type=int, default=60000)
    parser.add_argument("--bank-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-mode", choices=("retrieval", "hybrid", "neural"), default="retrieval",
                        help="mode served when the request has none (retrieval won the v2.0 benchmark)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    export_text_encoder(args.data / "text_student", args.out / "text_encoder.onnx")
    export_predictor(args.data / "predictor" / "predictor.pt", args.out / "predictor.onnx")
    shutil.copy2(args.data / "text_student" / "encoder" / "tokenizer.json", args.out / "tokenizer.json")

    bank = np.load(args.data / "bank" / "bank.npz")
    _, bank_emb = load_bank(args.data / "bank" / "bank.npz")
    _, hub, tuned = load_priors(args.data)
    keep = select_bank(args.data, args.bank_size, args.bank_top_k, args.seed)
    emb = bank_emb[keep]
    scale = float(np.abs(emb).max() / 127.0)
    np.savez_compressed(
        args.out / "bank.npz",
        emb=np.round(emb / scale).astype(np.int8),
        scale=np.float32(scale),
        params=bank["params"][keep].astype(np.float16),
        hub=hub[keep].astype(np.float16),
        source_index=keep.astype(np.int32),
    )

    # Sanity check: int8 ONNX encoder vs the PyTorch student.
    model, tokenizer = load_trained_student(args.data / "text_student", torch.device("cpu"))
    probe = ["porte qui claque dans une cathédrale", "soft piano", "basse saturée agressive", "wind howling through a cave"]
    reference = encode_texts(model, tokenizer, probe, torch.device("cpu"))
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    quantized = []
    for text in probe:
        tok = tokenizer([text], return_tensors="np")
        quantized.append(session.run(None, {"input_ids": tok["input_ids"].astype(np.int64), "attention_mask": tok["attention_mask"].astype(np.int64)})[0][0])
    agreement = float(np.min((np.array(quantized) * reference).sum(axis=1)))

    student_meta = json.loads((args.data / "text_student" / "meta.json").read_text(encoding="utf-8"))
    bank_meta = json.loads((args.data / "bank" / "bank.meta.json").read_text(encoding="utf-8"))
    files = {}
    for name in ("text_encoder.onnx", "predictor.onnx", "tokenizer.json", "bank.npz"):
        path = args.out / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "charter_version": "1.0",
        "engine": "app-1.1 (filter Q mapping + envelope release fix)",
        "max_tokens": MAX_TOKENS,
        "teacher": student_meta["teacher"],
        "student_base": student_meta["base_model"],
        "bank": {"presets": int(len(keep)), "rendered": bank_meta["n_kept"], "notes": bank_meta["notes"]},
        "retrieval": {
            "default_mode": args.default_mode,
            "top_k": 32,
            "temperature": 0.02,
            "csls_weight": float(tuned.get("csls_weight", 0.5)),
            "hybrid_weight": float(tuned.get("hybrid_weight", 0.15)),
            "param_weights": LOSS_WEIGHTS,
        },
        "int8_encoder_min_cosine_vs_fp32": round(agreement, 4),
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
