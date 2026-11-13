"""Tiny, offline v2 model directory (ONNX graphs built by hand) for the inference and server tests."""

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from src.charter import PARAM_NAMES

VOCAB = {"[UNK]": 0, "door": 1, "slam": 2, "soft": 3, "piano": 4, "wind": 5}
DIM = 512
IR_VERSION = 9  # readable by onnxruntime 1.30 (newer onnx releases default to IR 14)


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _text_encoder(table):
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "input_ids"], ["tokens"]),
            helper.make_node("ReduceMean", ["tokens"], ["pooled"], axes=[1], keepdims=0),
            helper.make_node("LpNormalization", ["pooled"], ["embedding"], axis=-1, p=2),
        ],
        "text_encoder",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None]),
        ],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, DIM])],
        [numpy_helper.from_array(table.astype(np.float32), "table")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=IR_VERSION)


def _predictor(weights):
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["embedding", "w"], ["logits"]), helper.make_node("Sigmoid", ["logits"], ["values"])],
        "predictor",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, DIM])],
        [helper.make_tensor_value_info("values", TensorProto.FLOAT, [None, len(PARAM_NAMES)])],
        [numpy_helper.from_array(weights.astype(np.float32), "w")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=IR_VERSION)


def text_embedding(table, words):
    return _normalize(table[[VOCAB[w] for w in words]].mean(axis=0))


def build_model_dir(path, seed=0):
    rng = np.random.default_rng(seed)
    table = _normalize(rng.normal(size=(len(VOCAB), DIM)))
    path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    onnx.save(_text_encoder(table), str(path / "text_encoder.onnx"))
    onnx.save(_predictor(rng.normal(scale=0.1, size=(DIM, len(PARAM_NAMES)))), str(path / "predictor.onnx"))

    # Bank: preset 0 sounds like "door slam", preset 1 like "soft piano", then unrelated presets.
    emb = np.stack([text_embedding(table, ["door", "slam"]), text_embedding(table, ["soft", "piano"])] + list(_normalize(rng.normal(size=(6, DIM)))))
    scale = float(np.abs(emb).max() / 127.0)
    params = rng.uniform(size=(len(emb), len(PARAM_NAMES)))
    np.savez(path / "bank.npz", emb=np.round(emb / scale).astype(np.int8), scale=np.float32(scale), params=params.astype(np.float16))
    manifest = {
        "model_version": "harmonia_v2",
        "max_tokens": 16,
        "retrieval": {"top_k": 4, "temperature": 0.05, "hybrid_weight": 0.15, "param_weights": {}},
        "files": {},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return params
