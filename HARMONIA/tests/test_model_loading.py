import json

import torch
from transformers import BertConfig, BertModel

from src.charter import PARAM_NAMES
from src.model import TextToParams, load_encoder_config, load_tokenizer

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "warm", "pad", "dark", "bass"]


def _write_legacy_encoder(path):
    """Tiny local encoder shaped like prajjwal1/bert-tiny: config.json has no `model_type`."""
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(path)
    config_path = path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("model_type", None)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    (path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return path


# A config.json without `model_type` (legacy checkpoints) must still resolve to a BERT config.
def test_load_encoder_config_handles_missing_model_type(tmp_path):
    encoder_dir = _write_legacy_encoder(tmp_path / "legacy-encoder")
    config = load_encoder_config(str(encoder_dir))
    assert config.model_type == "bert"
    assert config.hidden_size == 16


# The tokenizer of a legacy checkpoint loads and lowercases prompts like the original BERT tokenizer.
def test_load_tokenizer_handles_missing_model_type(tmp_path):
    encoder_dir = _write_legacy_encoder(tmp_path / "legacy-encoder")
    tokenizer = load_tokenizer(str(encoder_dir))
    ids = tokenizer("Warm PAD")["input_ids"]
    assert ids == [2, 5, 6, 3]


# The full charter model builds from a legacy encoder and produces 20 values in [0, 1].
def test_text_to_params_builds_from_legacy_encoder(tmp_path):
    encoder_dir = _write_legacy_encoder(tmp_path / "legacy-encoder")
    model = TextToParams(encoder_id=str(encoder_dir)).eval()
    tokenizer = load_tokenizer(str(encoder_dir))
    inputs = tokenizer("dark bass", return_tensors="pt")
    with torch.no_grad():
        out = model(inputs["input_ids"], inputs["attention_mask"])
    assert out.shape == (1, len(PARAM_NAMES))
    assert torch.all((out >= 0.0) & (out <= 1.0))
