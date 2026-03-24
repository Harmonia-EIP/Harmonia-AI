import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
import os
import json

import torch
from flask import Flask, jsonify, request
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.model import TextToParams

app = Flask(__name__)

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
MODEL_PATH = BASE_DIR / "saved_models" / "my_plugin_ai.pth"
MODEL_METADATA_PATH = BASE_DIR / "saved_models" / "my_plugin_ai.meta.json"
MAX_PROMPT_LENGTH = 512
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
PARAM_KEYS = [
    "frequency", "attack", "cutoff", "decay",
    "volume", "sustain", "resonance", "release",
    "waveform"
]

@dataclass
class InferenceRuntime:
    model: Optional[TextToParams]
    tokenizer: Optional[object]
    ready: bool
    error: str = ""
    model_version: str = "unknown"
    model_hash: str = "unknown"


def _load_model_metadata():
    if not MODEL_METADATA_PATH.exists():
        return {}

    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def _build_runtime() -> InferenceRuntime:
    print(f"Loading AI model from {MODEL_PATH}...")
    device = torch.device("cpu")
    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)
    metadata = _load_model_metadata()
    model_version = str(metadata.get("model_version", "unknown"))
    model_hash = str(metadata.get("model_hash", "unknown"))

    try:
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            # Compatibility fallback for older torch versions lacking weights_only.
            state_dict = torch.load(MODEL_PATH, map_location=device)  # nosec B614
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error="Model file not found.",
            model_version=model_version,
            model_hash=model_hash,
        )
    except RuntimeError as exc:
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error=f"Invalid model weights: {exc}",
            model_version=model_version,
            model_hash=model_hash,
        )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615
    return InferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        ready=True,
        model_version=model_version,
        model_hash=model_hash,
    )


@lru_cache(maxsize=1)
def _get_runtime() -> InferenceRuntime:
    return _build_runtime()


def _validate_prompt(payload: dict) -> Tuple[Optional[str], Optional[str]]:
    prompt = payload.get("prompt")

    if not isinstance(prompt, str):
        return None, "'prompt' must be a string."

    prompt = prompt.strip()
    if not prompt:
        return None, "No prompt provided."

    if len(prompt) > MAX_PROMPT_LENGTH:
        return None, f"'prompt' is too long (max {MAX_PROMPT_LENGTH} chars)."

    return prompt, None


@app.route("/health", methods=["GET"])
def health():
    runtime = _get_runtime()
    status = "ok" if runtime.ready else "degraded"
    return jsonify(
        {
            "status": status,
            "model_ready": runtime.ready,
            "model_path": str(MODEL_PATH),
            "model_metadata_path": str(MODEL_METADATA_PATH),
            "model_version": runtime.model_version,
            "model_hash": runtime.model_hash,
            "error": runtime.error,
        }
    )

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object."}), 400
    prompt, validation_error = _validate_prompt(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    runtime = _get_runtime()
    if not runtime.ready:
        return jsonify({"error": "Model unavailable. Train and save model first."}), 503

    print(f"Received request for: '{prompt}'")

    # 1. Prepare text
    inputs = runtime.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=32)

    # 2. Predict
    with torch.no_grad():
        prediction = runtime.model(inputs['input_ids'], inputs['attention_mask'])

    # 3. Formatd output
    param_list = prediction[0].tolist()

    named_parameters = {}
    for i, key in enumerate(PARAM_KEYS):
        if i < len(param_list):
            value = float(param_list[i])
            named_parameters[key] = round(min(1.0, max(0.0, value)), 6)

    # Return the clean JSON to the app
    response = {
        "metadata": {
            "name": prompt,
            "generated_by": "Harmonia-Server",
            "model_version": runtime.model_version,
            "model_hash": runtime.model_hash,
        },
        "parameters": named_parameters
    }

    return jsonify(response)

if __name__ == '__main__':
    print("Server is running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
