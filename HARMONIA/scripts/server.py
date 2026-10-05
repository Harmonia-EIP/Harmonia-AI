import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
import os
import json

import torch
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.charter import PARAM_NAMES, charter_metadata, normalise_vector
from src.dashboard_events import publish_generation
from src.model import TextToParams, load_tokenizer
from src.artifact_registry import resolve_latest_model

app = Flask(__name__)

# --- CONFIG ---
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
LEGACY_MODEL_PATH = SAVED_MODELS_DIR / "my_plugin_ai.pth"
LEGACY_METADATA_PATH = SAVED_MODELS_DIR / "my_plugin_ai.meta.json"
BENCHMARK_FILE = BASE_DIR / "benchmarks" / "history.json"
MAX_PROMPT_LENGTH = 512
TOKENIZER_MAX_LENGTH = 32
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
PARAM_KEYS = list(PARAM_NAMES)
PLUGIN_PARAM_COUNT = len(PARAM_KEYS)

DEFAULT_MODEL_KEY = "default"
V2_MODEL_KEY = "harmonia_v2"
V2_MODEL_DIR = Path(os.environ.get("HARMONIA_V2_MODEL_DIR", str(BASE_DIR / "models" / V2_MODEL_KEY)))
V2_MODES = ("retrieval", "hybrid", "neural")

# Named models selectable from the app's model switcher, in addition to the
# default model resolved via resolve_latest_model().
EXPLICIT_MODELS = {
    "charter_v1": {
        "model_path": SAVED_MODELS_DIR / "model_charter_v1.pth",
        "metadata_path": SAVED_MODELS_DIR / "model_charter_v1.meta.json",
    },
}

# Maps the model_id/model_name/model values the app can send to a key in
# EXPLICIT_MODELS, or DEFAULT_MODEL_KEY.
MODEL_KEY_ALIASES = {
    "model-1": DEFAULT_MODEL_KEY,
    "1": DEFAULT_MODEL_KEY,
    "model-2": "charter_v1",
    "charter_v1": "charter_v1",
    "2": "charter_v1",
    "model-3": V2_MODEL_KEY,
    "harmonia_v2": V2_MODEL_KEY,
    "v2": V2_MODEL_KEY,
    "3": V2_MODEL_KEY,
}


def _resolve_model_key(data: dict) -> str:
    raw = data.get("model_name") or data.get("model") or data.get("model_id")
    if raw is None:
        return DEFAULT_MODEL_KEY
    return MODEL_KEY_ALIASES.get(str(raw).strip().lower(), DEFAULT_MODEL_KEY)

@dataclass
class InferenceRuntime:
    model: Optional[TextToParams]
    tokenizer: Optional[object]
    ready: bool
    error: str = ""
    model_version: str = "unknown"
    model_hash: str = "unknown"
    model_path: str = ""
    model_metadata_path: str = ""
    param_keys: Tuple[str, ...] = tuple(PARAM_KEYS)
    tokenizer_max_length: int = TOKENIZER_MAX_LENGTH
    v2: Optional[object] = None


def _load_json_file(path):
    if path is None or not Path(path).exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def _build_v2_runtime() -> InferenceRuntime:
    manifest_path = V2_MODEL_DIR / "manifest.json"
    manifest = _load_json_file(manifest_path)
    if not manifest:
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error="v2 model files not found. Run scripts/v2/fetch_model.py.",
            model_version=V2_MODEL_KEY,
            model_path=str(V2_MODEL_DIR),
        )
    model_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        from src.v2.inference import HarmoniaV2

        engine = HarmoniaV2(V2_MODEL_DIR)
    except Exception as exc:  # missing runtime dependency or corrupt files
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error=f"Invalid v2 model: {exc}",
            model_version=str(manifest.get("model_version", V2_MODEL_KEY)),
            model_hash=model_hash,
            model_path=str(V2_MODEL_DIR),
        )
    return InferenceRuntime(
        model=None,
        tokenizer=None,
        ready=True,
        model_version=str(manifest.get("model_version", V2_MODEL_KEY)),
        model_hash=model_hash,
        model_path=str(V2_MODEL_DIR),
        model_metadata_path=str(manifest_path),
        tokenizer_max_length=engine.max_tokens,
        v2=engine,
    )


def _build_runtime(model_key: str = DEFAULT_MODEL_KEY) -> InferenceRuntime:
    if model_key == V2_MODEL_KEY:
        return _build_v2_runtime()
    if model_key == DEFAULT_MODEL_KEY:
        artifact = resolve_latest_model(
            SAVED_MODELS_DIR,
            legacy_model_path=LEGACY_MODEL_PATH,
            legacy_metadata_path=LEGACY_METADATA_PATH,
        )
    else:
        artifact = EXPLICIT_MODELS[model_key]

    model_path = artifact.get("model_path")
    metadata_path = artifact.get("metadata_path")
    if not model_path:
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error="Model file not found.",
            model_version="unknown",
            model_hash="unknown",
        )

    print(f"Loading AI model from {model_path}...")
    device = torch.device("cpu")
    metadata = _load_json_file(metadata_path)
    model_version = str(metadata.get("model_version", artifact.get("model_version", "unknown")))
    model_hash = str(metadata.get("model_hash", artifact.get("model_hash", "unknown")))
    metadata_param_keys = metadata.get("param_keys")
    runtime_param_keys = PARAM_KEYS
    if isinstance(metadata_param_keys, list) and metadata_param_keys:
        runtime_param_keys = [str(k) for k in metadata_param_keys]
    plugin_param_count = int(metadata.get("plugin_param_count", len(runtime_param_keys)))
    tokenizer_max_length = int(metadata.get("tokenizer_max_length", TOKENIZER_MAX_LENGTH))
    model = TextToParams(num_plugin_parameters=plugin_param_count)

    try:
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Unsafe model loading blocked: this PyTorch version does not support weights_only=True. "
                "Please upgrade PyTorch."
            ) from exc
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
            model_path=str(model_path),
            model_metadata_path=str(metadata_path) if metadata_path else "",
            param_keys=tuple(runtime_param_keys),
            tokenizer_max_length=tokenizer_max_length,
        )
    except RuntimeError as exc:
        return InferenceRuntime(
            model=None,
            tokenizer=None,
            ready=False,
            error=f"Invalid model weights: {exc}",
            model_version=model_version,
            model_hash=model_hash,
            model_path=str(model_path),
            model_metadata_path=str(metadata_path) if metadata_path else "",
            param_keys=tuple(runtime_param_keys),
            tokenizer_max_length=tokenizer_max_length,
        )

    tokenizer = load_tokenizer(TOKENIZER_MODEL_ID, TOKENIZER_MODEL_REVISION)
    return InferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        ready=True,
        model_version=model_version,
        model_hash=model_hash,
        model_path=str(model_path),
        model_metadata_path=str(metadata_path) if metadata_path else "",
        param_keys=tuple(runtime_param_keys),
        tokenizer_max_length=tokenizer_max_length,
    )


@lru_cache(maxsize=None)
def _get_runtime(model_key: str = DEFAULT_MODEL_KEY) -> InferenceRuntime:
    return _build_runtime(model_key)


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
            "model_path": runtime.model_path,
            "model_metadata_path": runtime.model_metadata_path,
            "model_version": runtime.model_version,
            "model_hash": runtime.model_hash,
            "plugin_param_count": len(runtime.param_keys),
            "tokenizer_max_length": runtime.tokenizer_max_length,
            "error": runtime.error,
        }
    )


@app.route("/metrics/latest", methods=["GET"])
def latest_metrics():
    if not BENCHMARK_FILE.exists():
        return jsonify({"error": "No benchmark history found."}), 404

    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Benchmark history is unreadable."}), 500

    if not isinstance(history, list) or not history:
        return jsonify({"error": "No benchmark runs available."}), 404

    latest = history[-1]
    report_payload = {}
    report_path = latest.get("evaluation_report_path")
    if report_path:
        report_payload = _load_json_file(report_path)

    return jsonify(
        {
            "status": "ok",
            "latest_benchmark": latest,
            "latest_evaluation_report": report_payload,
        }
    )

@app.route('/generate', methods=['POST'])
@app.route('/ai/generate-preset', methods=['POST'], endpoint='generate_preset_alias')
def generate():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object."}), 400
    prompt, validation_error = _validate_prompt(data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    runtime = _get_runtime(_resolve_model_key(data))
    if not runtime.ready:
        return jsonify({"error": "Model unavailable. Train and save model first."}), 503

    print(f"Received request for: '{prompt}'")

    generation_info = None
    if getattr(runtime, "v2", None) is not None:
        mode = data.get("mode")
        variation = data.get("variation")
        if mode is not None and mode not in V2_MODES:
            return jsonify({"error": f"'mode' must be one of {', '.join(V2_MODES)}."}), 400
        if variation is not None and (not isinstance(variation, int) or isinstance(variation, bool)):
            return jsonify({"error": "'variation' must be an integer."}), 400
        result = runtime.v2.generate(prompt, mode=mode, variation=variation)
        param_list = result.values
        generation_info = {"mode": result.mode, "bank_index": result.bank_index, "similarity": result.similarity}
    else:
        param_list, error_response = _predict_v1(runtime, prompt)
        if error_response is not None:
            return error_response

    is_charter = tuple(runtime.param_keys) == tuple(PARAM_NAMES)
    if is_charter:
        param_list = normalise_vector(param_list)

    named_parameters = {}
    for i, key in enumerate(runtime.param_keys):
        if i < len(param_list):
            value = float(param_list[i])
            named_parameters[key] = round(min(1.0, max(0.0, value)), 6)

    response = {
        "metadata": {
            "name": prompt,
            "generated_by": "Harmonia-Server",
            "model_version": runtime.model_version,
            "model_hash": runtime.model_hash,
            "charter_version": "1.0" if is_charter else None,
        },
        "parameters": named_parameters,
    }

    if is_charter:
        response["values"] = [named_parameters[name] for name in PARAM_NAMES]
        response["charter"] = charter_metadata()
    if generation_info is not None:
        response["generation"] = generation_info

    try:
        publish_generation(
            prompt,
            named_parameters,
            response.get("values"),
            model_version=runtime.model_version,
            model_hash=runtime.model_hash,
            charter_version="1.0" if is_charter else None,
            source="http",
        )
    except Exception as exc:  # pragma: no cover - best effort
        app.logger.warning("dashboard publish failed: %s", exc)

    return jsonify(response)


def _predict_v1(runtime: InferenceRuntime, prompt: str):
    """Returns (parameter list, None) or (None, error response) for the torch v1 models."""
    tokenized = runtime.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    token_count = int(tokenized["input_ids"].shape[1])
    if token_count > runtime.tokenizer_max_length:
        return None, (
            jsonify(
                {
                    "error": (
                        "Prompt exceeds model token context. "
                        f"Got {token_count} tokens, max {runtime.tokenizer_max_length}."
                    )
                }
            ),
            400,
        )

    inputs = runtime.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=runtime.tokenizer_max_length,
    )

    with torch.no_grad():
        prediction = runtime.model(inputs["input_ids"], inputs["attention_mask"])
    return prediction[0].tolist(), None


@app.route("/charter", methods=["GET"])
def charter():
    return jsonify({"version": "1.0", "parameters": charter_metadata()})

if __name__ == '__main__':
    print("Server is running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
