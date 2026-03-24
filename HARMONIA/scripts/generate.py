import torch
import json
import argparse
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.model import TextToParams
from src.artifact_registry import resolve_latest_model

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
LEGACY_MODEL_PATH = SAVED_MODELS_DIR / "my_plugin_ai.pth"
LEGACY_METADATA_PATH = SAVED_MODELS_DIR / "my_plugin_ai.meta.json"
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
TOKENIZER_MAX_LENGTH = 32

PARAM_KEYS = [
    "frequency",
    "attack",
    "cutoff",
    "decay",
    "volume",
    "sustain",
    "resonance",
    "release",
    "waveform"
]

def generate_preset(prompt, output_filename):
    artifact = resolve_latest_model(
        SAVED_MODELS_DIR,
        legacy_model_path=LEGACY_MODEL_PATH,
        legacy_metadata_path=LEGACY_METADATA_PATH,
    )
    model_path = artifact.get("model_path")
    if not model_path:
        print("Error: Could not find a trained model in saved_models/.")
        print("Did you run train.py first?")
        sys.exit(1)

    print(f"Loading brain from {model_path}...")

    metadata_payload = {}
    metadata_path = artifact.get("metadata_path")
    if metadata_path and Path(metadata_path).exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                metadata_payload = loaded
        except (json.JSONDecodeError, OSError):
            metadata_payload = {}

    runtime_param_keys = metadata_payload.get("param_keys", PARAM_KEYS)
    if not isinstance(runtime_param_keys, list) or not runtime_param_keys:
        runtime_param_keys = PARAM_KEYS
    runtime_param_keys = [str(k) for k in runtime_param_keys]
    plugin_param_count = int(metadata_payload.get("plugin_param_count", len(runtime_param_keys)))
    tokenizer_max_length = int(metadata_payload.get("tokenizer_max_length", TOKENIZER_MAX_LENGTH))

    model = TextToParams(num_plugin_parameters=plugin_param_count)

    try:
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Unsafe model loading blocked: this PyTorch version does not support weights_only=True. "
                "Please upgrade PyTorch."
            ) from exc
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}")
        print("Did you run train.py first?")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615
    tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    token_count = int(tokens["input_ids"].shape[1])
    if token_count > tokenizer_max_length:
        print(
            "Error: Prompt exceeds model token context "
            f"({token_count} > {tokenizer_max_length})."
        )
        sys.exit(1)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer_max_length,
    )

    print(f"Dreaming up parameters for: '{prompt}'...")
    with torch.no_grad():
        prediction = model(inputs['input_ids'], inputs['attention_mask'])

    param_list = prediction[0].tolist()

    named_parameters = {}
    if len(param_list) != len(runtime_param_keys):
        print(f"Warning: Model generated {len(param_list)} params, expected {len(runtime_param_keys)}.")

    for i, key in enumerate(runtime_param_keys):
        if i < len(param_list):
            named_parameters[key] = param_list[i]

    preset_data = {
        "metadata": {
            "name": prompt,
            "generated_by": "Harmonia-AI",
            "model_version": metadata_payload.get("model_version", artifact.get("model_version", "unknown")),
            "model_hash": metadata_payload.get("model_hash", "unknown"),
        },
        "parameters": named_parameters
    }

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preset_data, f, indent=4)

    print(f"Success! Preset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JUCE presets from text")
    parser.add_argument("prompt", type=str, help="The description of the sound")
    parser.add_argument("--output", type=str, default="generated_preset.json", help="Output filename")

    args = parser.parse_args()

    generate_preset(args.prompt, args.output)