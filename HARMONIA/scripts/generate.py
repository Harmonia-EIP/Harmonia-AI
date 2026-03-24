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

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
MODEL_PATH = BASE_DIR / "saved_models" / "my_plugin_ai.pth"
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")

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
    print(f"Loading brain from {MODEL_PATH}...")

    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)

    try:
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        except TypeError:
            # Compatibility fallback for older torch versions lacking weights_only.
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))  # nosec
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Could not find model at {MODEL_PATH}")
        print("Did you run train.py first?")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=32)

    print(f"Dreaming up parameters for: '{prompt}'...")
    with torch.no_grad():
        prediction = model(inputs['input_ids'], inputs['attention_mask'])

    param_list = prediction[0].tolist()

    named_parameters = {}
    if len(param_list) != len(PARAM_KEYS):
        print(f"Warning: Model generated {len(param_list)} params, expected {len(PARAM_KEYS)}.")

    for i, key in enumerate(PARAM_KEYS):
        if i < len(param_list):
            named_parameters[key] = param_list[i]

    preset_data = {
        "metadata": {
            "name": prompt,
            "generated_by": "Harmonia-AI"
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