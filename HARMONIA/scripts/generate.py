import torch
import json
import argparse
import sys
import os
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import TextToParams

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
MODEL_PATH = "../saved_models/my_plugin_ai.pth"

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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"Error: Could not find model at {MODEL_PATH}")
        print("Did you run train.py first?")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

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

    with open(output_filename, 'w') as f:
        json.dump(preset_data, f, indent=4)

    print(f"Success! Preset saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JUCE presets from text")
    parser.add_argument("prompt", type=str, help="The description of the sound")
    parser.add_argument("--output", type=str, default="generated_preset.json", help="Output filename")

    args = parser.parse_args()

    generate_preset(args.prompt, args.output)