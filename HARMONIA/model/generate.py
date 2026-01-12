import torch
import json
import argparse
import sys
from transformers import AutoTokenizer
from model import TextToParams

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9 # MUST match train.py
MODEL_PATH = "my_plugin_ai.pth"

# Keys to map the output numbers back to names for the JUCE plugin
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

    # 1. Initialize the empty model architecture
    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)

    # 2. Load the trained weight
    try:
        # Load the weights (map_location ensures it works on CPU if needed)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Error: Could not find trained model. Did you run train.py first?")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print(f"Check if PLUGIN_PARAM_COUNT ({PLUGIN_PARAM_COUNT}) matches the trained model.")
        sys.exit(1)

    model.eval() # Set to "test" mode

    # 3. Prepare the text
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # 4. The Thinking Part (Inference)
    print(f"Dreaming up parameters for: '{prompt}'...")
    with torch.no_grad():
        prediction = model(inputs['input_ids'], inputs['attention_mask'])

    # Get the list of float numbers
    param_list = prediction[0].tolist()

    # 5. Save the File (formatted for JUCE)
    named_parameters = {}

    # Safety check: ensure we have enough parameters
    if len(param_list) != len(PARAM_KEYS):
        print(f"Warning: Model generated {len(param_list)} params, but we expected {len(PARAM_KEYS)}.")

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