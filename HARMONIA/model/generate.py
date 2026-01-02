import torch
import json
import argparse
import sys
from transformers import AutoTokenizer
from model import TextToParams

# --- CONFIG ---
# Must match "train.py"
PLUGIN_PARAM_COUNT = 50
MODEL_PATH = "my_plugin_ai.pth"

def generate_preset(prompt, output_filename):
    print(f"Loading brain from {MODEL_PATH}...")

    # 1. Initialize the empty model architecture
    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)

    # 2. Load the trained weight
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print("Error: Could not find trained model. Did you run train.py first?")
        sys.exit(1)

    model.eval() # Set to "test" mode (turns off dropout, etc.)

    # 3. Prepare the text
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # 4. The Thinking Part (Inference)
    print(f"Dreaming up parameters for: '{prompt}'...")
    with torch.no_grad():
        # The model spits out a list of raw numbers
        prediction = model(inputs['input_ids'], inputs['attention_mask'])

    # Get the first (and only) result from the batch
    param_list = prediction[0].tolist()

    # 5. Save the File
    # Create the structure your JUCE plugin expects
    preset_data = {
        "metadata": {
            "name": prompt,
            "generated_by": "My-AI-Model"
        },
        "parameters": param_list
    }

    with open(output_filename, 'w') as f:
        json.dump(preset_data, f, indent=4)

    print(f"Success! Preset saved to: {output_filename}")

if __name__ == "__main__":
    # Allow running from command line
    parser = argparse.ArgumentParser(description="Generate JUCE presets from text")
    parser.add_argument("prompt", type=str, help="The description of the sound")
    parser.add_argument("--output", type=str, default="generated_preset.json", help="Output filename")

    args = parser.parse_args()

    generate_preset(args.prompt, args.output)