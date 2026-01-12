import torch
import json
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from model import TextToParams

app = Flask(__name__)

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
MODEL_PATH = "my_plugin_ai.pth"
PARAM_KEYS = [
    "frequency", "attack", "cutoff", "decay",
    "volume", "sustain", "resonance", "release",
    "waveform"
]

# --- LOAD BRAIN ONCE (AT STARTUP) ---
print(f"Loading AI model from {MODEL_PATH}...")
device = torch.device('cpu') # Force CPU for safety
model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set to test mode
    print("Model loaded successfully!")
except FileNotFoundError:
    print("CRITICAL ERROR: Model file not found. Train it first!")

# Pre-load tokenizer to avoid timeouts during generation
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

@app.route('/generate', methods=['POST'])
def generate():
    """
    Endpoint that accepts JSON: {"prompt": "Soft Piano"}
    Returns JSON: {"parameters": { ... }}
    """
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    print(f"Received request for: '{prompt}'")

    # 1. Prepare text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # 2. Predict
    with torch.no_grad():
        prediction = model(inputs['input_ids'], inputs['attention_mask'])

    # 3. Format output
    param_list = prediction[0].tolist()

    named_parameters = {}
    for i, key in enumerate(PARAM_KEYS):
        if i < len(param_list):
            named_parameters[key] = param_list[i]

    # Return the clean JSON to the app
    response = {
        "metadata": {
            "name": prompt,
            "generated_by": "Harmonia-Server"
        },
        "parameters": named_parameters
    }

    return jsonify(response)

if __name__ == '__main__':
    # Run the server on localhost port 5000
    print("Server is running on http://127.0.0.1:5000")
    app.run(port=5000)