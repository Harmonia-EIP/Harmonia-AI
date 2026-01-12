# Harmonia-AI

## Turn text descriptions into synthesizer presets.

This project uses a lightweight Artificial Intelligence model (BERT + Neural Network) to understand sound descriptions (e.g., *"A dark, distorted sci-fi bass"*) and translate them into configuration files (`.json`) that can be loaded directly into C++ JUCE audio plugin.

---

## 🚀 Features

* **Text-to-Param:** Type a description, get a preset.
* **Lightweight:** Runs on CPU. No heavy GPU required for generation.
* **Fast:** Generates parameters in milliseconds.
* **JUCE Ready:** Includes a Python server to communicate with your VST/AU plugin in real-time.

---

## 🛠️ Installation

1.  **Clone this repository** (or create your folder structure).
2.  **Install Python Dependencies:**
    You need PyTorch, Hugging Face Transformers, and Flask.

    ```bash
    pip install torch transformers flask
    ```

---

## 📂 Project Structure

```text
/Text2JUCE-AI
  ├── dataset/
  │   └── presets.json       # Your training data
  ├── model.py               # The AI Brain architecture
  ├── train.py               # Script to teach the AI
  ├── generate.py            # Command-line tool to make presets
  ├── server.py              # Server for JUCE connection
  └── README.md              # This file

```

---

## 🧠 1: Dataset

The AI learns from examples. It need a file named `dataset/presets.json`.

**Format:**

* `description`: The text prompt.
* `parameters`: A list of numbers (0.0 to 1.0) representing your plugin knobs.

**Example `presets.json`:**

```json
[
    {
        "description": "Warm analog pad with slow attack",
        "parameters": [0.5, 0.8, 0.1, 0.0, 0.9, 0.4] 
    },
    {
        "description": "Aggressive dubstep wobble bass",
        "parameters": [0.1, 0.9, 0.8, 1.0, 0.2, 0.0]
    }
]

```

* **Note:** The number of values in the `parameters` list MUST match the `PLUGIN_PARAM_COUNT` variable in the Python scripts.*

---

## 🎓 2: Training the Model

Once we have dataset, teaching the model:

```bash
python train.py
```

* This will run for the specified number of epochs.
* It saves the trained brain to `my_plugin_ai.pth`.

---

## 🎹 3: Generate Presets

### Option A: Command Line (CLI)

Great for batch generating files to drag-and-drop later.

```bash
python generate.py "A floating space drone sound" --output space_drone.json

```

### Option B: Real-time Server (For JUCE Integration)

Run the server to let the plugin "talk" to the AI.

```bash
python server.py

```

* The server listens at `http://127.0.0.1:5000/generate`.

---

## 🔌 JUCE Integration (C++)

To make this plugin request presets from the server, it can use `juce::URL` to send a POST request.

**C++ Example:**

```cpp
void AudioPluginAudioProcessor::requestPresetFromAI(juce::String userText)
{
    // 1. Prepare the JSON payload
    juce::DynamicObject* jsonBody = new juce::DynamicObject();
    jsonBody->setProperty("text", userText);
    juce::var jsonVar(jsonBody);
    
    // 2. Send Request to Python Server
    juce::URL url("[http://127.0.0.1:5000/generate](http://127.0.0.1:5000/generate)");
    juce::URL::InputStreamOptions options(juce::URL::ParameterHandling::inPostData);
    options.withExtraHeaders("Content-Type: application/json")
           .withPostData(juce::JSON::toString(jsonVar));
           
    std::unique_ptr<juce::InputStream> stream = url.createInputStream(options);
    
    // 3. Apply Parameters
    if (stream != nullptr)
    {
        var result = juce::JSON::parse(stream->readEntireStreamAsString());
        if (result.hasProperty("parameters"))
        {
            var params = result["parameters"];
            // Iterate and set your parameters here
            // myParam[0]->setValueNotifyingHost(params[0]);
        }
    }
}

```

---

## 📝 Configuration

If the plugin changes (adding / removing knobs), update the config at the top of `train.py` and `server.py`:

```python
# Change this number to match JUCE plugin parameter count!
PLUGIN_PARAM_COUNT = 8
```