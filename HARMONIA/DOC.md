# 🎹 HARMONIA-AI Documentation

> **AI-Powered VST Parameter Generator** > *Turn natural language descriptions into precise synthesizer presets.*

## 📖 1. Overview

**Harmonia-AI** is a lightweight deep learning system designed to bridge the gap between human language and audio synthesis. It utilizes a BERT-based architecture to translate semantic descriptions (e.g., *"Dark Cinematic Drone"*, *"Aggressive Reese Bass"*) into floating-point parameters compatible with JUCE/C++ audio plugins.

### Key Capabilities
* **Text-to-Param Inference:** Maps semantic meaning to plugin knobs (`0.0` – `1.0`).
* **Real-time API:** A Flask server providing millisecond-latency generation for integration with VSTs.
* **Auto-Training Watchdog:** Automatically detects new dataset files and retrains the model in the background.
* **Lightweight Architecture:** Optimized to run on standard CPUs without requiring heavy GPU resources.

---

## ⚙️ 2. System Architecture

Harmonia uses a transfer learning approach leveraging a pre-trained language model.

* **Encoder:** `prajjwal1/bert-tiny` (A compressed BERT model for efficiency).
* **Parameter Head:** A custom Neural Network mapping the 128-dimension text embedding to specific synthesizer controls. ***(SUBJECT TO CHANGES)***
* **Output:** 9 distinct floating-point values (Range: `0.0` to `1.0`).

### Supported Parameters
The model is currently tuned to predict the following 9 parameters:
1.  **Frequency**
2.  **Attack**
3.  **Cutoff**
4.  **Decay**
5.  **Volume**
6.  **Sustain**
7.  **Resonance**
8.  **Release**
9.  **Waveform**

(THE NUMBER OF PARAMETERS IS MOST LIKELY SUBJECT TO CHANGES IN THE FUTURE)

---

## 📦 3. Installation

### Prerequisites
* Python 3.8+
* pip

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/harmonia-ai.git](https://github.com/your-username/harmonia-ai.git)
    cd harmonia-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r HARMONIA/requirement.txt
    ```
    *(Note: Key libraries include `torch`, `transformers`, `flask`, and `watchdog`.)*

---

## 🚀 4. Usage Workflow

### Step A: Data Preparation
Harmonia learns from `.fxp` text dumps or raw text data representing plugin states.

1.  Place your raw text dump into:
    `HARMONIA/data/raw/my_raw_dump.txt`
2.  Run the processor to format the data for training:
    ```bash
    python3 HARMONIA/scripts/prepare_dataset.py
    ```
    *Output: `HARMONIA/data/processed/presets.json`*

### Step B: Training the Model

#### Option 1: Manual Training
Run the training script once to process the current dataset.
```bash
python3 HARMONIA/scripts/train.py

```

* **Artifacts:** Saves the model to `HARMONIA/saved_models/my_plugin_ai.pth`.
* **Logs:** Updates training history in `HARMONIA/benchmarks/history.json`.

#### Option 2: Automatic Training (Watchdog) 🤖

For continuous learning, run the auto-trainer. It monitors the `drop_zone` folder.

```bash
python3 HARMONIA/scripts/auto_trainer.py
```

* **Action:** Drag & drop a new text file into `HARMONIA/data/raw/drop_zone/`.
* **Result:** The system detects the file, updates the dataset, retrains the model, and logs performance automatically.

### Step C: Performance Benchmarking

To visualize training duration, loss reduction, and model evolution:

```bash
python3 HARMONIA/scripts/benchmark_viewer.py
```

---

## 🔌 5. Integration & Generation

Harmonia offers two methods for generating presets: CLI for testing and an HTTP API for production/plugin integration.

### Method 1: HTTP API (Recommended for VSTs)

Start the background server:

```bash
python3 HARMONIA/scripts/server.py

```

*The server will listen on `http://127.0.0.1:5000/generate`.*

**API Specification:**

* **Endpoint:** `POST /generate`
* **Content-Type:** `application/json`
* **Body:**
```json
{
  "prompt": "Soft ethereal pad with long release"
}

```


* **Response:**
```json
{
  "metadata": {
    "name": "Soft ethereal pad with long release",
    "generated_by": "Harmonia-Server"
  },
  "parameters": {
    "cutoff": 0.65,
    "attack": 0.8,
    "release": 0.9,
    "resonance": 0.2,
    ...
  }
}
```



### Method 2: Command Line Interface (CLI)

Generate a single preset file directly from the terminal.

```bash
python3 HARMONIA/scripts/generate.py "Aggressive distorted bass" --output my_preset.json
```

---

## 📂 6. Project Structure

```text
HARMONIA/
├── benchmarks/          # Stores training history and stats
├── data/
│   ├── processed/       # JSON datasets ready for training
│   └── raw/             # Raw text dumps and drop_zone
├── docs/                # Documentation files
├── model/               # Location of saved .pth models
├── scripts/             # Operational scripts (train, server, generate)
└── src/
    ├── dataset.py       # Data loading logic
    └── model.py         # PyTorch model definition (BERT + Head)

```

---

**© 2026 Harmonia Project** *Built for the future of audio synthesis.*
