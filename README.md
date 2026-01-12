# HARMONIA 🎹

> **AI-Powered VST Parameter Generator**

Harmonia is a deep learning system designed to control JUCE audio plugins.  
It translates natural language descriptions (e.g. *"Soft Piano"*, *"Aggressive Bass"*) into precise floating-point parameters (`0.0 – 1.0`) for synthesizers and effects.

---

## 🚀 Features

- **Text-to-Param Inference**  
  Uses a BERT-based architecture to map semantic meaning to plugin knobs.

- **Real-time API**  
  A Flask server to communicate directly with C++ / JUCE plugins.

- **Auto-Training Watchdog**  
  Automatically processes new data and retrains the model when files are dropped into a folder.

- **Performance Benchmarking**  
  Tracks loss, training time, and improvements over time.

- **9-Parameter Support**  
  Currently tuned for:
    - Frequency
    - Attack
    - Cutoff
    - Decay
    - Volume
    - Sustain
    - Resonance
    - Release
    - Waveform

---

## 📦 Installation

1. **Clone the repository**
2. **Install dependencies**

```bash
pip install torch transformers flask watchdog
```

---

## 🛠 Usage

### 1. Data Preparation

You can train the model using raw `.fxp` text dumps from plugins (e.g. Sylenth1).

1. Place your raw dump text into:
   ```
   dataset/my_raw_dump.txt
   ```
2. Run the converter:
   ```bash
   python3 dataset/prepare_dataset.py
   ```

This creates:
```
dataset/presets.json
```

---

### 2. Training the Model (Manual)

To train the AI on your current dataset:

```bash
python3 model/train.py
```

- Saves the model to:
  ```
  model/my_plugin_ai.pth
  ```
- Logs performance metrics to:
  ```
  benchmarks/
  ```

---

### 3. Auto-Training (Automatic) 🤖

For a continuous workflow, run the auto trainer.  
It watches the `drop_zone/` folder for new data.

```bash
python3 auto_trainer.py
```

**Action**  
Drag & drop a `.txt` file with new presets into:
```
drop_zone/
```

**Result**
- Dataset is updated automatically
- Model is retrained
- New benchmark results are displayed

---

### 4. Benchmarking 📊

To visualize the evolution of your model’s performance  
(training time, final loss, etc.):

```bash
python3 benchmark_viewer.py
```

---

## 🔌 Integration (JUCE / C++)

### Method A: HTTP API (Recommended)

Run the dedicated API server:

```bash
python3 model/server.py
```

The server listens on:
```
http://127.0.0.1:5000/generate
```

#### Request (JSON)

```json
POST /generate
{
  "prompt": "Dark Reese Bass"
}
```

#### Response (JSON)

```json
{
  "parameters": {
    "cutoff": 0.45,
    "resonance": 0.8,
    "attack": 0.1
  }
}
```

---

### Method B: CLI Generation

Generate presets directly from the command line:

```bash
python3 model/generate.py "Soft Piano" --output my_preset.json
```
