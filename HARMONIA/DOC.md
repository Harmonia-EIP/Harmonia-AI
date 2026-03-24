# HARMONIA Technical Documentation

Harmonia is a text-to-parameter AI system for VST/JUCE integration.

It converts natural-language prompts into normalized synthesizer parameters in the range `[0.0, 1.0]`.

## 1. Architecture

- Encoder: `prajjwal1/bert-tiny` from Hugging Face (`transformers`)
- Regressor head: small MLP (`Linear -> ReLU -> Dropout -> Linear -> Sigmoid`)
- Output: 9 parameters

Predicted parameter keys:
1. `frequency`
2. `attack`
3. `cutoff`
4. `decay`
5. `volume`
6. `sustain`
7. `resonance`
8. `release`
9. `waveform`

## 2. Runtime components

- `scripts/prepare_dataset.py`: parse raw dumps and build `data/processed/presets.json`
- `scripts/train.py`: train, save model, produce benchmark history and validation metrics report
- `scripts/generate.py`: CLI inference
- `scripts/server.py`: Flask API (`POST /generate`, `GET /health`)
- `scripts/auto_trainer.py`: filesystem watcher for continuous retraining
- `scripts/benchmark_viewer.py`: print training history evolution

## 3. Installation

Requirements:
- Python 3.9+
- pip

From repository root:

```bash
cd HARMONIA
python3 -m pip install -r requirement.txt
```

## 4. End-to-end workflow

Run from repository root (`/path/to/Harmonia-AI`):

### A. Dataset preparation

```bash
python3 HARMONIA/scripts/prepare_dataset.py
```

Input: `HARMONIA/data/raw/my_raw_dump.txt`  
Output: `HARMONIA/data/processed/presets.json`

### B. Model training

```bash
python3 HARMONIA/scripts/train.py
```

Deterministic seed (optional):

```bash
HARMONIA_SEED=123 python3 HARMONIA/scripts/train.py
```

Pinned model/tokenizer source (optional):

```bash
HARMONIA_MODEL_ID=prajjwal1/bert-tiny
HARMONIA_MODEL_REVISION=main
python3 HARMONIA/scripts/train.py
```

Outputs:
- `HARMONIA/saved_models/my_plugin_ai.pth`
- `HARMONIA/saved_models/my_plugin_ai.meta.json`
- `HARMONIA/benchmarks/history.json`
- `HARMONIA/benchmarks/reports/eval_*.json`

By default training uses a deterministic `train/validation` split (`HARMONIA_VAL_SPLIT=0.2`) and stores MAE/MSE per parameter in the evaluation report.

### C. CLI generation

```bash
python3 HARMONIA/scripts/generate.py "Aggressive distorted bass" --output HARMONIA/data/processed/my_preset.json
```

### D. HTTP API

```bash
python3 HARMONIA/scripts/server.py
```

Endpoint:

```text
POST http://127.0.0.1:5000/generate
GET  http://127.0.0.1:5000/health
```

Request:

```json
{
  "prompt": "Soft ethereal pad with long release"
}
```

Validation on `POST /generate`:
- body must be valid JSON
- `prompt` must be a non-empty string
- max prompt length: `512`
- if model weights are unavailable/invalid, API returns `503`

API traceability:
- `GET /health` returns `model_version` and `model_hash`
- `POST /generate` includes `model_version` and `model_hash` in `metadata`

Response:

```json
{
  "metadata": {
    "name": "Soft ethereal pad with long release",
    "generated_by": "Harmonia-Server"
  },
  "parameters": {
    "frequency": 0.44,
    "attack": 0.81,
    "cutoff": 0.65,
    "decay": 0.33,
    "volume": 0.72,
    "sustain": 0.56,
    "resonance": 0.20,
    "release": 0.90,
    "waveform": 0.18
  }
}
```

### E. Auto-training

```bash
python3 HARMONIA/scripts/auto_trainer.py
```

Drop new `.txt` files into `HARMONIA/data/raw/drop_zone/`.

### F. Benchmark visualization

```bash
python3 HARMONIA/scripts/benchmark_viewer.py
```

## 5. Known limitations

- Data scarcity: model quality is currently dataset-limited.
- Fixed output space: only 9 parameters are supported.
- No external model registry service yet (current metadata storage is local-file based).
- Flask development server is suitable for local integration, not production.
- Dependency manifest (`requirement.txt`) is broad and includes non-portable entries.

## 6. Next technical challenges

- Build a validated dataset pipeline with schema checks and outlier filtering.
- Add proper evaluation (MAE/MSE by parameter, holdout set, reproducibility seed).
- Introduce model versioning and rollback strategy for plugin integration safety.
- Harden API service (timeouts, structured logs, health checks, rate limiting).
- Add a compatibility layer for multiple plugins with per-plugin parameter schemas.

## 7. Engineering roadmap

- Short term: stabilize dependencies, add tests, lock E2E commands.
- Mid term: improve data quality + metrics + model registry.
- Long term: multi-plugin support, latency optimization, deployment-ready API.

## 8. Dev validation checks

Install dev tooling:

```bash
cd HARMONIA
python3 -m pip install -r requirements-dev.txt
```

Run checks:

```bash
cd HARMONIA
python3 -m pytest -q
python3 -m bandit -q -r scripts src
python3 -m compileall scripts src tests
```

