# HARMONIA Technical Documentation

Harmonia is a text-to-parameter AI system for VST/JUCE integration.

It converts natural-language prompts into normalized synthesizer parameters in the range `[0.0, 1.0]`.

## 1. Architecture

- Encoder: `prajjwal1/bert-tiny` from Hugging Face (`transformers`)
- Regressor head: small MLP (`Linear -> ReLU -> Dropout -> Linear -> Sigmoid`)
- Output: dynamic parameter vector (`len(param_keys)` extracted from dataset)
- Parameter order: deterministic flatten of `continuous`, then `binary`, then `categorical` keys (sorted per group)

## 2. Runtime components

- `scripts/prepare_dataset.py`: parse raw dumps and build `data/processed/presets.json`
- `scripts/train.py`: train, save model, produce benchmark history and validation metrics report
- `scripts/generate.py`: CLI inference
- `scripts/server.py`: Flask API (`POST /generate`, `GET /health`, `GET /metrics/latest`)
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

Optional Makefile shortcuts:

```bash
cd HARMONIA
make setup
make check
make train-cleaned
make metrics-local
make serve
```

### A. Dataset preparation

```bash
python3 HARMONIA/scripts/prepare_dataset.py
```

Input: `HARMONIA/data/raw/my_raw_dump.txt`  
Output: `HARMONIA/data/processed/presets.json`

Alternative dataset format for training:
- `HARMONIA/data/processed/presets.npy` (numpy object array, list of dicts)
- Record schema: `raw_text` + `parameters.continuous/binary/categorical`

### B. Model training

```bash
python3 HARMONIA/scripts/train.py
```

Custom dataset path (JSON or NPY):

```bash
HARMONIA_DATASET_PATH=/absolute/path/to/presets.npy python3 HARMONIA/scripts/train.py
```

Training with your `cleaned_dataset.npy`:

```bash
cd HARMONIA
source .venv/bin/activate
HARMONIA_DATASET_PATH=data/cleaned_dataset.npy python scripts/train.py
```

Training knobs (speed vs quality):

```bash
HARMONIA_EPOCHS=120 HARMONIA_BATCH_SIZE=16 HARMONIA_LR=0.0001 python3 HARMONIA/scripts/train.py
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
- `HARMONIA/saved_models/<model_version>/my_plugin_ai.pth`
- `HARMONIA/saved_models/<model_version>/my_plugin_ai.meta.json`
- `HARMONIA/saved_models/latest_model.json`
- `HARMONIA/benchmarks/history.json`
- `HARMONIA/benchmarks/reports/eval_*.json`

By default training uses a deterministic `train/validation` split (`HARMONIA_VAL_SPLIT=0.2`) and stores MAE/MSE per parameter in the evaluation report.

For NPY datasets:
- training auto-loads with `numpy.load(..., allow_pickle=True)`
- target vectors are flattened from `continuous + binary + categorical`
- categorical values can be normalized by per-key max (`0..1`) for stable regression

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
GET  http://127.0.0.1:5000/metrics/latest
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
- prompts exceeding model token context (default `32` tokens) are rejected
- if model weights are unavailable/invalid, API returns `503`

API traceability:
- `GET /health` returns `model_version` and `model_hash`
- `POST /generate` includes `model_version` and `model_hash` in `metadata`
- `GET /metrics/latest` returns the latest benchmark and the latest evaluation report payload

Model compatibility:
- Inference reads `param_keys`, `plugin_param_count`, and `tokenizer_max_length` from model metadata when available.
- Model loading enforces `torch.load(..., weights_only=True)` for safer deserialization.

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
- Dataset schema consistency is critical; changing key names changes output dimension/order.
- No external model registry service yet (current metadata storage is local-file based).
- Flask development server is suitable for local integration, not production.
- Production serving still requires external process management (e.g. gunicorn/systemd).

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

## 8. Commandes de test (local + CI)

Depuis la racine du repo, setup recommandé:

```bash
cd HARMONIA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt -r requirements-dev.txt
```

Checks (identiques a la CI):

```bash
cd HARMONIA
source .venv/bin/activate
python -m pytest -q
python -m bandit -q -r scripts src
python -m compileall scripts src tests
```

Run only API tests (fast path):

```bash
cd HARMONIA
source .venv/bin/activate
python -m pytest -q tests/test_server.py
```

CI note:
- The same three commands are executed in GitHub Actions on push/PR via `.github/workflows/harmonia-ci.yml`.

Metrics helpers:
- `make metrics-local` prints latest local benchmark + report.
- `make metrics-api` calls `GET /metrics/latest` when server is running.
- `make estimate-train-time` estimates expected duration from benchmark history.

