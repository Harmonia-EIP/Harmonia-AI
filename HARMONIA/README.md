# HARMONIA

AI-powered VST parameter generator for JUCE/C++ plugins.

Harmonia maps text prompts (for example, `"Soft Piano"` or `"Aggressive Bass"`) to normalized plugin parameters (`0.0` to `1.0`).

## Features

- Text-to-parameter inference with a lightweight BERT encoder (`prajjwal1/bert-tiny`)
- Flask API for realtime plugin integration
- API health endpoint (`GET /health`) + payload validation on `POST /generate`
- Latest evaluation endpoint (`GET /metrics/latest`)
- Dataset preparation from raw `.fxp` text dumps
- Manual training + benchmark history + validation metrics reports
- Optional auto-training watchdog pipeline

## Project layout

```text
HARMONIA/
├── benchmarks/
├── data/
│   ├── raw/
│   └── processed/
├── saved_models/
├── scripts/
└── src/
```

## Installation

From repository root:

```bash
cd HARMONIA
python3 -m pip install -r requirement.txt
```

## Quickstart (from repository root)

### 1) Prepare dataset

Place raw dump file in `HARMONIA/data/raw/my_raw_dump.txt`, then run:

```bash
python3 HARMONIA/scripts/prepare_dataset.py
```

Output: `HARMONIA/data/processed/presets.json`

### 2) Train model

```bash
python3 HARMONIA/scripts/train.py
```

Outputs:
- `HARMONIA/saved_models/<model_version>/my_plugin_ai.pth`
- `HARMONIA/saved_models/<model_version>/my_plugin_ai.meta.json`
- `HARMONIA/saved_models/latest_model.json`
- `HARMONIA/benchmarks/history.json`
- `HARMONIA/benchmarks/reports/eval_*.json`

### 3) Generate via CLI

```bash
python3 HARMONIA/scripts/generate.py "Soft Piano" --output HARMONIA/data/processed/test_preset.json
```

### 4) Run API server

```bash
python3 HARMONIA/scripts/server.py
```

Endpoint: `POST http://127.0.0.1:5000/generate`

Health check: `GET http://127.0.0.1:5000/health`

Latest metrics: `GET http://127.0.0.1:5000/metrics/latest`

Request body:

```json
{
  "prompt": "Dark Reese Bass"
}
```

Validation rules:
- `prompt` is required and must be a string
- empty/whitespace prompt returns `400`
- prompt length is capped to `512` chars
- prompts exceeding model token context (default `32` tokens) are rejected with `400`
- if model weights are unavailable, API returns `503`

`/health` and `/generate` expose `model_version` and `model_hash`.
`/metrics/latest` exposes the latest benchmark entry and latest evaluation report payload.

### 5) View benchmark history

```bash
python3 HARMONIA/scripts/benchmark_viewer.py
```

## Reproducible training

Training now uses a deterministic seed (`HARMONIA_SEED`, default `42`).

```bash
HARMONIA_SEED=123 python3 HARMONIA/scripts/train.py
```

Validation split can be configured with `HARMONIA_VAL_SPLIT` (default `0.2`):

```bash
HARMONIA_SEED=123 HARMONIA_VAL_SPLIT=0.25 python3 HARMONIA/scripts/train.py
```

Benchmark entries include dataset split and evaluation metadata. Per-parameter MAE/MSE are written to `HARMONIA/benchmarks/reports/`.

You can force a model version name with `HARMONIA_MODEL_VERSION`; otherwise training auto-generates one.

Trained metadata now carries `param_keys`, `plugin_param_count`, and `tokenizer_max_length`, and the API/CLI read these dynamically at inference time.

## Commandes de test (local + CI)

Install dependencies (runtime + dev):

```bash
cd HARMONIA
python3 -m pip install -r requirement.txt
python3 -m pip install -r requirements-dev.txt
```

Run checks:

```bash
cd HARMONIA
python3 -m pytest -q
python3 -m bandit -q -r scripts src
python3 -m compileall scripts src tests
```

Quick run one test file:

```bash
cd HARMONIA
python3 -m pytest -q tests/test_server.py
```

These commands are exactly the same as the GitHub Actions CI checks on each push/PR (`.github/workflows/harmonia-ci.yml`).

## Auto-training (optional)

Run watcher:

```bash
python3 HARMONIA/scripts/auto_trainer.py
```

Drop `.txt` files into `HARMONIA/data/raw/drop_zone/`.

## Production serving (recommended)

```bash
cd HARMONIA
gunicorn --bind 127.0.0.1:5000 --workers 2 --threads 4 scripts.server:app
```

## Current limitations

- Training quality is currently constrained by small dataset size.
- Training dataset and default profile still target 9 synth parameters.
- No model registry service yet (metadata is local JSON files only).
- API currently runs on local Flask dev server (not production hardened).

See `HARMONIA/PROJECT_TECH_AUDIT.md` for the detailed engineering analysis and roadmap.
