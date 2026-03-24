# HARMONIA

AI-powered VST parameter generator for JUCE/C++ plugins.

Harmonia maps text prompts (for example, `"Soft Piano"` or `"Aggressive Bass"`) to normalized plugin parameters (`0.0` to `1.0`).

## Features

- Text-to-parameter inference with a lightweight BERT encoder (`prajjwal1/bert-tiny`)
- Flask API for realtime plugin integration
- API health endpoint (`GET /health`) + payload validation on `POST /generate`
- Dataset preparation from raw `.fxp` text dumps
- Manual training + benchmark history logging
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
- `HARMONIA/saved_models/my_plugin_ai.pth`
- `HARMONIA/benchmarks/history.json`

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
- if model weights are unavailable, API returns `503`

### 5) View benchmark history

```bash
python3 HARMONIA/scripts/benchmark_viewer.py
```

## Reproducible training

Training now uses a deterministic seed (`HARMONIA_SEED`, default `42`).

```bash
HARMONIA_SEED=123 python3 HARMONIA/scripts/train.py
```

Benchmark entries include `seed` and `dataset_size` metadata.

## Dev checks (tests + security + compile)

Install dev dependencies:

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

## Auto-training (optional)

Run watcher:

```bash
python3 HARMONIA/scripts/auto_trainer.py
```

Drop `.txt` files into `HARMONIA/data/raw/drop_zone/`.

## Current limitations

- Training quality is currently constrained by small dataset size.
- The model predicts only 9 fixed parameters.
- No dedicated validation split, metrics suite, or model version registry yet.
- API currently runs on local Flask dev server (not production hardened).

See `HARMONIA/PROJECT_TECH_AUDIT.md` for the detailed engineering analysis and roadmap.
