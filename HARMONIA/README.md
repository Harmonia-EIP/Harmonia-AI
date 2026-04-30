# HARMONIA

AI-powered VST parameter generator for JUCE/C++ plugins.

Harmonia maps text prompts (for example, `"Soft Piano"` or `"Aggressive Bass"`) to normalized plugin parameters (`0.0` to `1.0`).

## Features

- Text-to-parameter inference with a lightweight BERT encoder (`prajjwal1/bert-tiny`)
- Flask API for realtime plugin integration
- API health endpoint (`GET /health`) + payload validation on `POST /generate`
- Latest evaluation endpoint (`GET /metrics/latest`)
- Dataset preparation from raw `.fxp` text dumps
- Native training support for large `.npy` datasets (numpy object arrays)
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

### 0) One-command setup with Makefile (recommended)

```bash
cd HARMONIA
make setup
```

Useful shortcuts:

```bash
cd HARMONIA
make check
make train-cleaned
make train-fast
make train-good
make metrics-local
make generate-cli
make pipeline-local
make serve
make metrics-api
```

## Commandes de test (ultra clair)

Depuis `HARMONIA/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt -r requirements-dev.txt
```

Checks (les 3 commandes CI):

```bash
python -m pytest -q
python -m bandit -q -r scripts src
python -m compileall scripts src tests
```

Equivalent Makefile:

```bash
make check
```

### 1) Prepare dataset

Place raw dump file in `HARMONIA/data/raw/my_raw_dump.txt`, then run:

```bash
python3 HARMONIA/scripts/prepare_dataset.py
```

Output: `HARMONIA/data/processed/presets.json`

You can also train directly from a prebuilt `.npy` dataset (list of dicts) placed in
`HARMONIA/data/processed/presets.npy`.

### 2) Train model

```bash
python3 HARMONIA/scripts/train.py
```

If your dataset is elsewhere:

```bash
HARMONIA_DATASET_PATH=/absolute/path/to/your_presets.npy python3 HARMONIA/scripts/train.py
```

With your new file:

```bash
cd HARMONIA
source .venv/bin/activate
HARMONIA_DATASET_PATH=data/cleaned_dataset.npy python scripts/train.py
```

Or with Makefile:

```bash
cd HARMONIA
make train-cleaned
```

Quick profiles:

```bash
cd HARMONIA
make train-fast   # smoke test: 1 epoch
make train-good   # better quality baseline: 20 epochs
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

You can also tune training speed/quality:

```bash
cd HARMONIA
source .venv/bin/activate
HARMONIA_DATASET_PATH=data/cleaned_dataset.npy HARMONIA_EPOCHS=120 HARMONIA_BATCH_SIZE=16 python scripts/train.py
```

### How long will training take?

From the latest observed local run on `data/cleaned_dataset.npy`:
- `1 epoch` on `52,763` presets took about `25s`.

Rough estimate on the same machine/profile:
- `20 epochs` -> about `8-10 min`
- `100 epochs` -> about `40-50 min`

Actual time can vary with CPU load, batch size, and model cache state.

Benchmark entries include dataset split and evaluation metadata. Per-parameter MAE/MSE are written to `HARMONIA/benchmarks/reports/`.

You can force a model version name with `HARMONIA_MODEL_VERSION`; otherwise training auto-generates one.

Trained metadata now carries `param_keys`, `plugin_param_count`, and `tokenizer_max_length`, and the API/CLI read these dynamically at inference time.
For `.npy` datasets, `param_keys` are auto-extracted from `continuous + binary + categorical` and ordered deterministically.

## Metrics at every step

Local metrics after each training run:

```bash
cd HARMONIA
make metrics-local
```

One-command local pipeline (checks + fast train + metrics + generation):

```bash
cd HARMONIA
make pipeline-local
```

Run one CLI generation test:

```bash
cd HARMONIA
make generate-cli
# or custom prompt/output:
make generate-cli PROMPT="Huge dark bass with short release" OUTPUT=data/processed/bass_test.json
```

Estimate expected training duration from your benchmark history:

```bash
cd HARMONIA
make estimate-train-time
```

API metrics (when server is running):

```bash
cd HARMONIA
make metrics-api
```

Automatic metrics push is enabled after training when a report exists.
It uses `HARMONIA/metrics_dashboard/.env.local` or the `METRICS_TOKEN` environment variable.
CI also refreshes the dashboard on each git push (if GitHub secret `METRICS_PUSH_TOKEN` is set).

Files ignored by git so a `git add *` won't stage secrets or generated metrics:

- `HARMONIA/metrics_dashboard/.env.local`
- `HARMONIA/metrics_dashboard/latest_metrics.json`
- `HARMONIA/benchmarks/reports/`

Test the HTTP API generation endpoint:

```bash
cd HARMONIA
source .venv/bin/activate
python scripts/server.py
# in another terminal
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huge dark bass with short release"}' | python -m json.tool
```

## Commandes de test (local + CI)

Depuis la racine du repo, le plus simple est de passer par un venv local:

```bash
cd HARMONIA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt -r requirements-dev.txt
```

Puis lance exactement les 3 checks CI:

```bash
cd HARMONIA
source .venv/bin/activate
python -m pytest -q
python -m bandit -q -r scripts src
python -m compileall scripts src tests
```

Test ciblé (rapide):

```bash
cd HARMONIA
source .venv/bin/activate
python -m pytest -q tests/test_server.py
```

Ces trois commandes sont celles exécutées en CI sur chaque push/PR via `.github/workflows/harmonia-ci.yml`.

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
- Output dimension now follows dataset keys dynamically; quality still depends on key consistency in the dataset.
- No model registry service yet (metadata is local JSON files only).
- API currently runs on local Flask dev server (not production hardened).

See `HARMONIA/PROJECT_TECH_AUDIT.md` for the detailed engineering analysis and roadmap.
