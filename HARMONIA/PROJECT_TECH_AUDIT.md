# Harmonia Project Technical Audit (2026-03-23)

## Scope

This audit reviews:
- `README.md`, `DOC.md`, `CHANGELOG.md`
- Core scripts in `scripts/`
- Core modules in `src/`
- End-to-end runability from workspace root

## Executive status

Current status after fixes in this branch:
- The core flow can run from repository root:
  1. `prepare_dataset.py`
  2. `train.py`
  3. `generate.py`
  4. `server.py`
- Several critical path bugs were fixed (path handling, API import path, model readiness handling).
- The project is now in a "functional prototype" state, not yet production-ready.

## What was broken before

1. Path resolution depended on current working directory.
   - Running scripts from repo root caused missing file errors.
2. `server.py` could not import `TextToParams` when launched from repo root.
3. Dataset preparation default output path was incorrect.
4. Benchmark path handling in training/viewer was inconsistent.
5. Auto-trainer subprocess calls assumed a specific current directory.
6. API server did not safely return `503` when model was unavailable.

## Fixes implemented

- Absolute path handling added in:
  - `scripts/prepare_dataset.py`
  - `scripts/train.py`
  - `scripts/generate.py`
  - `scripts/server.py`
  - `scripts/benchmark_viewer.py`
  - `scripts/auto_trainer.py`
- API robustness:
  - safe JSON parsing
  - clear `503` when model is unavailable
- Training robustness:
  - benchmark directory creation fixed
  - empty dataset guard
  - JSON decode safety in benchmark history loading
- Documentation alignment:
  - `README.md` and `DOC.md` updated to reflect real commands and paths.

## Current technical limits

### AI / ML

- Training data appears very small (quality bottleneck).
- No train/validation split in training script.
- No per-parameter metrics, only final loss trend.
- No confidence estimation for generated parameters.
- Single model architecture and single plugin schema only.

### Backend / API

- Flask dev server only (single-process, no hardened deployment).
- No authentication or request throttling.
- No request tracing or structured logs.
- No health endpoint for external orchestration.

### Network / Integration

- Contract is simple and fixed; no versioned API.
- No explicit timeout, retry, or fallback strategy for plugin-side client.
- No compatibility matrix for plugin versions.

### MLOps / Reliability

- No CI pipeline to validate scripts and docs automatically.
- No model registry or artifact lineage.
- No deterministic reproducibility policy (seed/device/package lock strategy).
- Dependency file is broad and not deployment-focused.

## Next technical challenges

1. Data engineering quality at scale
   - Define strict dataset schema and validation.
   - Remove malformed entries and enforce numeric ranges.
2. Evaluation discipline
   - Add validation split and stable baseline metrics.
   - Track MAE/MSE per parameter.
3. Inference reliability
   - Add model checksum/version in API responses.
   - Add optional output smoothing or constraints by parameter type.
4. Backend hardening
   - Move to production WSGI/ASGI setup.
   - Add health, readiness, logging, and error observability.
5. Productization
   - Multi-plugin support with per-plugin parameter maps.
   - Backward-compatible API versioning.

## Recommended improvements

### Priority P0 (make it concretely usable)

- Add integration test script that runs:
  - prepare -> train -> generate -> API smoke test
- Add `GET /health` route to API.
- Add payload validation (`prompt` length, type checks).
- Add deterministic seed in training for reproducibility.

### Priority P1 (quality and trust)

- Build train/validation split and metric report file.
- Add benchmark metadata:
  - dataset size
  - model hash/version
  - environment info
- Introduce minimal and portable dependency file for runtime/training.

### Priority P2 (scaling)

- Introduce model registry conventions:
  - `saved_models/<model_name>/<version>/`
- Add plugin profile abstraction:
  - `profiles/<plugin_name>.json`
- Add async inference queue option for plugin bursts.

## Suggested 30-60-90 day roadmap

### 30 days

- Stabilize dependency strategy.
- Add basic tests and API health endpoint.
- Add reproducible training setup.

### 60 days

- Add robust evaluation pipeline and reporting.
- Introduce model versioning and rollback policy.
- Define plugin contract versioning.

### 90 days

- Support multiple plugin schemas.
- Harden deployment architecture.
- Build monitoring dashboard for quality and latency drift.

## Practical commands

```bash
python3 HARMONIA/scripts/prepare_dataset.py
python3 HARMONIA/scripts/train.py
python3 HARMONIA/scripts/generate.py "warm ambient pad" --output HARMONIA/data/processed/demo.json
python3 HARMONIA/scripts/server.py
```

## Conclusion

Harmonia now has a reliable local prototype pipeline.
To become a concrete product, the next decisive gains are in data quality, evaluation rigor, API hardening, and model lifecycle management.

