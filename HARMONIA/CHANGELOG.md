# CHANGELOG.md 📜

All notable changes to the **Harmonia** project will be documented in this file.

## [0.0.16] - Bandit B310 Fix on Metrics Publisher
### Changed
- **Security hardening** (`src/metrics_publisher.py`):
  - Replaced `urllib.request.urlopen` with `http.client` to remove Bandit `B310` finding.
  - Added strict URL validation before push (scheme + host allowlist).
  - Restricted metrics push hosts to trusted targets (`harmonia.mcoet.com`, `localhost`, `127.0.0.1`).

### Verified
- `python -m bandit -q -r scripts src` -> no `B310` issue
- `python -m pytest -q` -> `25 passed`

## [0.0.15] - Secure Token Handling and Push-Time Dashboard Refresh
### Changed
- **Secret hygiene**:
  - Removed any hardcoded token fallback from `metrics_dashboard/push_metrics.sh`.
  - Hardened `metrics_dashboard/receiver.php` to reject requests when `METRICS_PUSH_TOKEN` is not configured.
- **Dashboard richness**:
  - Upgraded `metrics_dashboard/index.html` to display many more metrics:
    - dynamic KPI cards,
    - complete numeric metrics table,
    - history/per-parameter chart selection,
    - resilient fallback behavior.
- **Push-time refresh**:
  - Updated `.github/workflows/harmonia-ci.yml` to publish a CI metrics payload to `https://harmonia.mcoet.com/receiver.php` on every push (when `METRICS_PUSH_TOKEN` secret is present).

### Security
- Removed local plaintext token file and kept only `.gitignore`-protected secret paths (`metrics_dashboard/.env.local`, generated metrics files).

## [0.0.14] - Automatic Metrics Push and Safer Git Add
### Added
- **Automatic metrics publisher**:
  - Added `src/metrics_publisher.py` to send evaluation reports directly to `https://harmonia.mcoet.com/receiver.php`.
  - Added `scripts/push_latest_metrics.py` to push the latest report from benchmark history.
- **Training hook**:
  - `scripts/train.py` now pushes `eval_*.json` automatically after each successful training run.

### Changed
- **Makefile behavior**:
  - `make test` now runs tests and then pushes latest metrics (non-blocking when no report/token is available).
- **Git safety**:
  - Updated `HARMONIA/.gitignore` to ignore local dashboard secrets and generated metrics files:
    - `metrics_dashboard/.env.local`
    - `metrics_dashboard/latest_metrics.json`
    - `benchmarks/reports/`

### Verified
- `make test` -> `25 passed` + metrics pushed (`HTTP 200`)
- `HARMONIA_EPOCHS=1 make train DATASET=data/processed/presets.json` -> success + auto-push (`HTTP 200`)

## [0.0.13] - Static Web Metrics Dashboard (Apache/PHP)
### Added
- **Static dashboard package** (`HARMONIA/metrics_dashboard/`):
  - `index.html` with TailwindCSS + Chart.js for modern metric visualization.
  - Dynamic cards for key KPIs (`Loss`, `Accuracy`, `MSE`, `MAE`).
  - Automatic chart rendering from `per_param_mse` (bar) or `loss_history` (line).
  - Built-in fallback to simulated JSON when `latest_metrics.json` is missing.
- **Secure receiver endpoint**:
  - Added `receiver.php` to accept `POST` metric payloads and save `latest_metrics.json`.
  - Supports token auth via `Authorization: Bearer ...` or `POST token`.
  - Supports upload modes: multipart file (`metrics_file`), `metrics_json`, or raw JSON body.
- **Push automation scripts**:
  - Added `push_metrics.sh` (curl-based) and `push_metrics.py` (requests-based).
  - Designed for direct integration at the end of PyTorch training pipelines.
- **Usage documentation**:
  - Added `HARMONIA/metrics_dashboard/README.md` with deployment, local test, and production push examples.

### Verified
- `bash -n HARMONIA/metrics_dashboard/push_metrics.sh` -> success
- `python3 -m py_compile HARMONIA/metrics_dashboard/push_metrics.py` -> success
- PHP syntax check not run locally in this environment (`php` executable unavailable).

## [0.0.12] - CI Python Compatibility Fix
### Changed
- **GitHub Actions runtime**:
  - Updated `.github/workflows/harmonia-ci.yml` from Python `3.10` to `3.12`.
  - This fixes CI installation failure with `numpy==2.3.4` (which requires Python `>=3.11`).
- **Dependency compatibility guard**:
  - Added environment markers in `HARMONIA/requirements-ci.txt` and `HARMONIA/requirement.txt`:
    - `numpy==2.3.4` for Python `>=3.11`
    - `numpy==2.2.6` for Python `<3.11`
  - This keeps installs resilient if a runner/local environment uses Python `3.10`.

### Verified
- CI dependency set remains unchanged (`HARMONIA/requirements-ci.txt`, `HARMONIA/requirements-dev.txt`).
- Local quality checks still pass (`pytest`, `bandit`, `compileall`).

## [0.0.11] - Training Profiles and Practical AI Test Commands
### Added
- **Makefile execution profiles**:
  - Added `train-fast` (1 epoch smoke), `train-good` (20 epochs baseline), `generate-cli`, and `pipeline-local`.
  - Added `PROMPT` and `OUTPUT` variables for quick CLI inference tests.
- **Metrics utility script**:
  - Added `scripts/metrics_summary.py` to print latest local metrics and training-time estimates from benchmark history.
- **Operational testing flow**:
  - `pipeline-local` now provides a single local sequence: checks -> fast train -> local metrics -> generation test.

### Changed
- **Documentation UX**:
  - Updated `HARMONIA/README.md` with explicit commands to:
    - train using `cleaned_dataset.npy`,
    - test AI inference in CLI and API mode,
    - read metrics at each step.
  - Added practical training duration estimates based on latest observed benchmark run.
- **Makefile reliability**:
  - `metrics-local` and `estimate-train-time` now call `scripts/metrics_summary.py` (stable shell behavior).

### Verified
- `make check` -> success (`25 passed`, bandit warnings only on existing `# nosec B615`, compileall ok)
- `HARMONIA_EPOCHS=1 HARMONIA_BATCH_SIZE=32 make train-cleaned` -> success on `data/cleaned_dataset.npy`
- `make metrics-local` and `make estimate-train-time` -> success

## [0.0.10] - Makefile Workflow and End-to-End Metrics
### Added
- **Operational Makefile**:
  - Added `HARMONIA/Makefile` with practical targets:
    - `setup`, `check`, `test`, `security`, `compile`
    - `train`, `train-cleaned`, `train-and-report`
    - `serve`, `metrics-local`, `metrics-api`, `estimate-train-time`
- **Metrics Everywhere Workflow**:
  - Added local metrics helpers that print latest benchmark and evaluation report after training.
  - Added training time estimation from benchmark history (`estimate-train-time`).

### Changed
- **Training Configurability** (`scripts/train.py`):
  - `HARMONIA_EPOCHS`, `HARMONIA_BATCH_SIZE`, and `HARMONIA_LR` now override defaults safely.
  - This allows fast debug runs and longer production-quality runs without code edits.
- **Documentation**:
  - Updated `HARMONIA/README.md` and `HARMONIA/DOC.md` with Makefile usage, cleaned dataset training commands, and metrics commands.

### Verified
- `python -m pytest -q` -> `25 passed`
- `python -m bandit -q -r scripts src` -> success (warnings only on existing `# nosec B615` markers)
- `python -m compileall scripts src tests` -> success

## [0.0.9] - Dynamic NPY Training Dataset and Parameter Scaling
### Added
- **Dynamic Dataset Loader**:
  - `src/dataset.py` now supports `.npy` datasets via `numpy.load(..., allow_pickle=True)`.
  - Added flattening for `parameters.continuous`, `parameters.binary`, and `parameters.categorical` into one 1D tensor.
  - Added deterministic key extraction (`param_keys`) to keep output order stable across training and inference.
- **Categorical Normalization**:
  - Added optional per-key categorical normalization (divide by dataset max) to keep targets in `[0, 1]`.
- **Test Coverage**:
  - Added `tests/test_dataset.py` to validate `.npy` loading, key ordering, flattening, and categorical normalization.
  - Extended `tests/test_train.py` for dynamic dataset path resolution and dynamic metric key mapping.

### Changed
- **Dynamic Output Dimension**:
  - `scripts/train.py` now derives `plugin_param_count` from `len(dataset.param_keys)` instead of fixed constants.
  - `src/model.py` now validates output dimension and derives encoder hidden size from the loaded transformer config.
- **Training Dataset Selection**:
  - Added `HARMONIA_DATASET_PATH` environment override.
  - Training now auto-prefers `data/processed/presets.npy` when available, then falls back to `presets.json`.
- **Dependency Compatibility Fixes**:
  - Upgraded `torch` pin from `2.8.0` to `2.11.0` in `requirement.txt` and `requirements-ci.txt`.
  - Upgraded `bandit` pin from `1.8.6` to `1.9.4` in `requirements-dev.txt` to avoid Python 3.14 AST failures.
  - Upgraded `numpy` pin from `2.0.2` to `2.3.4` in `requirements-ci.txt` and added `numpy==2.3.4` to `requirement.txt`.
- **Documentation**:
  - Updated `HARMONIA/README.md` and `HARMONIA/DOC.md` with `.npy` dataset workflow and dynamic parameter behavior.

### Verified
- `python -m pytest -q` -> `25 passed`
- `python -m compileall scripts src tests` -> success
- `python -m bandit -q -r scripts src` -> success (warnings only on existing `# nosec B615` markers)

## [0.0.8] - CI Workflow Restoration and Test Commands Clarity
### Added
- **Missing CI Workflow**:
  - Added `.github/workflows/harmonia-ci.yml` at repository root.
  - Workflow runs on push/pull_request and executes the 3 project checks from `HARMONIA/`:
    - `python -m pytest -q`
    - `python -m bandit -q -r scripts src`
    - `python -m compileall scripts src tests`

### Changed
- **Documentation Clarity**:
  - Added an explicit `Commandes de test (local + CI)` section in `HARMONIA/README.md`.
  - Added the same `Commandes de test (local + CI)` section in `HARMONIA/DOC.md`.
  - Documented a quick targeted run command (`tests/test_server.py`) for fast local validation.

## [0.0.7] - Security Hardening and Dependency Cleanup
### Added
- **Dynamic Inference Configuration**:
  - Training metadata now includes `plugin_param_count`, `param_keys`, and `tokenizer_max_length`.
  - Inference (`scripts/server.py`, `scripts/generate.py`) now reads these metadata fields to avoid hardcoded parameter mapping.
- **Token Context Guard**:
  - `POST /generate` now rejects prompts that exceed model token context instead of silently truncating.

### Changed
- **Dependency Hygiene**:
  - Replaced the non-portable environment dump in `requirement.txt` with a minimal, portable runtime manifest.
- **Safer Model Loading**:
  - Removed unsafe fallback model loading path and enforce `weights_only=True` for PyTorch deserialization.
- **CLI Consistency**:
  - `scripts/generate.py` now mirrors server-side behavior for model/token context validation and metadata-driven output keys.

### Verified
- `python3 -m pytest -q` -> `21 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.6] - Metrics Endpoint, Model Versioning, and CI
### Added
- **API Metrics Endpoint**:
  - Added `GET /metrics/latest` in `scripts/server.py` to expose the latest benchmark entry and latest evaluation report payload.
- **Automatic Model Versioning**:
  - `scripts/train.py` now stores artifacts in `saved_models/<model_version>/`.
  - Added `saved_models/latest_model.json` pointer to resolve the most recent model automatically.
  - Added `src/artifact_registry.py` to centralize model artifact path resolution.
- **CI Automation**:
  - Added GitHub Actions workflow `.github/workflows/harmonia-ci.yml` running:
    - `python -m pytest -q`
    - `python -m bandit -q -r scripts src`
    - `python -m compileall scripts src tests`
  - Added `requirements-ci.txt` for portable CI dependency installation.

### Changed
- **Runtime Model Resolution**:
  - `scripts/server.py` and `scripts/generate.py` now auto-resolve the latest versioned model from `saved_models/`.
- **Training Metadata**:
  - `scripts/train.py` now writes versioned metadata and updates the latest-pointer file each run.
- **Tests and Docs**:
  - Extended `tests/test_server.py` and `tests/test_train.py` for new API and versioning behavior.
  - Updated `README.md` and `DOC.md` with new endpoint, model layout, and CI commands.

### Verified
- `python3 -m pytest -q` -> `19 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.5] - P1 Evaluation Metrics and Model Traceability
### Added
- **Validation Metrics Pipeline**:
  - `scripts/train.py` now creates a deterministic train/validation split (`HARMONIA_VAL_SPLIT`, default `0.2`).
  - Validation metrics now include global MAE/MSE and per-parameter MAE/MSE.
  - Added evaluation report artifacts in `benchmarks/reports/eval_*.json`.
- **Model Metadata Artifact**:
  - Training now writes `saved_models/my_plugin_ai.meta.json` with model version/hash and training context.
- **API Traceability**:
  - `GET /health` now exposes `model_version` and `model_hash`.
  - `POST /generate` now includes `model_version` and `model_hash` in response `metadata`.
- **Test Coverage**:
  - Extended `tests/test_train.py` to cover split sizing and evaluation report creation.
  - Extended API tests for traceability fields.

### Changed
- **Benchmark Enrichment**:
  - `benchmarks/history.json` entries now include train/validation sizes, evaluation summary, report path, and model identifiers.
- **Documentation**:
  - Updated `README.md` and `DOC.md` with P1 outputs and run/test commands (commands unchanged).

### Verified
- `python3 -m pytest -q` -> `15 passed`
- `python3 -m bandit -q -r scripts src` -> clean run
- `python3 -m compileall scripts src tests` -> success

## [0.0.4] - Reliability, Security Baseline, and Tests
### Added
- **API Reliability**:
  - Added `GET /health` endpoint in `scripts/server.py` for runtime status checks.
  - Added strict payload validation for `POST /generate` (`prompt` type, emptiness, max length 512).
- **Reproducible Training**:
  - Added deterministic seed support in `scripts/train.py` via `HARMONIA_SEED` (default: `42`).
  - Benchmark entries now include `seed` and `dataset_size` metadata.
- **Test Suite**:
  - Added `tests/test_server.py` for API behavior and error-path validation.
  - Added `tests/test_prepare_dataset.py` for parser and conversion checks.
  - Added `tests/test_train.py` for deterministic seed and benchmark metadata tests.
  - Added `tests/test_e2e_smoke.py` for a lightweight prepare+generate smoke path.
- **Dev Tooling**:
  - Added `requirements-dev.txt` with `pytest` and `bandit`.

### Changed
- **Model Supply Chain Controls**:
  - Added model source pinning support with `HARMONIA_MODEL_ID` and `HARMONIA_MODEL_REVISION`.
  - Applied pinned revision loading for tokenizer/model in `scripts/train.py`, `scripts/server.py`, `scripts/generate.py`, and `src/model.py`.
- **Safer Weight Loading**:
  - Updated model loading paths to prefer `torch.load(..., weights_only=True)` with compatibility fallback.
- **Auto-Training Execution Safety**:
  - Added a strict script allowlist and resolved-path checks before subprocess execution in `scripts/auto_trainer.py`.
- **Documentation**:
  - Updated `README.md` and `DOC.md` with health endpoint, payload validation, reproducible training, and dev validation commands.

### Verified
- `python3 -m pytest -q` -> `13 passed`
- `python3 -m compileall scripts src tests` -> success
- `python3 -m bandit -q -r scripts src` -> clean run (no reported findings)

## [0.0.3] - Runtime Stability and Technical Audit
### Added
- **Technical Audit**: Added `PROJECT_TECH_AUDIT.md` with current limitations, engineering risks, and a concrete 30-60-90 roadmap.
- **Runtime Guards**:
  - `server.py` now returns `503` when model weights are missing/invalid.
  - `train.py` now stops early on empty datasets.
  - `auto_trainer.py` now ignores non-`.txt` files.

### Changed
- **Path Handling**: Switched critical scripts to absolute paths derived from script location:
  - `prepare_dataset.py`
  - `train.py`
  - `generate.py`
  - `server.py`
  - `benchmark_viewer.py`
  - `auto_trainer.py`
- **Documentation Alignment**:
  - Updated `README.md` and `DOC.md` to match executable commands from repository root.

### Fixed
- **Server Import**: Fixed `ModuleNotFoundError` in `server.py` by ensuring `src/` is available in `sys.path`.
- **Dataset Output Path**: `prepare_dataset.py` default output now correctly targets `data/processed/presets.json`.
- **Benchmark Directory Logic**: `train.py` now creates the real benchmark directory before writing history.
- **Benchmark Viewer Path**: `benchmark_viewer.py` now reads benchmark history correctly regardless of current working directory.
- **Auto-Trainer Subprocesses**: `auto_trainer.py` now executes scripts with deterministic absolute paths and `sys.executable`.

## [0.0.2] - Refactored Architecture
### Changed
- **Project Structure**: Reorganized repository into professional standard structure.
    - `src/`: Core logic (Model definition, Dataset class).
    - `scripts/`: Executable scripts (Train, Generate, Auto-Trainer).
    - `data/`: Storage for raw and processed datasets.
- **Model Definition**: Consolidated `model.py` into `src/` to prevent duplication.
- **Imports**: Updated all scripts to dynamically find the `src` package.

### Fixed
- **Auto Trainer**: Fixed subprocess paths to work within the new `scripts/` directory.
- **Training**: Fixed data paths in `train.py` to point to `../data/`.


## [0.0.1] : Server and Benchmark
### Added
- **Flask API Server (`server.py`)**: A lightweight web server allowing JUCE plugins to request parameters via HTTP POST requests instead of file I/O.
- **Auto-Trainer (`auto_trainer.py`)**: A `watchdog` system that monitors a `drop_zone` folder. Dropping a text file now automatically triggers dataset ingestion, training, and benchmarking.
- **Benchmark System**:
- `train.py` now logs training duration, hyperparameters, and final loss to `benchmarks/history.json`.
- `benchmark_viewer.py` provides a CLI table view to track model improvement (or regression) over time.
- **9-Parameter Mapping**: Updated the model output layer to support specific target knobs: *Frequency, Attack, Cutoff, Decay, Volume, Sustain, Resonance, Release, Waveform*.
- **Robust Data Parser**: `prepare_dataset.py` now handles malformed text dumps (missing parentheses) gracefully.

### Changed
- **Model Architecture**: Switched output layer size from 50 to 9 to match the target VST requirements.
- **Inference Output**: `generate.py` now produces a named JSON dictionary (e.g., `"cutoff": 0.5`) instead of a raw list of floats, making integration with C++ easier.

### Removed
- **Spectrogram Support**: Deprecated the audio-spectrogram approach in favor of direct parameter regression (Text-to-Param).