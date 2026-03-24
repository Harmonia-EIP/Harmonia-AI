# CHANGELOG.md 📜

All notable changes to the **Harmonia** project will be documented in this file.

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