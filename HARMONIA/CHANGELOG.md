# CHANGELOG.md 📜

All notable changes to the **Harmonia** project will be documented in this file.

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