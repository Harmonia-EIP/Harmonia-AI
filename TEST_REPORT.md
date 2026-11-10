# Test Report - HARMONIA AI Project

**Execution Date:** 09/11/2026
**Target Environment:** Windows PC

## Test Execution Results

| ID | Module | Test Scenario | Expected Result | Actual Result | Status | Ticket |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ENV-01** | Setup | Execute `make setup` or manual installation via the virtual environment (`requirements-ci.txt`, `requirements-dev.txt`) | The `.venv` virtual environment is created and all dependencies are installed without error. | Successful installation | ✅ SUCCESS | - |
| **CI-01** | Checks | Execute the test suite and static analysis via `make check` | `pytest`, `bandit`, and `compileall` execute successfully and report no blocking errors. | Tests and security checks OK | ✅ SUCCESS | - |
| **DATA-01** | Dataset | Prepare the data with `python3 scripts/prepare_dataset.py` from a raw text dump file | The script correctly generates the output file `data/processed/presets.json` or `.npy`. | File generated successfully | ✅ SUCCESS | - |
| **TRAIN-01** | Training | Run a fast training test (smoke test) via `make train-fast` | The training successfully completes 1 epoch and saves the files (`.pth`, `.meta.json`) in `saved_models/`. | 1 epoch completed, models saved | ✅ SUCCESS | - |
| **TRAIN-02** | Training | Verify reproducibility with a fixed seed (`HARMONIA_SEED=123 python3 scripts/train.py`) | The training results and generated weights are identical for two executions with the same seed. | Reproducibility confirmed | ✅ SUCCESS | - |
| **CLI-01** | Inference | Generate a preset via the CLI: `python3 scripts/generate.py "Soft Piano"` | A valid JSON file containing the preset is created in the specified output path. | JSON file generated | ✅ SUCCESS | - |
| **API-01** | Server | Start the API (`scripts/server.py`) and call `GET /health` | The API responds with 200 OK and correctly exposes `model_version` and `model_hash`. | Server UP, 200 status | ✅ SUCCESS | - |
| **API-02** | Server | Call `POST /generate` with a valid payload (`{"prompt": "Dark Reese Bass"}`) | The API responds with 200 OK and returns the generated preset. | Preset generated via API | ✅ SUCCESS | - |
| **API-03** | Validation | Send an empty prompt or one containing only spaces to `POST /generate` | The API rejects the request and returns an HTTP 400 code. | Correct rejection (Error 400) | ✅ SUCCESS | - |
| **API-04** | Validation | Send a prompt exceeding the character limit (512) or token limit (32) | The API rejects the request and returns an HTTP 400 code. | Correct rejection (Error 400) | ✅ SUCCESS | - |
| **METRICS-01** | Pipeline | Run the local metrics calculation with `make metrics-local` | The evaluation files `eval_*.json` and `history.json` are updated in `benchmarks/`. | Metrics generated and stored | ✅ SUCCESS | - |