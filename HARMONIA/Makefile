VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DATASET ?= data/cleaned_dataset.npy
PROMPT ?= Warm cinematic pad with soft attack
OUTPUT ?= data/processed/generated_from_prompt.json

.PHONY: help venv setup check test security compile train train-cleaned train-fast train-good train-and-report generate-cli pipeline-local serve metrics-local metrics-api estimate-train-time

help:
	@echo "Targets:"
	@echo "  make setup              # Create venv and install dependencies"
	@echo "  make check              # pytest + bandit + compileall"
	@echo "  make train              # Train with DATASET (default: data/cleaned_dataset.npy)"
	@echo "  make train-cleaned      # Shortcut for cleaned_dataset.npy"
	@echo "  make train-fast         # 1 epoch smoke run on cleaned_dataset.npy"
	@echo "  make train-good         # Longer run (20 epochs) on cleaned_dataset.npy"
	@echo "  make train-and-report   # Train then print latest local metrics"
	@echo "  make generate-cli       # Generate one preset JSON from PROMPT"
	@echo "  make pipeline-local     # check + train-fast + metrics + generate"
	@echo "  make metrics-local      # Print latest benchmark/report from local files"
	@echo "  make metrics-api        # Query http://127.0.0.1:5000/metrics/latest"
	@echo "  make estimate-train-time# Estimate duration from benchmark history"
	@echo "  make serve              # Run Flask API"

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"

setup: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-ci.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q && $(PYTHON) scripts/push_latest_metrics.py

security:
	$(PYTHON) -m bandit -q -r scripts src

compile:
	$(PYTHON) -m compileall scripts src tests

check: test security compile

train:
	HARMONIA_DATASET_PATH="$(DATASET)" $(PYTHON) scripts/train.py

train-cleaned:
	HARMONIA_DATASET_PATH="data/cleaned_dataset.npy" $(PYTHON) scripts/train.py

train-fast:
	HARMONIA_DATASET_PATH="data/cleaned_dataset.npy" HARMONIA_EPOCHS=1 HARMONIA_BATCH_SIZE=32 $(PYTHON) scripts/train.py

train-good:
	HARMONIA_DATASET_PATH="data/cleaned_dataset.npy" HARMONIA_EPOCHS=20 HARMONIA_BATCH_SIZE=16 $(PYTHON) scripts/train.py

train-and-report: train metrics-local

generate-cli:
	$(PYTHON) scripts/generate.py "$(PROMPT)" --output "$(OUTPUT)"
	@echo "Generated preset written to $(OUTPUT)"

pipeline-local: check train-fast metrics-local generate-cli

serve:
	$(PYTHON) scripts/server.py

metrics-local:
	$(PYTHON) scripts/metrics_summary.py

metrics-api:
	@curl -fsS http://127.0.0.1:5000/metrics/latest | $(PYTHON) -m json.tool

estimate-train-time:
	$(PYTHON) scripts/metrics_summary.py --estimate

