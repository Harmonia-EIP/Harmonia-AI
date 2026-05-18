VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DATASET ?= data/processed/synthetic_dataset.charter.npy
SYNTH_PER_PROMPT ?= 50
OUTPUT ?=
METRICS_TOKEN ?= metrics_dashboard/.env

.PHONY: help venv setup check test security compile train train-cleaned train-fast train-good train-and-report generate-cli pipeline-local serve metrics-local metrics-api estimate-train-time push-metrics dashboard-stats dashboard-snapshot dashboard-serve charterize-dataset synthesize-dataset train-synth

help:
	@echo "Targets:"
	@echo "  make setup              # Create venv and install dependencies"
	@echo "  make check              # pytest + bandit + compileall"
	@echo "  make train              # Train with DATASET ($(DATASET))"
	@echo "  make train-synth        # Rebuild synthetic dataset then train (15 ep)"
	@echo "  make synthesize-dataset # Build synthetic_dataset.charter.npy (100 concepts)"
	@echo "  make train-fast         # 1 epoch smoke run on DATASET"
	@echo "  make train-good         # Longer run (20 epochs) on DATASET"
	@echo "  make train-and-report   # Train then print latest local metrics"
	@echo "  make generate-cli       # Generate one preset JSON from a prompt via CLI"
	@echo "  make pipeline-local     # check + train-fast + metrics + generate"
	@echo "  make metrics-local      # Print latest benchmark/report from local files"
	@echo "  make metrics-api        # Query http://127.0.0.1:5000/metrics/latest"
	@echo "  make estimate-train-time# Estimate duration from benchmark history"
	@echo "  make serve              # Run Flask API"
	@echo "  make push-metrics       # Flush local event buffer to the remote dashboard"
	@echo "  make dashboard-stats    # Build snapshot + push to harmonia.mcoet.com"
	@echo "  make dashboard-snapshot # Build local snapshot only (no push)"
	@echo "  make dashboard-serve    # Serve metrics_dashboard/ on http://127.0.0.1:8080"
	@echo "  make charterize-dataset # Convert cleaned_dataset.npy -> 20-param charter NPY"

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"

setup: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-ci.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

security:
	$(PYTHON) -m bandit -q -r scripts src

compile:
	$(PYTHON) -m compileall scripts src tests

check: test security compile

train:
	HARMONIA_DATASET_PATH="$(DATASET)" $(PYTHON) scripts/train.py

push-metrics:
	@METRICS_TOKEN=$${METRICS_TOKEN:-} && if [ -z "$$METRICS_TOKEN" ]; then echo "Set METRICS_TOKEN env var before running"; exit 1; fi; \
	$(PYTHON) scripts/push_pending_events.py

synthesize-dataset:
	$(PYTHON) scripts/synthesize_dataset.py --per-prompt $(SYNTH_PER_PROMPT) --mix-anchors

train-synth: synthesize-dataset
	HARMONIA_DATASET_PATH=data/processed/synthetic_dataset.charter.npy \
	HARMONIA_EPOCHS=15 HARMONIA_BATCH_SIZE=64 HARMONIA_LR=2e-4 HARMONIA_VAL_SPLIT=0.05 \
	HARMONIA_MODEL_VERSION=synthetic_v$$(date +%Y%m%d_%H%M%S) \
	$(PYTHON) scripts/train.py

train-fast:
	HARMONIA_DATASET_PATH="$(DATASET)" HARMONIA_EPOCHS=1 HARMONIA_BATCH_SIZE=32 $(PYTHON) scripts/train.py

train-good:
	HARMONIA_DATASET_PATH="$(DATASET)" HARMONIA_EPOCHS=20 HARMONIA_BATCH_SIZE=16 $(PYTHON) scripts/train.py

train-and-report: train metrics-local

generate-cli:
	@printf "Entrez le nom du preset (Prompt)%s: " "$(if $(strip $(PROMPT)), [$(PROMPT)],)"; \
	read -r USER_PROMPT; \
	USER_PROMPT=$${USER_PROMPT:-$(PROMPT)}; \
	if [ -n "$(strip $(OUTPUT))" ]; then \
		OUTPUT_PATH="$(OUTPUT)"; \
	else \
		SAFE_PROMPT=$$($(PYTHON) -c 'import re, sys; value = sys.argv[1].strip(); safe = re.sub(r"[^A-Za-z0-9._-]", "-", value).strip(".-"); print((safe or "prompt").lower())' "$$USER_PROMPT"); \
		OUTPUT_PATH="data/processed/presets/$${SAFE_PROMPT}.json"; \
	fi; \
	mkdir -p "$$(dirname "$$OUTPUT_PATH")"; \
	$(PYTHON) scripts/generate.py "$$USER_PROMPT" --output "$$OUTPUT_PATH"; \
	echo "Generated preset written to $$OUTPUT_PATH"

pipeline-local: check train-fast metrics-local generate-cli

serve:
	$(PYTHON) scripts/server.py

metrics-local:
	$(PYTHON) scripts/metrics_summary.py

metrics-api:
	@curl -fsS http://127.0.0.1:5000/metrics/latest | $(PYTHON) -m json.tool

estimate-train-time:
	$(PYTHON) scripts/metrics_summary.py --estimate

dashboard-stats:
	$(PYTHON) scripts/dashboard_stats.py

dashboard-snapshot:
	$(PYTHON) scripts/dashboard_stats.py --no-push

dashboard-serve:
	@echo "Serving metrics_dashboard/ on http://127.0.0.1:8080 (Ctrl+C to stop)"
	cd metrics_dashboard && php -S 127.0.0.1:8080

charterize-dataset:
	$(PYTHON) scripts/charterize_dataset.py \
		--input data/processed/cleaned_dataset.npy \
		--output data/processed/cleaned_dataset.charter.npy

