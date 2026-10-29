VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DATASET ?= data/processed/synthetic_dataset.charter.npy
SYNTH_PER_PROMPT ?= 50
OUTPUT ?=
METRICS_TOKEN ?= metrics_dashboard/.env
FSD50K ?= $(HOME)/Datasets/FSD50K
V2_BANK_SIZE ?= 200000
V2_TAG ?=
JUDGE_PYTHON ?= .venv-judge/bin/python

.PHONY: help venv setup check test security compile train train-cleaned train-fast train-good train-and-report generate-cli pipeline-local serve run-flask metrics-local metrics-api estimate-train-time push-metrics dashboard-stats dashboard-snapshot dashboard-serve charterize-dataset synthesize-dataset train-synth setup-train v2-bank v2-fsd50k v2-corpus v2-text v2-analyze v2-tune v2-predictor v2-tune-hybrid v2-export v2-train v2-bench v2-package v2-fetch

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
	@echo "  make serve              # Run Flask API (port 5000)"
	@echo "  make run-flask          # Alias for serve"
	@echo "  make push-metrics       # Flush local event buffer to the remote dashboard"
	@echo "  make dashboard-stats    # Build snapshot + push to harmonia.mcoet.com"
	@echo "  make dashboard-snapshot # Build local snapshot only (no push)"
	@echo "  make dashboard-serve    # Serve metrics_dashboard/ on http://127.0.0.1:8080"
	@echo "  make charterize-dataset # Convert cleaned_dataset.npy -> 20-param charter NPY"
	@echo "  make setup-train        # Install the v2 training dependencies"
	@echo "  make v2-train           # Full v2 pipeline: bank, FSD50K, corpus, text encoder, tuning, predictor, export"
	@echo "  make v2-bench           # Benchmark v1 vs v2 with the independent judge (needs $(JUDGE_PYTHON))"
	@echo "  make v2-package V2_TAG=ai-v2.0.0  # Build the release asset and pin it in models/release.json"
	@echo "  make v2-fetch           # Download the pinned v2 model into models/harmonia_v2"

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

run-flask: serve

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


setup-train: venv
	$(PIP) install -r requirements-train.txt -r requirements-dev.txt

v2-bank:
	$(PYTHON) scripts/v2/build_preset_bank.py --n $(V2_BANK_SIZE) --out data/v2/bank

v2-fsd50k:
	$(PYTHON) scripts/v2/embed_fsd50k.py --fsd50k "$(FSD50K)" --split dev
	$(PYTHON) scripts/v2/embed_fsd50k.py --fsd50k "$(FSD50K)" --split eval

v2-corpus:
	$(PYTHON) scripts/v2/build_text_corpus.py --fsd50k "$(FSD50K)"
	$(PYTHON) scripts/v2/translate_corpus.py

v2-text:
	$(PYTHON) scripts/v2/train_text_encoder.py

v2-analyze:
	$(PYTHON) scripts/v2/analyze_bank.py

v2-tune:
	$(PYTHON) scripts/v2/tune_retrieval.py

v2-predictor:
	$(PYTHON) scripts/v2/train_predictor.py

v2-tune-hybrid:
	$(PYTHON) scripts/v2/tune_retrieval.py --with-predictor

v2-export:
	$(PYTHON) scripts/v2/export_v2.py

v2-train: v2-bank v2-fsd50k v2-corpus v2-text v2-analyze v2-tune v2-predictor v2-tune-hybrid v2-export

v2-bench:
	$(PYTHON) scripts/v2/benchmark_generate.py
	$(JUDGE_PYTHON) scripts/v2/benchmark_judge.py --fsd50k "$(FSD50K)"

v2-package:
	@test -n "$(V2_TAG)" || (echo "Set V2_TAG, e.g. make v2-package V2_TAG=ai-v2.0.0"; exit 1)
	$(PYTHON) scripts/v2/package_release.py --tag $(V2_TAG)

v2-fetch:
	$(PYTHON) scripts/v2/fetch_model.py
