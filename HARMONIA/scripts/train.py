import torch
import json
import hashlib
import os
import random
import time
import sys
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.model import TextToParams
from src.dataset import PresetDataset
from src.artifact_registry import build_versioned_paths, write_latest_pointer

# --- CONFIG ---
def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Warning: {name} must be a positive integer; using default {default}.", file=sys.stderr)
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Warning: {name} must be a positive float; using default {default}.", file=sys.stderr)
        return default


EPOCHS = _env_int("HARMONIA_EPOCHS", 100)
LR = _env_float("HARMONIA_LR", 1e-4)
BATCH_SIZE = _env_int("HARMONIA_BATCH_SIZE", 8)
try:
    SEED = int(os.environ.get("HARMONIA_SEED", "42"))
except ValueError:
    print("Warning: HARMONIA_SEED must be an integer; using default 42.", file=sys.stderr)
    SEED = 42
try:
    VAL_SPLIT = float(os.environ.get("HARMONIA_VAL_SPLIT", "0.2"))
except ValueError:
    print("Warning: HARMONIA_VAL_SPLIT must be a float; using default 0.2.", file=sys.stderr)
    VAL_SPLIT = 0.2
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
TOKENIZER_MAX_LENGTH = 32
MODEL_VERSION_OVERRIDE = os.environ.get("HARMONIA_MODEL_VERSION", "").strip()
DATASET_PATH_OVERRIDE = os.environ.get("HARMONIA_DATASET_PATH", "").strip()
BENCHMARK_FILE = BASE_DIR / "benchmarks" / "history.json"
EVAL_REPORT_DIR = BASE_DIR / "benchmarks" / "reports"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
LEGACY_MODEL_SAVE_PATH = SAVED_MODELS_DIR / "my_plugin_ai.pth"
LEGACY_METADATA_PATH = SAVED_MODELS_DIR / "my_plugin_ai.meta.json"


def set_seed(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- LOGGING FUNCTION ---
def save_benchmark(
    duration,
    final_loss,
    epoch_history,
    dataset_size,
    train_size=None,
    val_size=None,
    eval_metrics=None,
    evaluation_report_path=None,
    model_version=None,
    model_hash=None,
):
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Prepere new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2),
        "final_loss": round(final_loss, 6),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "dataset_size": dataset_size,
        "train_size": train_size,
        "val_size": val_size,
        "eval_metrics": eval_metrics,
        "evaluation_report_path": str(evaluation_report_path) if evaluation_report_path else None,
        "model_version": model_version,
        "model_hash": model_hash,
        "loss_history": epoch_history # Saving curve graph
    }

    # 3. Load existing history
    history = []
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

    # 4. Append and Save
    history.append(entry)
    with open(BENCHMARK_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    print(f"\n[BENCHMARK] Stats saved to {BENCHMARK_FILE}")
    print(f"Time: {entry['duration_seconds']}s | Final Loss: {entry['final_loss']}")


def compute_split_sizes(dataset_size, val_ratio):
    if dataset_size < 2:
        return dataset_size, 0

    safe_ratio = min(0.9, max(0.0, val_ratio))
    val_size = max(1, int(dataset_size * safe_ratio))
    val_size = min(val_size, dataset_size - 1)
    return dataset_size - val_size, val_size


def evaluate_model(model, loader, param_keys):
    if loader is None:
        return None

    if not param_keys:
        return None

    model.eval()
    mse_sum = torch.zeros(len(param_keys), dtype=torch.float32)
    mae_sum = torch.zeros(len(param_keys), dtype=torch.float32)
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            preds = model(batch['input_ids'], batch['attention_mask'])
            labels = batch['labels']

            error = preds - labels
            mse_sum += (error * error).sum(dim=0).cpu()
            mae_sum += error.abs().sum(dim=0).cpu()
            sample_count += labels.shape[0]

    if sample_count == 0:
        return None

    per_param_mse = (mse_sum / sample_count).tolist()
    per_param_mae = (mae_sum / sample_count).tolist()
    return {
        "mse": round(float(sum(per_param_mse) / len(per_param_mse)), 6),
        "mae": round(float(sum(per_param_mae) / len(per_param_mae)), 6),
        "per_param_mse": {k: round(float(v), 6) for k, v in zip(param_keys, per_param_mse)},
        "per_param_mae": {k: round(float(v), 6) for k, v in zip(param_keys, per_param_mae)},
        "sample_count": sample_count,
    }


def write_evaluation_report(report_payload):
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = EVAL_REPORT_DIR / report_name
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=4)
    return report_path


def compute_file_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def resolve_model_version():
    if MODEL_VERSION_OVERRIDE:
        return MODEL_VERSION_OVERRIDE
    return f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{SEED}"


def ensure_unique_model_version(model_version):
    candidate = model_version
    suffix = 1
    while (SAVED_MODELS_DIR / candidate).exists():
        suffix += 1
        candidate = f"{model_version}-r{suffix}"
    return candidate


def write_model_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)
    return path


def resolve_dataset_path():
    if DATASET_PATH_OVERRIDE:
        return Path(DATASET_PATH_OVERRIDE)

    npy_candidate = BASE_DIR / "data" / "processed" / "presets.npy"
    if npy_candidate.exists():
        return npy_candidate

    return BASE_DIR / "data" / "processed" / "presets.json"

# --- TRAINING LOOP ---
def train():
    # Start Timer
    start_time = time.time()
    set_seed(SEED)
    print(f"Using deterministic seed: {SEED}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615
    dataset_path = resolve_dataset_path()

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found. Run prepare_dataset.py first.")
        return

    dataset = PresetDataset(
        dataset_path,
        tokenizer,
        tokenizer_max_length=TOKENIZER_MAX_LENGTH,
        normalize_categorical=True,
    )
    if len(dataset) == 0:
        print(f"Error: {dataset_path} is empty. Add presets before training.")
        return

    param_keys = list(dataset.param_keys)
    if not param_keys:
        print(f"Error: no parameter keys detected in {dataset_path}.")
        return
    plugin_param_count = len(param_keys)

    train_size, val_size = compute_split_sizes(len(dataset), VAL_SPLIT)
    split_generator = torch.Generator().manual_seed(SEED)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    else:
        train_dataset, val_dataset = dataset, None

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset is not None else None

    model = TextToParams(num_plugin_parameters=plugin_param_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    print(
        f"Starting training on {len(dataset)} presets "
        f"(train={train_size}, val={val_size}, params={plugin_param_count})..."
    )

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(preds, batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # End Timer
    end_time = time.time()
    duration = end_time - start_time

    model_version = ensure_unique_model_version(resolve_model_version())
    versioned_paths = build_versioned_paths(SAVED_MODELS_DIR, model_version)

    # Save model in a versioned directory
    versioned_paths["model_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), versioned_paths["model_path"])
    print(f"Model saved to {versioned_paths['model_path']}!")

    model_hash = compute_file_sha256(versioned_paths["model_path"])

    eval_metrics = evaluate_model(model, val_loader, param_keys)
    eval_report_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": model_version,
        "model_hash": model_hash,
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "train_size": train_size,
        "val_size": val_size,
        "metrics": eval_metrics,
    }
    eval_report_path = write_evaluation_report(eval_report_payload)

    metadata_payload = {
        "model_path": str(versioned_paths["model_path"]),
        "model_dir": str(versioned_paths["model_dir"]),
        "model_version": model_version,
        "model_hash": model_hash,
        "plugin_param_count": plugin_param_count,
        "param_keys": param_keys,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "val_split": VAL_SPLIT,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "tokenizer_model_revision": TOKENIZER_MODEL_REVISION,
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "dataset_path": str(dataset_path),
        "normalize_categorical": True,
        "evaluation_report_path": str(eval_report_path),
    }
    metadata_path = write_model_metadata(versioned_paths["metadata_path"], metadata_payload)
    write_latest_pointer(
        SAVED_MODELS_DIR,
        {
            "model_version": model_version,
            "model_path": str(versioned_paths["model_path"]),
            "metadata_path": str(metadata_path),
            "model_hash": model_hash,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Backward-compatible metadata mirror for legacy tooling.
    write_model_metadata(LEGACY_METADATA_PATH, metadata_payload)
    print(f"Model metadata saved to {metadata_path}")
    print(f"Evaluation report saved to {eval_report_path}")

    # Save Benchmark Stats
    final_loss = loss_history[-1] if loss_history else 0.0
    save_benchmark(
        duration,
        final_loss,
        loss_history,
        len(dataset),
        train_size=train_size,
        val_size=val_size,
        eval_metrics=eval_metrics,
        evaluation_report_path=eval_report_path,
        model_version=model_version,
        model_hash=model_hash,
    )

if __name__ == "__main__":
    train()
