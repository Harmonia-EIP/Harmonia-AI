import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.artifact_registry import archive_artifacts, build_flat_model_paths, resolve_latest_model, write_latest_pointer
from src.charter import CHARTER, DISCRETE_INDICES, PARAM_NAMES
from src.dashboard_events import publish_command, publish_training
from src.dataset import PresetDataset
from src.metrics_publisher import push_metrics_report
from src.model import TextToParams


# Perceptual loss weights: discrete topology choices and the most audible
# continuous controls weigh more than utility params. Keeps the model honest
# about waveform/filter type/cutoff even on small datasets.
CHARTER_LOSS_WEIGHTS = {
    "osc_1_waveform": 2.0,
    "osc_2_waveform": 1.5,
    "osc_mix": 1.0,
    "osc_2_detune": 0.8,
    "noise_level": 0.8,
    "filter_cutoff": 2.0,
    "filter_resonance": 1.2,
    "filter_type": 2.0,
    "amp_attack": 1.5,
    "amp_decay": 1.2,
    "amp_sustain": 1.2,
    "amp_release": 1.5,
    "filter_env_amount": 1.0,
    "filter_env_decay": 1.0,
    "lfo_rate": 0.8,
    "lfo_to_pitch": 0.6,
    "lfo_to_cutoff": 0.8,
    "velocity_to_filter": 0.8,
    "distortion_mix": 1.5,
    "reverb_mix": 1.2,
}


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, default))
        return value if value >= 0 else default
    except ValueError:
        return default


EPOCHS = _env_int("HARMONIA_EPOCHS", 120)
LR = _env_float("HARMONIA_LR", 1e-4)
BATCH_SIZE = _env_int("HARMONIA_BATCH_SIZE", 16)
try:
    SEED = int(os.environ.get("HARMONIA_SEED", "42"))
except ValueError:
    SEED = 42
VAL_SPLIT = _env_float("HARMONIA_VAL_SPLIT", 0.0)
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
TOKENIZER_MAX_LENGTH = 32
MODEL_VERSION_OVERRIDE = os.environ.get("HARMONIA_MODEL_VERSION", "").strip()
DATASET_PATH_OVERRIDE = os.environ.get("HARMONIA_DATASET_PATH", "").strip()
PUSH_METRICS_ON_TRAIN = os.environ.get("HARMONIA_PUSH_METRICS", "1").strip().lower() not in {"0", "false", "no", "off"}
METRICS_PUSH_URL = os.environ.get("HARMONIA_METRICS_URL", os.environ.get("METRICS_URL", "https://harmonia.mcoet.com/receiver.php"))

BENCHMARK_FILE = BASE_DIR / "benchmarks" / "history.json"
EVAL_REPORT_DIR = BASE_DIR / "benchmarks" / "reports"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
HISTORY_METRICS_FILE = SAVED_MODELS_DIR / "history_metrics.json"
ARCHIVE_DIR = SAVED_MODELS_DIR / "archive"
DASHBOARD_DIR = BASE_DIR / "metrics_dashboard"
DASHBOARD_LATEST_FILE = DASHBOARD_DIR / "latest_metrics.json"
DASHBOARD_HISTORY_FILE = DASHBOARD_DIR / "history_metrics.json"
LEGACY_MODEL_SAVE_PATH = SAVED_MODELS_DIR / "my_plugin_ai.pth"
LEGACY_METADATA_PATH = SAVED_MODELS_DIR / "my_plugin_ai.meta.json"
MODEL_FILE_PREFIX = "model_"


def set_seed(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_json_list(path):
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def save_benchmark(duration, final_loss, epoch_history, dataset_size, train_size=None, val_size=None, eval_metrics=None, evaluation_report_path=None, model_version=None, model_hash=None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(float(duration), 2),
        "final_loss": round(float(final_loss), 6),
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
        "loss_history": epoch_history,
    }
    history = _load_json_list(BENCHMARK_FILE)
    history.append(entry)
    _write_json(BENCHMARK_FILE, history)


def _append_history_metrics(entry):
    history = _load_json_list(HISTORY_METRICS_FILE)
    history.append(entry)
    if len(history) > 500:
        history = history[-500:]
    _write_json(HISTORY_METRICS_FILE, history)
    return history


def _sync_dashboard_metrics(latest_report_payload, history_payload):
    _write_json(DASHBOARD_LATEST_FILE, latest_report_payload)
    _write_json(DASHBOARD_HISTORY_FILE, history_payload)


def compute_split_sizes(dataset_size, val_ratio):
    if dataset_size < 2 or val_ratio <= 0:
        return dataset_size, 0
    val_size = max(1, min(dataset_size - 1, int(dataset_size * min(0.9, val_ratio))))
    return dataset_size - val_size, val_size


def evaluate_model(model, loader, param_keys, continuous_count=0, binary_count=0, device=None):
    if loader is None or not param_keys:
        return None
    total = len(param_keys)
    continuous_count = max(0, min(int(continuous_count), total))
    binary_count = max(0, min(int(binary_count), total - continuous_count))
    if continuous_count + binary_count == 0:
        continuous_count = total

    model.eval()
    mse_sum = torch.zeros(total, dtype=torch.float32)
    mae_sum = torch.zeros(total, dtype=torch.float32)
    cont_sum = 0.0
    bin_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device) if device else batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device) if device else batch["attention_mask"]
            labels = (batch["labels"].to(device) if device else batch["labels"]).float()[:, :total]
            preds = model(input_ids, attention_mask)[:, :total]
            err = preds - labels
            mse_sum += (err * err).sum(dim=0).cpu()
            mae_sum += err.abs().sum(dim=0).cpu()
            if continuous_count:
                cont_sum += float((err[:, :continuous_count] ** 2).sum().cpu())
            if binary_count:
                bp = preds[:, continuous_count:continuous_count + binary_count].clamp(1e-6, 1 - 1e-6)
                bl = labels[:, continuous_count:continuous_count + binary_count]
                bin_sum += float(F.binary_cross_entropy(bp, bl, reduction="sum").cpu())
            count += labels.shape[0]

    if count == 0:
        return None

    per_param_mse = (mse_sum / count).tolist()
    per_param_mae = (mae_sum / count).tolist()
    continuous_mse = cont_sum / (count * continuous_count) if continuous_count else None
    binary_bce = bin_sum / (count * binary_count) if binary_count else None
    loss = (continuous_mse or 0.0) + (binary_bce or 0.0)
    if loss == 0.0:
        loss = float(sum(per_param_mse) / len(per_param_mse))

    return {
        "loss": round(loss, 6),
        "final_loss": round(loss, 6),
        "mse": round(float(sum(per_param_mse) / len(per_param_mse)), 6),
        "mae": round(float(sum(per_param_mae) / len(per_param_mae)), 6),
        "continuous_mse": round(continuous_mse, 6) if continuous_mse is not None else None,
        "binary_bce": round(binary_bce, 6) if binary_bce is not None else None,
        "per_param_mse": {k: round(float(v), 6) for k, v in zip(param_keys, per_param_mse)},
        "per_param_mae": {k: round(float(v), 6) for k, v in zip(param_keys, per_param_mae)},
        "sample_count": count,
    }


def write_evaluation_report(report_payload):
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_REPORT_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(path, report_payload)
    return path


def compute_file_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def prompt_model_name():
    if not sys.stdin.isatty():
        return ""
    try:
        return input("Nom du modèle (laisser vide pour automatique) : ").strip()
    except EOFError:
        return ""


def sanitize_model_name(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", value.strip())
    return sanitized.strip(".-")

def resolve_model_version():
    if MODEL_VERSION_OVERRIDE:
        return sanitize_model_name(MODEL_VERSION_OVERRIDE)

    custom_name = prompt_model_name()
    timestamp = datetime.now().strftime("%d-%m-%y")

    if custom_name:
        safe_name = sanitize_model_name(custom_name)
        if safe_name:
            return f"{MODEL_FILE_PREFIX}{safe_name}_{timestamp}"

    return f"{MODEL_FILE_PREFIX}{timestamp}"

def ensure_unique_model_version(model_version):
    candidate = model_version
    suffix = 1
    while (
        (SAVED_MODELS_DIR / candidate).exists()
        or (SAVED_MODELS_DIR / f"{candidate}.pth").exists()
        or (SAVED_MODELS_DIR / f"{candidate}.meta.json").exists()
    ):
        suffix += 1
        candidate = f"{model_version}-r{suffix}"
    return candidate


def write_model_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return path


def maybe_push_metrics(eval_report_path):
    if not PUSH_METRICS_ON_TRAIN:
        print("Metrics push disabled (HARMONIA_PUSH_METRICS=0).")
        return

    report_path = Path(eval_report_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[METRICS] Could not read report for dashboard event: {exc}")
        report = {}

    dashboard_result = publish_training(report, base_dir=BASE_DIR) if report else {"skipped": True, "reason": "missing-report"}
    if dashboard_result.get("ok"):
        print(f"[DASHBOARD] Training event pushed (HTTP {dashboard_result.get('status')}).")
    elif dashboard_result.get("skipped"):
        print(f"[DASHBOARD] {dashboard_result.get('reason', 'skipped')}")
    else:
        print(f"[DASHBOARD] push error: {dashboard_result.get('error', dashboard_result.get('status'))}")

    result = push_metrics_report(report_path, url=METRICS_PUSH_URL, base_dir=BASE_DIR)
    if result.get("skipped"):
        print(f"[METRICS] {result.get('reason', 'skipped')}")
        return
    status = result.get("status", "n/a")
    if result.get("ok"):
        print(f"[METRICS] Pushed evaluation report to {result.get('url')} (HTTP {status})")
    else:
        print(f"[METRICS] Push failed to {result.get('url')} (HTTP {status})")
        if result.get("response"):
            print(result["response"])


def resolve_dataset_path():
    if DATASET_PATH_OVERRIDE:
        return Path(DATASET_PATH_OVERRIDE)
    npy_candidate = BASE_DIR / "data" / "processed" / "presets.npy"
    if npy_candidate.exists():
        return npy_candidate
    return BASE_DIR / "data" / "processed" / "presets.json"


def train():
    start_time = time.time()
    set_seed(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615
    dataset_path = resolve_dataset_path()
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found. Run prepare_dataset.py first.")
        return

    dataset = PresetDataset(
        dataset_path,
        tokenizer,
        tokenizer_max_length=TOKENIZER_MAX_LENGTH,
        normalize_categorical=True,
        prompt_augment=True,
    )
    if len(dataset) == 0:
        print(f"Error: {dataset_path} is empty. Add presets before training.")
        return

    training_keys = list(getattr(dataset, "training_param_keys", [])) or list(dataset.param_keys)
    if not training_keys:
        print(f"Error: no trainable parameter keys detected in {dataset_path}.")
        return

    charter_mode = bool(getattr(dataset, "charter_mode", False))
    if charter_mode:
        continuous_count = len(training_keys)
        binary_count = 0
        loss_weights = torch.tensor(
            [CHARTER_LOSS_WEIGHTS.get(name, 1.0) for name in training_keys],
            dtype=torch.float32,
        )
    else:
        continuous_count = len(getattr(dataset, "_continuous_keys", []))
        binary_count = len(getattr(dataset, "_binary_keys", []))
        if continuous_count + binary_count == 0:
            continuous_count = len(training_keys)
            binary_count = 0
        loss_weights = None

    train_size, val_size = compute_split_sizes(len(dataset), VAL_SPLIT)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset is not None else None
    reference_loader = val_loader if val_loader is not None else DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    model = TextToParams(num_plugin_parameters=len(training_keys)).to(device)
    if loss_weights is not None:
        loss_weights = loss_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))

    print(f"Starting training on {len(dataset)} presets (train={train_size}, val={val_size}, params={len(training_keys)}, device={device})...")

    loss_history = []
    best_monitor_loss = None
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()[:, : len(training_keys)]
            preds = model(input_ids, attention_mask)[:, : len(training_keys)]

            loss = 0.0
            if charter_mode and loss_weights is not None:
                err = (preds - labels) ** 2
                loss = loss + (err * loss_weights).mean()
            else:
                if continuous_count > 0:
                    loss = loss + F.mse_loss(preds[:, :continuous_count], labels[:, :continuous_count])
                if binary_count > 0:
                    bp = preds[:, continuous_count:continuous_count + binary_count].clamp(1e-6, 1 - 1e-6)
                    loss = loss + F.binary_cross_entropy(bp, labels[:, continuous_count:continuous_count + binary_count])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        loss_history.append(avg_loss)
        scheduler.step()
        current_monitor_loss = avg_loss
        if val_loader is not None:
            metrics = evaluate_model(model, val_loader, training_keys, continuous_count, binary_count, device=device)
            if metrics:
                current_monitor_loss = metrics.get("loss", avg_loss)
        if best_monitor_loss is None or current_monitor_loss < best_monitor_loss:
            best_monitor_loss = current_monitor_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
            print(f"Epoch {epoch + 1}: train_loss = {avg_loss:.6f}")

    duration = time.time() - start_time
    model_version = ensure_unique_model_version(resolve_model_version())
    artifact_paths = build_flat_model_paths(SAVED_MODELS_DIR, model_version)
    model_version = artifact_paths["model_version"]
    model_path = artifact_paths["model_path"]
    metadata_path = artifact_paths["metadata_path"]
    eval_file_path = artifact_paths["evaluation_report_path"]

    previous_artifact = resolve_latest_model(SAVED_MODELS_DIR, legacy_model_path=LEGACY_MODEL_SAVE_PATH, legacy_metadata_path=LEGACY_METADATA_PATH)
    previous_model_path = previous_artifact.get("model_path") if previous_artifact else None
    if previous_model_path and Path(previous_model_path) != model_path:
        archived = archive_artifacts(ARCHIVE_DIR / str(previous_artifact.get("model_version", "legacy")), previous_artifact)
        if archived:
            print(f"Archived previous model to {ARCHIVE_DIR / str(previous_artifact.get('model_version', 'legacy'))}")

    selected_state = best_state_dict if best_state_dict is not None else model.state_dict()
    torch.save({k: v.cpu() for k, v in selected_state.items()}, model_path)
    model_hash = compute_file_sha256(model_path)

    eval_metrics = evaluate_model(model, reference_loader, training_keys, continuous_count, binary_count, device=device)
    if eval_metrics is None:
        fallback_loss = round(float(loss_history[-1] if loss_history else 0.0), 6)
        eval_metrics = {"loss": fallback_loss, "final_loss": fallback_loss, "mse": 0.0, "mae": 0.0, "continuous_mse": None, "binary_bce": None, "per_param_mse": {}, "per_param_mae": {}, "sample_count": 0}

    eval_report_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": model_version,
        "model_hash": model_hash,
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "train_size": train_size,
        "val_size": val_size,
        "param_keys": training_keys,
        "metrics": eval_metrics,
        "loss_history": loss_history,
    }
    eval_report_written = write_evaluation_report(eval_report_payload)
    if eval_file_path != eval_report_written:
        _write_json(eval_file_path, eval_report_payload)

    history_entry = {
        "timestamp": eval_report_payload["timestamp"],
        "model_version": model_version,
        "model_hash": model_hash,
        "loss": eval_metrics.get("loss"),
        "final_loss": eval_metrics.get("final_loss", eval_metrics.get("loss")),
        "mse": eval_metrics.get("mse"),
        "mae": eval_metrics.get("mae"),
        "continuous_mse": eval_metrics.get("continuous_mse"),
        "binary_bce": eval_metrics.get("binary_bce"),
        "metrics": eval_metrics,
        "loss_history": loss_history,
        "evaluation_report_path": str(eval_report_written),
    }
    history_metrics = _append_history_metrics(history_entry)
    _sync_dashboard_metrics(eval_report_payload, history_metrics)

    metadata_payload = {
        "model_path": str(model_path),
        "model_version": model_version,
        "model_hash": model_hash,
        "plugin_param_count": len(training_keys),
        "param_keys": training_keys,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "val_split": VAL_SPLIT,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "continuous_param_count": continuous_count,
        "binary_param_count": binary_count,
        "tokenizer_model_id": TOKENIZER_MODEL_ID,
        "tokenizer_model_revision": TOKENIZER_MODEL_REVISION,
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "dataset_path": str(dataset_path),
        "normalize_categorical": True,
        "evaluation_report_path": str(eval_report_written),
        "charter_mode": charter_mode,
        "charter_version": "1.0" if charter_mode else None,
    }
    write_model_metadata(metadata_path, metadata_payload)
    write_latest_pointer(SAVED_MODELS_DIR, {"model_version": model_version, "model_path": str(model_path), "metadata_path": str(metadata_path), "evaluation_report_path": str(eval_report_written), "model_hash": model_hash, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    write_model_metadata(LEGACY_METADATA_PATH, metadata_payload)
    maybe_push_metrics(eval_report_written)
    save_benchmark(duration, loss_history[-1] if loss_history else 0.0, loss_history, len(dataset), train_size=train_size, val_size=val_size, eval_metrics=eval_metrics, evaluation_report_path=eval_report_written, model_version=model_version, model_hash=model_hash)
    publish_command(
        "train.py",
        status="ok",
        duration_seconds=duration,
        detail={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dataset_size": len(dataset),
            "charter_mode": charter_mode,
        },
        model_version=model_version,
        base_dir=BASE_DIR,
    )
    print(f"Model saved to {model_path}!")
    print(f"Model metadata saved to {metadata_path}")
    print(f"Evaluation report saved to {eval_report_written}")
    print(f"History metrics saved to {HISTORY_METRICS_FILE}")


if __name__ == "__main__":
    train()

