import torch
import json
import os
import random
import time
import sys
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
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

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 8
SEED = int(os.environ.get("HARMONIA_SEED", "42"))
TOKENIZER_MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
TOKENIZER_MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")
DATASET_PATH = BASE_DIR / "data" / "processed" / "presets.json"
BENCHMARK_FILE = BASE_DIR / "benchmarks" / "history.json"
MODEL_SAVE_PATH = BASE_DIR / "saved_models" / "my_plugin_ai.pth"


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
def save_benchmark(duration, final_loss, epoch_history, dataset_size):
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

# --- TRAINING LOOP ---
def train():
    # Start Timer
    start_time = time.time()
    set_seed(SEED)
    print(f"Using deterministic seed: {SEED}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_MODEL_REVISION)  # nosec B615

    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"Error: {DATASET_PATH} not found. Run prepare_dataset.py first.")
        return

    dataset = PresetDataset(DATASET_PATH, tokenizer)
    if len(dataset) == 0:
        print(f"Error: {DATASET_PATH} is empty. Add presets before training.")
        return

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = TextToParams(num_plugin_parameters=PLUGIN_PARAM_COUNT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    print(f"Starting training on {len(dataset)} presets...")

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            preds = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(preds, batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # End Timer
    end_time = time.time()
    duration = end_time - start_time

    # Save Model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}!")

    # Save Benchmark Stats
    final_loss = loss_history[-1] if loss_history else 0.0
    save_benchmark(duration, final_loss, loss_history, len(dataset))

if __name__ == "__main__":
    train()
