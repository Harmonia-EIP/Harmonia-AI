import torch
import json
import time
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import TextToParams
from dataset import PresetDataset

# --- CONFIG ---
PLUGIN_PARAM_COUNT = 9
EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 8
DATASET_PATH = "../data/processed/presets.json"
BENCHMARK_FILE = "../benchmarks/history.json"
MODEL_SAVE_PATH = "../saved_models/my_plugin_ai.pth"
# --- LOGGING FUNCTION ---
def save_benchmark(duration, final_loss, epoch_history):
    if not os.path.exists("../model/benchmarks"):
        os.makedirs("../model/benchmarks")

    # 2. Prepere new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2),
        "final_loss": round(final_loss, 6),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "loss_history": epoch_history # Saving curve graph
    }

    # 3. Load existing history
    history = []
    if os.path.exists(BENCHMARK_FILE):
        try:
            with open(BENCHMARK_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []

    # 4. Append and Save
    history.append(entry)
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\n[BENCHMARK] Stats saved to {BENCHMARK_FILE}")
    print(f"Time: {entry['duration_seconds']}s | Final Loss: {entry['final_loss']}")

# --- TRAINING LOOP ---
def train():
    # Start Timer
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found. Run prepare_dataset.py first.")
        return

    dataset = PresetDataset(DATASET_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    if not os.path.exists("../saved_models"):
        os.makedirs("../saved_models")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}!")

    # Save Benchmark Stats
    save_benchmark(duration, loss_history[-1], loss_history)

if __name__ == "__main__":
    train()
