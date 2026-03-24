import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = BASE_DIR / "benchmarks" / "history.json"

def view_stats():
    if not BENCHMARK_FILE.exists():
        print("No benchmarks found yet. Run train.py first!")
        return

    try:
        with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print(f"Benchmark file is invalid JSON: {BENCHMARK_FILE}")
        return

    print(f"\n{'ID':<5} | {'TIMESTAMP':<20} | {'TIME (s)':<10} | {'EPOCHS':<8} | {'FINAL LOSS':<12} | {'STATUS'}")
    print("-" * 85)

    previous_loss = float('inf')

    for i, entry in enumerate(history):
        loss = entry['final_loss']

        if loss < previous_loss:
            status = "✅ IMPROVED"
        elif loss > previous_loss:
            status = "❌ WORSE"
        else:
            status = "="

        if i == 0: status = "START"

        print(f"{i+1:<5} | {entry['timestamp']:<20} | {entry['duration_seconds']:<10} | {entry['epochs']:<8} | {loss:<12} | {status}")

        previous_loss = loss

    print("-" * 85)
    print(f"Total Training Runs: {len(history)}\n")

if __name__ == "__main__":
    view_stats()
