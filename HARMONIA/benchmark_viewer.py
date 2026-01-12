import json
import os

BENCHMARK_FILE = "benchmarks/history.json"

def view_stats():
    if not os.path.exists(BENCHMARK_FILE):
        print("No benchmarks found yet. Run train.py first!")
        return

    with open(BENCHMARK_FILE, 'r') as f:
        history = json.load(f)

    print(f"\n{'ID':<5} | {'TIMESTAMP':<20} | {'TIME (s)':<10} | {'EPOCHS':<8} | {'FINAL LOSS':<12} | {'STATUS'}")
    print("-" * 85)

    previous_loss = float('inf')

    for i, entry in enumerate(history):
        loss = entry['final_loss']

        # Calculate improvement
        if loss < previous_loss:
            status = "✅ IMPROVED"
        elif loss > previous_loss:
            status = "❌ WORSE"
        else:
            status = "="

        # Don't show status for the very first run
        if i == 0: status = "START"

        print(f"{i+1:<5} | {entry['timestamp']:<20} | {entry['duration_seconds']:<10} | {entry['epochs']:<8} | {loss:<12} | {status}")

        previous_loss = loss

    print("-" * 85)
    print(f"Total Training Runs: {len(history)}\n")

if __name__ == "__main__":
    view_stats()
