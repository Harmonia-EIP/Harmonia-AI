import argparse
import json
from pathlib import Path


def load_history(history_path: Path):
    if not history_path.exists():
        print(f"No benchmark history found at {history_path}")
        return []

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print("Benchmark history payload is not a list.")
        return []
    return payload


def print_latest_summary(history_path: Path):
    history = load_history(history_path)
    if not history:
        print("Benchmark history is empty.")
        return

    latest = history[-1]
    report_path = latest.get("evaluation_report_path")
    report = {}
    if report_path and Path(report_path).exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    metrics = report.get("metrics") or latest.get("eval_metrics") or {}

    print("Latest local metrics:")
    print(f"  model_version: {latest.get('model_version', 'n/a')}")
    print(f"  duration_seconds: {latest.get('duration_seconds', 'n/a')}")
    print(f"  final_loss: {latest.get('final_loss', 'n/a')}")
    print(f"  val_mse: {metrics.get('mse', 'n/a')}")
    print(f"  val_mae: {metrics.get('mae', 'n/a')}")
    print(f"  evaluation_report_path: {latest.get('evaluation_report_path', 'n/a')}")


def print_duration_estimate(history_path: Path):
    history = load_history(history_path)
    durations = []
    for item in history:
        value = item.get("duration_seconds")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            durations.append(value)

    if not durations:
        print("No usable duration entries in benchmark history.")
        return

    average = sum(durations) / len(durations)
    recent = durations[-3:]
    recent_average = sum(recent) / len(recent)
    print(f"Average duration over {len(durations)} run(s): {average:.1f}s")
    print(f"Average duration over last {len(recent)} run(s): {recent_average:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print local benchmark metrics summaries.")
    parser.add_argument(
        "--history",
        default="benchmarks/history.json",
        help="Path to benchmark history JSON file.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print duration estimates from benchmark history.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if args.estimate:
        print_duration_estimate(history_path)
    else:
        print_latest_summary(history_path)

