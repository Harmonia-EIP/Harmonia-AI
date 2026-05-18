#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.metrics_publisher import latest_report_from_history, push_metrics_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Push the latest Harmonia evaluation metrics to the web receiver.")
    parser.add_argument("--report", type=Path, default=None, help="Explicit evaluation report JSON file.")
    parser.add_argument("--history", type=Path, default=BASE_DIR / "benchmarks" / "history.json", help="Benchmark history JSON file.")
    parser.add_argument("--url", default=None, help="Metrics receiver URL.")
    parser.add_argument("--token", default=None, help="Security token.")
    args = parser.parse_args()

    report_path = args.report or latest_report_from_history(args.history)
    if report_path is None:
        print("No evaluation report found; skipping metrics push.")
        return 0

    result = push_metrics_report(report_path, url=args.url, token=args.token, base_dir=BASE_DIR)
    if result.get("skipped"):
        print(result.get("reason", "metrics push skipped"))
        return 0

    print(f"Metrics pushed to {result.get('url')} (HTTP {result.get('status', 'n/a')})")
    if result.get("response"):
        print(result["response"])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

