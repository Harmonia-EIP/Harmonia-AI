#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model metrics JSON to receiver.php")
    parser.add_argument("json_file", type=Path, help="Path to eval report JSON file")
    parser.add_argument(
        "--url",
        default=os.environ.get("METRICS_URL", "https://harmonia.mcoet.com/receiver.php"),
        help="Receiver URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("METRICS_TOKEN", ""),
        help="Security token (or set METRICS_TOKEN)",
    )
    args = parser.parse_args()

    if not args.json_file.exists():
        raise SystemExit(f"File not found: {args.json_file}")
    if not args.token:
        raise SystemExit("Missing token. Use --token or METRICS_TOKEN env var or set METRICS_TOKEN in your environment.")

    with args.json_file.open("rb") as f:
        files = {
            "metrics_file": (args.json_file.name, f, "application/json"),
        }
        headers = {"Authorization": f"Bearer {args.token}"}
        response = requests.post(args.url, files=files, headers=headers, timeout=30)

    print(f"HTTP {response.status_code}")
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()

