#!/usr/bin/env python3
"""Flush the local dashboard event buffer to the remote receiver.

All `publish_*` calls in Harmonia write JSON events to
``metrics_dashboard/events/`` instead of pushing immediately. This script is
invoked by ``make push-metrics`` and ships everything in one batch:
events are POSTed one by one, successful uploads are deleted from the buffer,
and an up-to-date snapshot is pushed at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.dashboard_events import flush_pending_events, publish_system  # noqa: E402


def _build_snapshot() -> dict | None:
    try:
        from scripts.dashboard_stats import build_snapshot
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[snapshot] Skipping snapshot build: {exc}")
        return None
    try:
        return build_snapshot()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[snapshot] build_snapshot() failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Push buffered Harmonia dashboard events.")
    parser.add_argument("--url", default=None, help="Override receiver URL.")
    parser.add_argument("--token", default=None, help="Override METRICS_TOKEN.")
    parser.add_argument("--keep", action="store_true", help="Do not delete events after successful upload.")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip the trailing system snapshot push.")
    args = parser.parse_args()

    summary = flush_pending_events(
        url=args.url,
        token=args.token,
        base_dir=BASE_DIR,
        delete_on_success=not args.keep,
    )

    pending = summary.get("pending", 0)
    succeeded = summary.get("succeeded", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)

    print(f"[push-metrics] buffer: {summary.get('buffer_dir')}")
    print(f"[push-metrics] pending={pending} succeeded={succeeded} failed={failed} skipped={skipped}")
    if summary.get("reason"):
        print(f"[push-metrics] reason: {summary['reason']}")
    for failure in summary.get("failures", [])[:10]:
        print(f"  - {failure}")

    if not args.no_snapshot:
        snapshot = _build_snapshot()
        if snapshot:
            snap_result = publish_system(snapshot, url=args.url, token=args.token, base_dir=BASE_DIR)
            if snap_result.get("buffered"):
                # publish_system also defers; flush that single new file too.
                flush2 = flush_pending_events(
                    url=args.url,
                    token=args.token,
                    base_dir=BASE_DIR,
                    delete_on_success=not args.keep,
                )
                print(f"[push-metrics] snapshot flushed: succeeded={flush2.get('succeeded', 0)} failed={flush2.get('failed', 0)}")

    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
