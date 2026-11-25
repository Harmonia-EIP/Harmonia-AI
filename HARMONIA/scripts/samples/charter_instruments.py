#!/usr/bin/env python3
"""Keep the charter's sample source in step with the catalogue of sampled instruments (#42).

The charter (charter/harmonia_v3.json) is the single source shared with the plugin. Its `sample_instrument`
choice lists the instruments of samples/instruments.json, index 0 being "Off". A preset stores the index, so
the list is append-only: an instrument keeps its index forever, a new one is added at the end, one that left
the catalogue keeps its place (marked missing) so that no preset changes instrument.

    python scripts/samples/charter_instruments.py           # update the charter
    python scripts/samples/charter_instruments.py --check   # exit 1 if it is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CHARTER = BASE_DIR / "charter" / "harmonia_v3.json"
INSTRUMENTS = BASE_DIR / "samples" / "instruments.json"
OFF = "Off"


def updated(charter: dict, catalogue: dict) -> dict:
    block = charter.setdefault("sample_instruments", {})
    ids = list(block.get("ids", []))
    known = {i["id"]: i for i in catalogue["instruments"]}
    for instrument in catalogue["instruments"]:
        if instrument["id"] not in ids:
            ids.append(instrument["id"])
    block.update({
        "description": "Index i >= 1 of sample_instrument plays ids[i - 1] (samples/instruments.json). Append only: "
                       "an index never changes instrument. Regenerate with scripts/samples/charter_instruments.py.",
        "ids": ids,
        "missing": [i for i in ids if i not in known],
    })
    names = [OFF] + [known[i]["name"] if i in known else f"(missing) {i}" for i in ids]
    for param in charter["engines"]["analog"]["parameters"]:
        if param["id"] == "sample_instrument":
            param["choices"] = names
            param["max"] = len(names) - 1
    return charter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    before = CHARTER.read_text(encoding="utf-8")
    charter = updated(json.loads(before), json.loads(INSTRUMENTS.read_text(encoding="utf-8")))
    after = json.dumps(charter, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if after != before:
            print("charter/harmonia_v3.json is out of date: run scripts/samples/charter_instruments.py")
            return 1
        return 0
    CHARTER.write_text(after, encoding="utf-8")
    print(f"{len(charter['sample_instruments']['ids'])} instruments in the charter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
