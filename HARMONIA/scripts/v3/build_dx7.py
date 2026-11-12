#!/usr/bin/env python3
"""Compile the DX7 renderer (native/dx7 + Dexed's msfa FM core) into a shared library for ctypes.

    python scripts/v3/fetch_presets.py --only dexed_src
    python scripts/v3/build_dx7.py              # -> data/v3/build/dx7/libharmonia_dx7.{dylib,so}
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404 - fixed compiler invocation
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.fetch_presets import DEFAULT_OUT  # noqa: E402

NATIVE_DIR = BASE_DIR / "native" / "dx7"
BUILD_DIR = BASE_DIR / "data" / "v3" / "build" / "dx7"
MSFA_SOURCES = ["dx7note.cc", "env.cc", "exp2.cc", "fm_core.cc", "fm_op_kernel.cc", "freqlut.cc", "lfo.cc",
                "pitchenv.cc", "porta.cpp", "sin.cc"]
LIBRARY = BUILD_DIR / ("libharmonia_dx7.dylib" if sys.platform == "darwin" else "libharmonia_dx7.so")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msfa", type=Path, default=DEFAULT_OUT / "dexed_src" / "Source" / "msfa")
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()

    msfa = BUILD_DIR / "msfa"  # msfa includes "../Dexed.h": keep the shims one level above the sources
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    shutil.copytree(args.msfa, msfa)
    for shim in (NATIVE_DIR / "shim").iterdir():
        shutil.copy(shim, BUILD_DIR / shim.name)
    command = [args.cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-DNDEBUG", f"-I{msfa}", f"-I{BUILD_DIR}",
               "-o", str(LIBRARY), str(NATIVE_DIR / "dx7_render.cpp"), *[str(msfa / name) for name in MSFA_SOURCES]]
    subprocess.run(command, check=True)  # nosec B603 - fixed arguments
    print(f"built {LIBRARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
