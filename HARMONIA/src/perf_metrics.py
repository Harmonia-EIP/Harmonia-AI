"""Lightweight performance metrics sampler.

Captures CPU / RAM / disk / Apple-Silicon (MPS) GPU usage so dashboard events
can carry a `system_metrics` field. Designed to never raise: any sampling
error returns a best-effort partial dict.

A `PerfSampler` context manager takes periodic samples on a background thread,
suitable for collecting a time-series during long-running training jobs.
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / (1024 * 1024), 2)


def _safe_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _mps_memory_mb() -> Dict[str, Optional[float]]:
    try:
        import torch  # type: ignore
    except Exception:
        return {"available": False}
    try:
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            return {"available": False}
    except Exception:
        return {"available": False}

    info: Dict[str, Any] = {"available": True}
    try:
        info["allocated_mb"] = _mb(torch.mps.current_allocated_memory())
    except Exception:
        info["allocated_mb"] = None
    try:
        info["driver_allocated_mb"] = _mb(torch.mps.driver_allocated_memory())
    except Exception:
        info["driver_allocated_mb"] = None
    try:
        recommended = torch.mps.recommended_max_memory()
        info["recommended_max_mb"] = _mb(recommended)
    except Exception:
        info["recommended_max_mb"] = None
    return info


def _cpu_ram(psutil_mod) -> Dict[str, Any]:
    if psutil_mod is None:
        try:
            load = os.getloadavg()
            return {"load_avg_1m": round(load[0], 2)}
        except (AttributeError, OSError):
            return {}
    out: Dict[str, Any] = {}
    try:
        out["cpu_percent"] = psutil_mod.cpu_percent(interval=None)
    except Exception:  # nosec B110 - perf sampling must never raise
        pass
    try:
        out["cpu_count_logical"] = psutil_mod.cpu_count(logical=True)
        out["cpu_count_physical"] = psutil_mod.cpu_count(logical=False)
    except Exception:  # nosec B110 - perf sampling must never raise
        pass
    try:
        vm = psutil_mod.virtual_memory()
        out["ram_total_mb"] = _mb(vm.total)
        out["ram_used_mb"] = _mb(vm.used)
        out["ram_available_mb"] = _mb(vm.available)
        out["ram_percent"] = vm.percent
    except Exception:  # nosec B110 - perf sampling must never raise
        pass
    try:
        proc = psutil_mod.Process(os.getpid())
        out["proc_rss_mb"] = _mb(proc.memory_info().rss)
        out["proc_cpu_percent"] = proc.cpu_percent(interval=None)
    except Exception:  # nosec B110 - perf sampling must never raise
        pass
    try:
        load = os.getloadavg()
        out["load_avg_1m"] = round(load[0], 2)
        out["load_avg_5m"] = round(load[1], 2)
        out["load_avg_15m"] = round(load[2], 2)
    except (AttributeError, OSError):
        pass
    return out


def _disk_info(path: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "disk_total_mb": _mb(usage.total),
            "disk_used_mb": _mb(usage.used),
            "disk_free_mb": _mb(usage.free),
            "disk_percent": round(usage.used * 100.0 / usage.total, 1) if usage.total else None,
        }
    except OSError:
        return {}


def capture_snapshot(*, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return a one-shot system performance snapshot. Never raises."""
    base = Path(base_dir) if base_dir else Path.cwd()
    psutil_mod = _safe_psutil()
    snapshot: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "psutil_available": psutil_mod is not None,
    }
    snapshot.update(_cpu_ram(psutil_mod))
    snapshot.update(_disk_info(base))
    snapshot["gpu_mps"] = _mps_memory_mb()
    return snapshot


class PerfSampler:
    """Context manager that samples `capture_snapshot()` periodically.

    Usage:
        with PerfSampler(interval=2.0) as sampler:
            train_long_thing()
        series = sampler.samples  # list of dicts, each with `elapsed_s`
    """

    def __init__(self, interval: float = 2.0, *, base_dir: Optional[Path] = None, max_samples: int = 600) -> None:
        self.interval = max(0.5, float(interval))
        self.base_dir = base_dir
        self.max_samples = int(max_samples)
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = capture_snapshot(base_dir=self.base_dir)
                snap["elapsed_s"] = round(time.monotonic() - self._t0, 3)
                self.samples.append(snap)
                if len(self.samples) > self.max_samples:
                    step = max(2, len(self.samples) // self.max_samples)
                    self.samples = self.samples[::step]
            except Exception:  # nosec B110 - perf sampling must never raise
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "PerfSampler":
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="harmonia-perf-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
            self._thread = None

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}
        cpu = [s.get("cpu_percent") for s in self.samples if isinstance(s.get("cpu_percent"), (int, float))]
        ram = [s.get("ram_percent") for s in self.samples if isinstance(s.get("ram_percent"), (int, float))]
        gpu = [
            (s.get("gpu_mps") or {}).get("allocated_mb")
            for s in self.samples
            if isinstance((s.get("gpu_mps") or {}).get("allocated_mb"), (int, float))
        ]
        def _stat(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {"avg": round(sum(values) / len(values), 2), "max": round(max(values), 2)}
        return {
            "sample_count": len(self.samples),
            "cpu_percent": _stat(cpu),
            "ram_percent": _stat(ram),
            "gpu_mps_allocated_mb": _stat(gpu),
            "duration_s": self.samples[-1].get("elapsed_s"),
        }
