"""Fail-safe resource guard (RAM + disk) for long TotalSegmentator runs.

Why this exists: the earlier preprocessing driver compared values in bash. Windows
Python prints "\\r\\n"; bash ``read`` kept the trailing "\\r" on the last field, so
``[ "$AV" -lt N ]`` errored ("integer expression expected"), the test counted as
false, and the RAM stop could never fire (fail-OPEN). Here every comparison is a
Python numeric comparison and every missing/invalid reading stops the run (fail-SAFE).

The guard only READS resources. It never deletes files, frees memory, or touches
other processes. Every check (pass or stop) is appended to a JSONL log.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

GB = 1024 ** 3


class ResourceStop(RuntimeError):
    """Raised when a limit is crossed or a reading is missing/invalid."""

    def __init__(self, reason: str, record: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.record = record


def read_ram_available() -> Any:
    return psutil.virtual_memory().available


def make_disk_reader(path: Path) -> Callable[[], Any]:
    def read() -> Any:
        return shutil.disk_usage(str(path)).free
    return read


def _validate(name: str, value: Any) -> tuple[int | None, str | None]:
    """Return (bytes, None) for a valid reading, else (None, problem)."""
    if value is None:
        return None, f"{name} reading missing (None)"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} reading has invalid type {type(value).__name__}: {value!r}"
    if isinstance(value, float) and not math.isfinite(value):
        return None, f"{name} reading not finite: {value!r}"
    if value < 0:
        return None, f"{name} reading negative: {value!r}"
    return int(value), None


@dataclass
class ResourceGuard:
    min_ram_bytes: int
    min_disk_bytes: int
    disk_path: Path
    log_path: Path | None = None
    ram_reader: Callable[[], Any] = read_ram_available
    disk_reader: Callable[[], Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for n, v in (("min_ram_bytes", self.min_ram_bytes), ("min_disk_bytes", self.min_disk_bytes)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{n} must be a positive int, got {v!r}")
        if self.disk_reader is None:
            self.disk_reader = make_disk_reader(self.disk_path)

    def _read(self, name: str, reader: Callable[[], Any]) -> tuple[int | None, str | None]:
        try:
            raw = reader()
        except Exception as e:  # noqa: BLE001 - any failure to read is a stop
            return None, f"{name} reading failed: {e!r}"
        return _validate(name, raw)

    def check(self, stage: str) -> dict[str, Any]:
        ram, ram_err = self._read("RAM", self.ram_reader)
        disk, disk_err = self._read("disk", self.disk_reader)
        problems = [p for p in (ram_err, disk_err) if p]
        if ram is not None and ram < self.min_ram_bytes:
            problems.append(f"available RAM {ram / GB:.3f} GB < limit {self.min_ram_bytes / GB:.3f} GB")
        if disk is not None and disk < self.min_disk_bytes:
            problems.append(f"free disk {disk / GB:.3f} GB < limit {self.min_disk_bytes / GB:.3f} GB")
        rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "stage": stage,
               "ram_available_bytes": ram, "disk_free_bytes": disk, "disk_path": str(self.disk_path),
               "min_ram_bytes": self.min_ram_bytes, "min_disk_bytes": self.min_disk_bytes,
               "ok": not problems, "stop_reasons": problems}
        self.history.append(rec)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        if problems:
            raise ResourceStop("; ".join(problems), rec)
        return rec
