"""What is on disk, without touching the disk to find out.

A store that has to scan its own files to answer "what have I got?" gets slower
every day it runs: the question is asked on startup, by ``doctor``, by any tool
that wants coverage, and the honest answer requires opening every partition.

So the answer is maintained rather than derived. The manifest is one small JSON
file holding, per dataset and instrument, the row count, the first and last
timestamp, and the number of sessions. It is updated on every write, and it is
advisory: if it is missing or stale the store still works, it just takes the slow
path to find out.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.calendar import IST

MANIFEST_VERSION = 1


class StoreManifest:
    """A small, self-healing index of what the store holds."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict | None = None

    # ------------------------------------------------------------------ read

    def read(self) -> dict:
        if self._cache is not None:
            return self._cache
        payload: dict[str, Any] = {"version": MANIFEST_VERSION, "datasets": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    payload.update(loaded)
                    payload.setdefault("datasets", {})
            except (json.JSONDecodeError, OSError):
                # A corrupt manifest is not a reason to lose data; the next write
                # rebuilds it from the files, which remain the source of truth.
                payload["datasets"] = {}
        self._cache = payload
        return payload

    def dataset(self, name: str, key: str) -> dict:
        return self.read()["datasets"].get(name, {}).get(key, {})

    def coverage(self, name: str, key: str) -> tuple[str | None, str | None]:
        entry = self.dataset(name, key)
        return entry.get("first"), entry.get("last")

    def row_count(self, name: str, key: str) -> int:
        return int(self.dataset(name, key).get("rows", 0) or 0)

    def instruments(self, name: str) -> list[str]:
        return sorted(self.read()["datasets"].get(name, {}))

    # ----------------------------------------------------------------- write

    def update(self, name: str, key: str, **fields: Any) -> None:
        payload = self.read()
        section = payload["datasets"].setdefault(name, {})
        entry = section.setdefault(key, {})
        entry.update(fields)
        entry["updated_at"] = datetime.now(IST).isoformat(timespec="seconds")
        payload["updated_at"] = entry["updated_at"]
        self._write(payload)

    def forget(self, name: str, key: str | None = None) -> None:
        payload = self.read()
        if key is None:
            payload["datasets"].pop(name, None)
        else:
            payload["datasets"].get(name, {}).pop(key, None)
        self._write(payload)

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._cache = payload

    def describe(self) -> str:
        payload = self.read()
        parts = []
        for name, section in payload.get("datasets", {}).items():
            rows = sum(int(entry.get("rows", 0) or 0) for entry in section.values())
            parts.append(f"{name}: {len(section)} instruments, {rows:,} rows")
        return " · ".join(parts) if parts else "empty store"


__all__ = ["MANIFEST_VERSION", "StoreManifest"]
