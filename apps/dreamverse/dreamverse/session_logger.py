# pyright: reportMissingTypeArgument=false
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import socket
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionEventLogger:

    def __init__(self, root_dir: Path):
        self.hostname = socket.gethostname()
        timestamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S_%f")
        self.directory = root_dir / self.hostname
        self.path = self.directory / f"{timestamp}.jsonl"
        self._lock = asyncio.Lock()

        self.directory.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    async def write_event(
        self,
        *,
        event: str,
        client_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "event": event,
            "hostname": self.hostname,
            "client_id": client_id,
        }
        if payload:
            entry.update(payload)

        async with self._lock:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
