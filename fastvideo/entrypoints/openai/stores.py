# Adapted from SGLang
# (https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/entrypoints/openai/stores.py)

import asyncio
from typing import Any


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key: str,
                            updates: dict[str, Any]) -> dict[str, Any] | None:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_values(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._items.values())


# Global stores shared by OpenAI entrypoints
VIDEO_STORE = AsyncDictStore()
IMAGE_STORE = AsyncDictStore()
