"""Creation studio capability routes."""

from __future__ import annotations

from fastapi import APIRouter

from dreamverse.creation_capabilities import lobby_capabilities_as_dict

creation_router = APIRouter(tags=["creation"])


@creation_router.get("/creation-capabilities")
async def creation_capabilities() -> dict[str, object]:
    return lobby_capabilities_as_dict()
