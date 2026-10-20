"""Raw, bounded media uploads keep large binary data out of websocket messages."""

from __future__ import annotations

import asyncio
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from dreamverse.assets import IMAGE_LIMIT, MEDIA_LIMIT, MIME_TYPES, asset_store

router = APIRouter()
_upload_lock = asyncio.Lock()


@router.post("/assets", status_code=201)
async def upload_asset(request: Request) -> dict:
    mime_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if mime_type not in MIME_TYPES:
        raise HTTPException(415, "Unsupported asset type. Select a supported image, video, or audio file.")
    limit = IMAGE_LIMIT if MIME_TYPES[mime_type][0] == "image" else MEDIA_LIMIT
    try:
        if int(request.headers.get("content-length", "0")) > limit:
            raise HTTPException(413, f"Asset exceeds the {limit // (1024 * 1024)} MB upload limit.")
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length.") from exc
    async with _upload_lock:
        try:
            path = asset_store.staging_path(mime_type)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            size = 0
            with path.open("xb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"Asset exceeds the {limit // (1024 * 1024)} MB upload limit.")
                    await run_in_threadpool(handle.write, chunk)
            asset = await run_in_threadpool(asset_store.add, path,
                                            unquote(request.headers.get("x-asset-name", "Untitled asset")), mime_type)
            return asset.public()
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            path.unlink(missing_ok=True)
            raise


@router.api_route("/assets/{asset_id}", methods=["GET", "HEAD"])
async def get_asset(asset_id: str) -> FileResponse:
    try:
        asset = asset_store.get(asset_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(asset.path, media_type=asset.mime_type, headers={"X-Content-Type-Options": "nosniff"})


@router.delete("/assets/{asset_id}", status_code=204)
async def delete_asset(asset_id: str) -> Response:
    try:
        asset_store.get(asset_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        asset_store.delete(asset_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(status_code=204)
