# SPDX-License-Identifier: Apache-2.0
"""Same-origin prompt UI for the H3 text-to-video/audio server."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from fastvideo.api.compat import explicit_request_updates
from fastvideo.entrypoints.openai.state import get_default_request, get_served_model_name, get_server_args
from fastvideo.registry import get_preset_selection

ASSETS = Path(__file__).with_name("static")
HEADERS = {
    "Cache-Control":
    "no-store",
    "X-Content-Type-Options":
    "nosniff",
    "Content-Security-Policy":
    ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
     "media-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
}


def require_h3() -> None:
    args = get_server_args()
    _, family = get_preset_selection(args.model_path)
    override = getattr(args, "override_pipeline_cls_name", None)
    if family != "minimax_h3" or (override and override != "MiniMaxH3ModularPipeline"):
        raise HTTPException(status_code=404, detail="The playground requires an H3 text-to-video/audio server.")


router = APIRouter(prefix="/playground", dependencies=[Depends(require_h3)], include_in_schema=False)


@router.get("/")
async def playground() -> FileResponse:
    return FileResponse(ASSETS / "playground.html", headers=HEADERS)


@router.get("/config")
async def playground_config(raw_request: Request) -> dict:
    request = get_default_request()
    # Report operator defaults only; do not guess model presets or hardware.
    sampling = explicit_request_updates(request) if request is not None else {}
    fields = ("width", "height", "num_frames", "fps", "seed")
    return {
        "model": get_served_model_name(),
        "runtime": raw_request.app.state.runtime,
        "defaults": {
            field: sampling.get(field)
            for field in fields
        },
    }


@router.get("/{asset}")
async def playground_asset(asset: str) -> FileResponse:
    if asset not in {"playground.css", "playground.js"}:
        raise HTTPException(status_code=404, detail="Playground asset not found.")
    return FileResponse(ASSETS / asset, headers=HEADERS)
