# SPDX-License-Identifier: Apache-2.0
"""API routes for the scene writer: start a draft (or a clip revision), then poll it.

A draft runs a DeepSeek Harness session for minutes, so each one runs on its
own thread and the browser polls `GET /api/scene-writer/drafts/{run_id}`.
Drafts never create studio jobs; the browser does that after review.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from fastvideo_studio.models import ClipRevisionRequest, SceneDraftRequest
from fastvideo_studio.scene_writer.runner import (ReferenceFile, SceneWriterSettings, draft_scene, fetched_references,
                                                  job_file_form, revise_clip)

logger = logging.getLogger("fastvideo.studio.scene_writer")

router = APIRouter(prefix="/api/scene-writer")

_drafts: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _run(run_id: str, req: SceneDraftRequest, settings: SceneWriterSettings) -> None:
    try:
        result = draft_scene(
            script=req.script,
            references=[ReferenceFile(r.path, r.media_type, r.description) for r in req.references],
            scene_name=req.scene_name,
            instructions=req.instructions,
            settings=settings,
            run_id=run_id,
            find_references=req.find_references,
        )
        update = {
            "status": "completed",
            "jobs": result.jobs,
            "job_files": result.job_files,
            "final_response": result.final_response,
            "finish_reason": result.finish_reason,
        }
    except Exception as e:  # surfaced to the browser; the run directory keeps the details
        logger.exception("Scene draft %s failed", run_id)
        update = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    with _lock:
        _drafts[run_id].update(update, finished_at=time.time())


@router.post("/drafts", status_code=202)
def start_draft(req: SceneDraftRequest) -> dict[str, Any]:
    settings = SceneWriterSettings()
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _drafts[run_id] = {
            "run_id": run_id,
            "scene_name": req.scene_name,
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "jobs": [],
            "error": None,
            "run_dir": str(settings.runs_root / run_id),
        }
    threading.Thread(target=_run, args=(run_id, req, settings), daemon=True,
                     name=f"scene-draft-{run_id}").start()
    return get_draft(run_id)


@router.get("/drafts/{run_id}")
def get_draft(run_id: str) -> dict[str, Any]:
    with _lock:
        draft = _drafts.get(run_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        draft = dict(draft)
    run_dir = Path(draft.pop("run_dir"))
    if draft["status"] == "running":
        # Show clips as the agent writes them; each file already passed validation.
        draft["jobs"] = [json.loads(f.read_text()) for f in sorted((run_dir / "jobs").glob("*.json"))]
    # Each image is served by name at /drafts/{run_id}/references/{file name}.
    draft["fetched_references"] = fetched_references(run_dir)
    return draft


@router.get("/drafts/{run_id}/references/{name}")
def get_fetched_reference(run_id: str, name: str) -> FileResponse:
    """One image the writer downloaded for a draft, for review in the browser."""
    with _lock:
        draft = _drafts.get(run_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    refs_dir = Path(draft["run_dir"]) / "refs"
    if name not in {Path(image["path"]).name for image in fetched_references(refs_dir.parent)}:
        raise HTTPException(status_code=404, detail="No such fetched reference")
    return FileResponse(refs_dir / name)


# --- Revisions ----------------------------------------------------------------

_revisions: dict[str, dict[str, Any]] = {}


def _run_revision(run_id: str, req: ClipRevisionRequest, settings: SceneWriterSettings) -> None:
    try:
        result = revise_clip(req.job, req.feedback, req.neighbors, settings=settings, run_id=run_id)
        if result.job is None:
            update = {"status": "failed", "final_response": result.final_response,
                      "error": "The reviser finished without writing a revised job."}
        else:
            update = {"status": "completed", "job": result.job, "final_response": result.final_response}
    except Exception as e:  # surfaced to the browser; the run directory keeps the details
        logger.exception("Clip revision %s failed", run_id)
        update = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    with _lock:
        _revisions[run_id].update(update, finished_at=time.time())


@router.post("/revisions", status_code=202)
def start_revision(req: ClipRevisionRequest) -> dict[str, Any]:
    if not req.job.get("name"):
        raise HTTPException(status_code=422, detail="The job to revise has no name.")
    settings = SceneWriterSettings()
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _revisions[run_id] = {
            "run_id": run_id,
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "original": job_file_form(req.job),
            "job": None,
            "final_response": None,
            "error": None,
        }
    threading.Thread(target=_run_revision, args=(run_id, req, settings), daemon=True,
                     name=f"clip-revision-{run_id}").start()
    return get_revision(run_id)


@router.get("/revisions/{run_id}")
def get_revision(run_id: str) -> dict[str, Any]:
    with _lock:
        revision = _revisions.get(run_id)
        if revision is None:
            raise HTTPException(status_code=404, detail="Revision not found")
        return dict(revision)
