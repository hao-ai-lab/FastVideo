# SPDX-License-Identifier: Apache-2.0
"""Serve a job's image reference so the job form can show a thumbnail of it.

Only a file the job itself lists as an image reference is served: the API may
be reachable from outside (a tunnel), so a path from the request alone never
opens a file.
"""

from __future__ import annotations

import mimetypes
import os
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def reference_image_response(references: list[dict[str, Any]] | None, source: str) -> FileResponse:
    listed = any(r.get("source") == source and r.get("media_type") == "image" for r in references or [])
    ext = os.path.splitext(source)[1].lower()
    if not listed or ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="No such image reference on this job")
    if not os.path.isfile(source):
        raise HTTPException(status_code=404, detail="Reference file not found on disk")
    return FileResponse(source, media_type=mimetypes.guess_type(source)[0] or "application/octet-stream")
