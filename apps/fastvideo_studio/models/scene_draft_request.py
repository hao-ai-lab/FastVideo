# SPDX-License-Identifier: Apache-2.0
"""Request model for drafting a scene's jobs with the scene writer."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SceneDraftReference(BaseModel):
    path: str
    media_type: Literal["image", "audio", "video"]
    description: str = ""


class SceneDraftRequest(BaseModel):
    script: str = Field(min_length=1)
    scene_name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    references: list[SceneDraftReference] = Field(default_factory=list)
    instructions: str = ""
    # Let the writer download reference photos from Wikimedia Commons for subjects with none.
    find_references: bool = False

    @model_validator(mode="after")
    def _needs_references(self) -> "SceneDraftRequest":
        if not self.references and not self.find_references:
            raise ValueError("Give at least one reference file, or let the writer find references online.")
        return self


class ClipRevisionRequest(BaseModel):
    """Revise one clip from feedback on its generated take; `job` and `neighbors` are Studio jobs."""

    job: dict[str, Any]
    feedback: str = Field(min_length=1)
    # The clips before and after it in the scene, for continuity.
    neighbors: list[dict[str, Any]] = Field(default_factory=list)
