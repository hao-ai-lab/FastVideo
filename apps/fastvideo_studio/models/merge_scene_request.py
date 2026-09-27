# SPDX-License-Identifier: Apache-2.0
"""Request model for merging a scene's clips into one video."""

from pydantic import BaseModel


class MergeSceneRequest(BaseModel):
    #: Completed jobs, in the order their videos should play.
    job_ids: list[str]
    #: Used to name the merged file.
    name: str = ""
