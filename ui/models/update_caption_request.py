# SPDX-License-Identifier: Apache-2.0
"""Request model for updating a dataset file caption."""

from pydantic import BaseModel


class UpdateCaptionRequest(BaseModel):
    file_name: str
    caption: str
