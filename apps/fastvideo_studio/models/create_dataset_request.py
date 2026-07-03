# SPDX-License-Identifier: Apache-2.0
"""Request model for creating a dataset."""

from pydantic import BaseModel


class CreateDatasetRequest(BaseModel):
    name: str
    upload_path: str = ""  # One-time path from upload; moved to datasets_upload_dir/{id}
    file_names: list[str] = []
    captions: dict[str, str] = {}  # optional: file_name -> caption from videos2caption.json
