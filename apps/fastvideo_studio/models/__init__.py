# SPDX-License-Identifier: Apache-2.0
"""Pydantic request/response models for the API, plus tiny shared helpers
usable by both the real server and the dependency-light mock server."""

from fastvideo_studio.models.create_job_request import CreateJobRequest
from fastvideo_studio.models.settings_update import SettingsUpdate
from fastvideo_studio.models.create_dataset_request import CreateDatasetRequest
from fastvideo_studio.models.update_caption_request import UpdateCaptionRequest


def model_label(model_path: str) -> str:
    """Derive a readable label from an HF-style model path."""
    return model_path.split("/")[-1].replace("-", " ").replace("_", " ")


__all__ = [
    "CreateJobRequest",
    "SettingsUpdate",
    "CreateDatasetRequest",
    "UpdateCaptionRequest",
    "model_label",
]
