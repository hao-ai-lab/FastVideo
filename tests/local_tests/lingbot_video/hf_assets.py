# SPDX-License-Identifier: Apache-2.0
"""Pinned public Hugging Face assets used by LingBot-Video local tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True, slots=True)
class HubModel:
    """Identify one immutable public model snapshot."""

    repo_id: str
    revision: str


OFFICIAL_DENSE = HubModel(
    "robbyant/lingbot-video-dense-1.3b",
    "f9789a7d9b4772a47aba62d4eb5282ddefd1da21",
)
OFFICIAL_MOE = HubModel(
    "robbyant/lingbot-video-moe-30b-a3b",
    "f2e538f64afe00cc4ae674db2aeb52e2945edfd5",
)
FASTVIDEO_DENSE = HubModel(
    "FastVideo/LingBot-Video-Dense-1.3B-Diffusers",
    "743ed04b96d77150d952eb08a59a56ee61b9bc95",
)
FASTVIDEO_MOE = HubModel(
    "FastVideo/LingBot-Video-MoE-30B-A3B-Diffusers",
    "401dce84db5897cb950969e766410c8eadd4fbdf",
)


def download_patterns(model: HubModel, *patterns: str) -> Path:
    """Download selected files from one pinned snapshot and return its root."""
    if not patterns:
        raise ValueError("at least one Hugging Face allow-pattern is required")
    return Path(
        snapshot_download(
            repo_id=model.repo_id,
            revision=model.revision,
            allow_patterns=list(patterns),
        )
    )


def download_components(model: HubModel, *components: str) -> Path:
    """Download root metadata and the requested Diffusers component folders."""
    return download_patterns(
        model,
        "model_index.json",
        *(f"{component}/**" for component in components),
    )


def materialize_component_view(
    model: HubModel,
    destination: Path,
    *components: str,
) -> Path:
    """Create a temporary valid model layout containing selected components."""
    snapshot = download_components(model, *components)
    selected = set(components)
    model_index = json.loads((snapshot / "model_index.json").read_text())
    filtered_index = {
        key: value
        for key, value in model_index.items()
        if not (
            isinstance(value, list)
            and value
            and value[0] is not None
            and key not in selected
        )
    }
    destination.mkdir(parents=True)
    (destination / "model_index.json").write_text(
        json.dumps(filtered_index, indent=2) + "\n",
        encoding="utf-8",
    )
    for component in components:
        (destination / component).symlink_to(snapshot / component, target_is_directory=True)
    return destination
