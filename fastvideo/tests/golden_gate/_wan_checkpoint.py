# SPDX-License-Identifier: Apache-2.0
"""Immutable Wan 1.3B inputs shared by its small numerical gates."""

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def component_path(component):
    root = snapshot_download(WAN_REPO, revision=WAN_REVISION, allow_patterns=[f"{component}/*"])
    return Path(root) / component


def checkpoint_identity(path):
    return {
        "repo": WAN_REPO,
        "revision": WAN_REVISION,
        "component": path.name,
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
    }
