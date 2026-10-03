# SPDX-License-Identifier: Apache-2.0
"""Download the private FP8 model without exposing its credential."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
snapshot_download(
    "FastVideo/FastH3-Pruned-8Step-FP8-ckpt300",
    local_dir="/workspace/vol/pruned_fp8_300",
    token=Path("/root/.hf-fastvideo/token").read_text().strip(),
)
