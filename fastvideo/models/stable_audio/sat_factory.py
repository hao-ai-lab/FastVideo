# SPDX-License-Identifier: Apache-2.0
# create_pretransform_from_config for autoencoder type (in-repo, no clone).
from __future__ import annotations

from typing import Any, Dict

from torch import nn

from fastvideo.models.stable_audio.sat_autoencoders import create_autoencoder_from_config
from fastvideo.models.stable_audio.sat_pretransforms import AutoencoderPretransform


def create_pretransform_from_config(
    pretransform_config: Dict[str, Any], sample_rate: int
) -> nn.Module:
    pretransform_type = pretransform_config.get("type")
    if pretransform_type != "autoencoder":
        raise ValueError(
            f"Only pretransform type 'autoencoder' is supported in-repo; "
            f"got {pretransform_type}"
        )
    cfg = pretransform_config["config"]
    autoencoder_config = {"sample_rate": sample_rate, "model": cfg}
    autoencoder = create_autoencoder_from_config(autoencoder_config)
    scale = pretransform_config.get("scale", 1.0)
    model_half = pretransform_config.get("model_half", False)
    iterate_batch = pretransform_config.get("iterate_batch", False)
    chunked = pretransform_config.get("chunked", False)
    pretransform = AutoencoderPretransform(
        autoencoder,
        scale=scale,
        model_half=model_half,
        iterate_batch=iterate_batch,
        chunked=chunked,
    )
    pretransform.enable_grad = pretransform_config.get("enable_grad", False)
    pretransform.eval().requires_grad_(pretransform.enable_grad)
    return pretransform
