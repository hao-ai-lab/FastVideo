# SPDX-License-Identifier: Apache-2.0
"""Configuration compatibility for the LTX-2.5 convolutional video VAE."""

from fastvideo.models.vaes.ltx2vae import VideoDecoderConfigurator


def test_conv_vae_honors_checkpoint_decoder_base_channels() -> None:
    decoder = VideoDecoderConfigurator.from_config({
        "vae": {
            "dims": 3,
            "latent_channels": 8,
            "out_channels": 3,
            "decoder_blocks": [],
            "patch_size": 1,
            "norm_layer": "pixel_norm",
            "causal_decoder": False,
            "timestep_conditioning": False,
            "decoder_base_channels": 12,
        }
    })

    assert decoder.conv_in.in_channels == 8
    assert decoder.conv_in.out_channels == 12
