# SPDX-License-Identifier: Apache-2.0
"""Synthetic ordered reference geometry for CPU model tests."""
from types import SimpleNamespace
import torch


def _training_config(data_type: str = "ref2va") -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            preprocessed_data_type=data_type,
            train_batch_size=1,
            training_cfg_rate=0.0,
            num_latent_t=2,
            num_frames=5,
            num_height=64,
            num_width=64,
            native_shape_bucketing=False,
        ),
        distributed=SimpleNamespace(sp_size=1),
        model=SimpleNamespace(enable_gradient_checkpointing_type=None, minimax_h3_activation_offload_min_tokens=0),
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(uniform_parameter_dtype=False)),
        vsa_sparsity=0.0,
        vsa_ref_keep_rate=None,
        vsa_tile_size=256,
    )


def _raw_batch(seed: int = 7) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    # Ordered references are: video+soundtrack, image, audio. Their modality
    # row tensors stay modality-major, exactly as the transformer API expects.
    video_rows = torch.cat((
        torch.full((8, 96), 10.0),
        torch.full((4, 96), 20.0),
    ))
    audio_rows = torch.cat((
        torch.full((6, 32), 30.0),
        torch.full((4, 32), 40.0),
    ))
    return {
        "vae_latent": torch.randn(1, 24, 2, 4, 4, generator=generator),
        "audio_latent": torch.randn(1, 2, 32, 8, generator=generator),
        "text_embedding": torch.randn(1, 5, 5120, generator=generator),
        "text_attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32),
        "text_token_tags": torch.tensor([[1, 0, 0, 1, 0]], dtype=torch.float32),
        "reference_video_latent_rows": video_rows[None],
        "reference_audio_latent_rows": audio_rows[None],
        "info_list": [{
            "reference_media_types": ["video", "image", "audio"],
            "reference_has_audio": [True, False, True],
            "reference_num_latent_frames": [2, 1, 0],
            "reference_latent_heights": [4, 4, 0],
            "reference_latent_widths": [4, 4, 0],
            "reference_num_audio_latents": [3, 0, 2],
            "reference_noise_seed": 1234,
        }],
    }
