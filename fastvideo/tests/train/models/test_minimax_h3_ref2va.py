# SPDX-License-Identifier: Apache-2.0
"""CPU integration contracts for MiniMax H3 Ref2VA training packing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import load_minimax_h3_ref2va_raw_samples
from fastvideo.train.models.minimax_h3.minimax_h3_ref2va import MiniMaxH3Ref2VAModel


class _EchoTransformer:
    patch_size = (1, 2, 2)

    def __call__(self, **kwargs):
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def test_raw_manifest_preserves_relative_paths_and_reference_union(tmp_path: Path) -> None:
    for relative_path in ("target.mp4", "image.png", "silent.mp4", "with-audio.mp4", "audio.wav"):
        (tmp_path / relative_path).touch()
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({
                "schema_version": "minimax_h3_ref2va_raw_v1",
                "id": sample_id,
                "target": {
                    "video": "target.mp4"
                },
                "caption": f"caption {sample_id}",
                "references": references,
            }) for sample_id, references in (
                ("mixed", [
                    {
                        "type": "image",
                        "image": "image.png"
                    },
                    {
                        "type": "video",
                        "video": "silent.mp4"
                    },
                    {
                        "type": "video_audio",
                        "video": "with-audio.mp4",
                        "audio": "audio.wav"
                    },
                    {
                        "type": "audio",
                        "audio": "audio.wav"
                    },
                ]),
                ("prompt-only", []),
            )) + "\n",
        encoding="utf-8",
    )

    samples = load_minimax_h3_ref2va_raw_samples(manifest)

    assert [sample.sample_id for sample in samples] == ["mixed", "prompt-only"]
    assert [reference.media_type for reference in samples[0].references] == [
        "image",
        "video",
        "video_audio",
        "audio",
    ]
    assert all(path.is_absolute() for reference in samples[0].references for path in (
        reference.image_path,
        reference.video_path,
        reference.audio_path,
    ) if path is not None)


def test_ref2va_training_packs_conditions_and_slices_only_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MiniMaxH3Ref2VAModel, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3Ref2VAModel.__new__(MiniMaxH3Ref2VAModel)
    model.training_config = SimpleNamespace(
        data=SimpleNamespace(
            num_latent_t=2,
            num_frames=5,
            num_height=32,
            num_width=32,
        ),
        distributed=SimpleNamespace(sp_size=1),
    )
    model.transformer = _EchoTransformer()
    model.sp_group = None
    raw_batch = {
        "vae_latent": torch.zeros(1, 24, 2, 2, 2),
        "audio_latent": torch.zeros(1, 2, 32, 8),
        "text_embedding": torch.zeros(1, 2, 5120),
        "text_attention_mask": torch.ones(1, 2),
        "text_token_tags": torch.tensor([[0, 1]], dtype=torch.long),
        "ref_visual_anchor": torch.zeros(1, 1, 96),
        "ref_audio_anchor": torch.zeros(1, 4, 32),
        "info_list": [{
            "references": [
                {
                    "media_type": "image",
                    "has_audio": False,
                    "num_latent_frames": 1,
                    "latent_height": 2,
                    "latent_width": 2,
                    "num_audio_latents": 0,
                },
                {
                    "media_type": "audio",
                    "has_audio": True,
                    "num_latent_frames": 0,
                    "latent_height": 0,
                    "latent_width": 0,
                    "num_audio_latents": 2,
                },
            ]
        }],
    }

    batch = model.prepare_batch(
        raw_batch,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )
    prediction = model.predict_noise(
        batch.noisy_model_input.permute(0, 2, 1, 3, 4),
        batch.timesteps,
        batch,
        conditional=True,
    )

    assert batch.minimax_h3_layout.num_condition_video_rows == 1
    assert batch.minimax_h3_layout.num_condition_audio_rows == 4
    assert prediction[0].shape == batch.latents.shape == (1, 2, 24, 2, 2)
    assert prediction[1].shape == batch.audio_latents.shape == (1, 2, 32, 8)
    torch.testing.assert_close(prediction[0], -batch.noisy_model_input.permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(prediction[1], -batch.audio_noisy_model_input)
