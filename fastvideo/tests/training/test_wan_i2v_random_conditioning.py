# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.testing import assert_close

import fastvideo.training.wan_i2v_training_pipeline as wan_i2v_training_pipeline
from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.wan_i2v_training_pipeline import WanI2VTrainingPipeline


def _build_pipeline(
    min_condition_latents: int = 1,
    max_condition_latents: int = 1,
    num_latent_t: int = 5,
) -> WanI2VTrainingPipeline:
    # Build just enough pipeline state to exercise the random-context helpers
    # without loading model weights, schedulers, or distributed process groups.
    pipeline = WanI2VTrainingPipeline.__new__(WanI2VTrainingPipeline)
    pipeline.training_args = TrainingArgs(
        model_path="dummy-model",
        num_latent_t=num_latent_t,
        min_condition_latents=min_condition_latents,
        max_condition_latents=max_condition_latents,
    )
    pipeline.noise_gen_cuda = torch.Generator(device="cpu").manual_seed(11)
    return pipeline


def test_training_args_defaults_enable_random_context() -> None:
    training_args = TrainingArgs(model_path="dummy-model")

    assert training_args.min_condition_latents == 1
    assert training_args.max_condition_latents == 1
    assert WanI2VTrainingPipeline._random_context_enabled(training_args)


def test_sample_condition_latent_counts_inclusive_range() -> None:
    pipeline = _build_pipeline(
        min_condition_latents=1,
        max_condition_latents=5,
        num_latent_t=13,
    )

    counts = pipeline._sample_condition_latent_counts(
        batch_size=256,
        device=torch.device("cpu"),
    )

    # torch.randint uses an exclusive upper bound, so this verifies the helper
    # exposes the configured max as an inclusive training knob.
    assert counts.min().item() >= 1
    assert counts.max().item() <= 5
    assert set(counts.tolist()).issubset({1, 2, 3, 4, 5})


def test_apply_random_context_conditioning_clean_prefix_and_masks() -> None:
    pipeline = _build_pipeline(num_latent_t=5)
    # Force different prefix lengths per sample so this test covers both the
    # single-frame-compatible path and the multi-latent continuation path.
    pipeline._sample_condition_latent_counts = (  # type: ignore[method-assign]
        lambda batch_size, device: torch.tensor([1, 3], device=device)
    )

    # The increasing latent values make prefix replacement easy to inspect:
    # clean latents should replace -1.0 noise only inside each sampled prefix.
    latents = torch.arange(20, dtype=torch.float32).view(2, 2, 5, 1, 1)
    noisy_model_input = torch.full_like(latents, -1.0)
    training_batch = TrainingBatch(
        latents=latents,
        noisy_model_input=noisy_model_input,
    )

    training_batch = pipeline._apply_random_context_conditioning(training_batch)

    expected_noisy = noisy_model_input.clone()
    expected_noisy[0, :, :1] = latents[0, :, :1]
    expected_noisy[1, :, :3] = latents[1, :, :3]
    assert_close(training_batch.noisy_model_input, expected_noisy)

    expected_condition = torch.zeros_like(latents, dtype=torch.bfloat16)
    expected_condition[0, :, :1] = latents[0, :, :1].to(torch.bfloat16)
    expected_condition[1, :, :3] = latents[1, :, :3].to(torch.bfloat16)
    assert_close(training_batch.image_latents, expected_condition)

    # Wan-Fun expects a 4-channel latent-size conditioning mask. It should mark
    # exactly the same prefix timesteps as the clean context tensor above.
    assert training_batch.mask_lat_size is not None
    assert training_batch.mask_lat_size.shape == (2, 4, 5, 1, 1)
    assert training_batch.mask_lat_size.dtype == torch.bfloat16
    assert_close(
        training_batch.mask_lat_size[0, :, :1],
        torch.ones(4, 1, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[0, :, 1:],
        torch.zeros(4, 4, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[1, :, :3],
        torch.ones(4, 3, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[1, :, 3:],
        torch.zeros(4, 2, 1, 1, dtype=torch.bfloat16),
    )

    # The loss mask is the inverse of the clean-prefix mask: False means "do
    # not train on this conditioned latent"; True means generated suffix.
    assert training_batch.generation_loss_mask is not None
    assert training_batch.generation_loss_mask.shape == (2, 1, 5, 1, 1)
    assert training_batch.generation_loss_mask[0, 0, :1].tolist() == [[[False]]]
    assert training_batch.generation_loss_mask[0, 0, 1:].all()
    assert not training_batch.generation_loss_mask[1, 0, :3].any()
    assert training_batch.generation_loss_mask[1, 0, 3:].all()


def test_apply_legacy_i2v_conditioning_keeps_first_frame_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(min_condition_latents=0, max_condition_latents=0, num_latent_t=5)
    first_frame_latent = torch.randn(2, 16, 5, 1, 1)
    training_batch = TrainingBatch(image_latents=first_frame_latent)

    training_batch = pipeline._apply_legacy_i2v_conditioning(training_batch)

    assert_close(training_batch.image_latents, first_frame_latent.to(torch.bfloat16))
    assert training_batch.mask_lat_size is not None
    assert training_batch.mask_lat_size.shape == (2, 4, 5, 1, 1)
    assert training_batch.mask_lat_size.dtype == torch.bfloat16
    assert_close(
        training_batch.mask_lat_size[:, :, :1],
        torch.ones(2, 4, 1, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[:, :, 1:],
        torch.zeros(2, 4, 4, 1, 1, dtype=torch.bfloat16),
    )
    assert training_batch.condition_latent_counts is None
    assert training_batch.generation_loss_mask is None


def test_get_next_batch_requires_clip_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(num_latent_t=5)
    pipeline.current_epoch = 0
    pipeline.train_loader_iter = iter([{
        "vae_latent": torch.randn(2, 16, 13, 1, 1),
        "text_embedding": torch.randn(2, 1, 4096),
        "text_attention_mask": torch.ones(2, 1),
        "info_list": [{}, {}],
    }])

    with pytest.raises(ValueError, match="requires clip_feature"):
        pipeline._get_next_batch(TrainingBatch())


def test_get_next_batch_requires_first_frame_latent_for_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(min_condition_latents=0, max_condition_latents=0, num_latent_t=5)
    pipeline.current_epoch = 0
    pipeline.train_loader_iter = iter([{
        "vae_latent": torch.randn(2, 16, 13, 1, 1),
        "text_embedding": torch.randn(2, 1, 4096),
        "text_attention_mask": torch.ones(2, 1),
        "clip_feature": torch.randn(2, 257, 1280),
        "info_list": [{}, {}],
    }])

    with pytest.raises(ValueError, match="requires first_frame_latent"):
        pipeline._get_next_batch(TrainingBatch())


def test_get_next_batch_accepts_endpoint_clip_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(num_latent_t=5)
    pipeline.current_epoch = 0
    # Endpoint CLIP caches store one image-token sequence per allowed context
    # length. The context-specific sequence is selected after ctx is sampled.
    pipeline.train_loader_iter = iter([{
        "vae_latent": torch.randn(2, 16, 13, 1, 1),
        "text_embedding": torch.randn(2, 1, 4096),
        "text_attention_mask": torch.ones(2, 1),
        "clip_feature": torch.randn(2, 5, 257, 1280),
        "info_list": [{}, {}],
    }])

    training_batch = pipeline._get_next_batch(TrainingBatch())

    assert training_batch.latents is not None
    assert training_batch.latents.shape == (2, 16, 5, 1, 1)
    assert training_batch.encoder_hidden_states is not None
    assert training_batch.encoder_hidden_states.shape == (2, 1, 4096)
    assert training_batch.encoder_attention_mask is not None
    assert training_batch.encoder_attention_mask.shape == (2, 1)
    assert training_batch.preprocessed_image is None
    assert training_batch.image_embeds is not None
    assert training_batch.image_embeds.shape == (2, 5, 257, 1280)
    assert training_batch.image_latents is None


def test_build_input_kwargs_requires_clip_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(num_latent_t=5)
    training_batch = TrainingBatch(
        noisy_model_input=torch.zeros(1, 36, 5, 1, 1),
        encoder_hidden_states=torch.zeros(1, 1, 4096),
        encoder_attention_mask=torch.ones(1, 1),
        timesteps=torch.zeros(1),
        image_embeds=None,
    )

    with pytest.raises(ValueError, match="requires non-empty CLIP"):
        pipeline._build_input_kwargs(training_batch)


def test_build_input_kwargs_selects_endpoint_clip_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(num_latent_t=5)
    image_embeds = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).view(2, 4, 2, 3)
    training_batch = TrainingBatch(
        noisy_model_input=torch.zeros(2, 36, 5, 1, 1),
        encoder_hidden_states=torch.zeros(2, 1, 4096),
        encoder_attention_mask=torch.ones(2, 1),
        timesteps=torch.zeros(2),
        image_embeds=image_embeds,
        condition_latent_counts=torch.tensor([1, 3]),
    )

    training_batch = pipeline._build_input_kwargs(training_batch)

    assert training_batch.input_kwargs is not None
    assert training_batch.input_kwargs["hidden_states"].shape == (2, 36, 5, 1, 1)
    expected_clip = torch.stack([image_embeds[0, 0], image_embeds[1, 2]]).to(torch.bfloat16)
    assert_close(training_batch.input_kwargs["encoder_hidden_states_image"], expected_clip)


def test_build_input_kwargs_selects_first_endpoint_clip_for_legacy_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(min_condition_latents=0, max_condition_latents=0, num_latent_t=5)
    image_embeds = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).view(2, 4, 2, 3)
    training_batch = TrainingBatch(
        noisy_model_input=torch.zeros(2, 36, 5, 1, 1),
        encoder_hidden_states=torch.zeros(2, 1, 4096),
        encoder_attention_mask=torch.ones(2, 1),
        timesteps=torch.zeros(2),
        image_embeds=image_embeds,
        condition_latent_counts=None,
    )

    training_batch = pipeline._build_input_kwargs(training_batch)

    assert training_batch.input_kwargs is not None
    assert_close(training_batch.input_kwargs["encoder_hidden_states_image"], image_embeds[:, 0].to(torch.bfloat16))


def test_build_input_kwargs_keeps_legacy_first_frame_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(num_latent_t=5)
    image_embeds = torch.arange(2 * 2 * 3, dtype=torch.float32).view(2, 2, 3)
    training_batch = TrainingBatch(
        noisy_model_input=torch.zeros(2, 36, 5, 1, 1),
        encoder_hidden_states=torch.zeros(2, 1, 4096),
        encoder_attention_mask=torch.ones(2, 1),
        timesteps=torch.zeros(2),
        image_embeds=image_embeds,
        condition_latent_counts=torch.tensor([1, 3]),
    )

    training_batch = pipeline._build_input_kwargs(training_batch)

    assert training_batch.input_kwargs is not None
    assert_close(training_batch.input_kwargs["encoder_hidden_states_image"], image_embeds.to(torch.bfloat16))


def test_build_input_kwargs_keeps_clip_for_explicit_legacy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wan_i2v_training_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))
    pipeline = _build_pipeline(min_condition_latents=0, max_condition_latents=0, num_latent_t=5)
    image_embeds = torch.arange(2 * 2 * 3, dtype=torch.float32).view(2, 2, 3)
    training_batch = TrainingBatch(
        noisy_model_input=torch.zeros(2, 36, 5, 1, 1),
        encoder_hidden_states=torch.zeros(2, 1, 4096),
        encoder_attention_mask=torch.ones(2, 1),
        timesteps=torch.zeros(2),
        image_embeds=image_embeds,
    )

    training_batch = pipeline._build_input_kwargs(training_batch)

    assert training_batch.input_kwargs is not None
    assert_close(training_batch.input_kwargs["encoder_hidden_states_image"], image_embeds.to(torch.bfloat16))


def test_masked_loss_ignores_conditioned_prefix() -> None:
    target = torch.zeros(1, 2, 4, 1, 1)
    model_pred = torch.zeros_like(target)
    # Put all error in the conditioned prefix. A correct mask should reduce
    # this to zero loss instead of penalizing the clean context frames.
    model_pred[:, :, :2] = 100.0
    training_batch = TrainingBatch(
        generation_loss_mask=torch.tensor(
            [[[[[False]], [[False]], [[True]], [[True]]]]]
        )
    )

    loss = TrainingPipeline._compute_loss(model_pred, target, training_batch)

    assert_close(loss, torch.tensor(0.0))


def test_masked_loss_normalizes_over_generated_suffix() -> None:
    target = torch.zeros(1, 2, 4, 1, 1)
    model_pred = torch.zeros_like(target)
    # Only the generated suffix is active in the loss mask; the result should
    # be the mean squared error over that suffix, not diluted by prefix zeros.
    model_pred[:, :, 2:] = 2.0
    training_batch = TrainingBatch(
        generation_loss_mask=torch.tensor(
            [[[[[False]], [[False]], [[True]], [[True]]]]]
        )
    )

    loss = TrainingPipeline._compute_loss(model_pred, target, training_batch)

    assert_close(loss, torch.tensor(4.0))


def test_masked_loss_rejects_empty_generated_suffix() -> None:
    target = torch.zeros(1, 2, 4, 1, 1)
    model_pred = torch.zeros_like(target)
    training_batch = TrainingBatch(
        generation_loss_mask=torch.zeros(1, 1, 4, 1, 1, dtype=torch.bool)
    )

    with pytest.raises(ValueError, match="at least one generated element"):
        TrainingPipeline._compute_loss(model_pred, target, training_batch)


def test_disabled_condition_latent_args_are_valid() -> None:
    training_args = TrainingArgs(
        model_path="dummy-model",
        num_latent_t=13,
        min_condition_latents=0,
        max_condition_latents=0,
    )

    WanI2VTrainingPipeline._validate_condition_latent_args(training_args)


@pytest.mark.parametrize(
    "min_condition_latents,max_condition_latents,num_latent_t",
    [
        # Only 0/0 disables the feature. Enabled ranges must have positive min,
        # max must not precede min, and at least one latent timestep must
        # remain available as a generated target.
        (0, 1, 13),
        (1, 0, 13),
        (3, 2, 13),
        (1, 13, 13),
    ],
)
def test_invalid_condition_latent_args_raise(
    min_condition_latents: int,
    max_condition_latents: int,
    num_latent_t: int,
) -> None:
    training_args = TrainingArgs(
        model_path="dummy-model",
        num_latent_t=num_latent_t,
        min_condition_latents=min_condition_latents,
        max_condition_latents=max_condition_latents,
    )

    with pytest.raises(ValueError):
        WanI2VTrainingPipeline._validate_condition_latent_args(training_args)
