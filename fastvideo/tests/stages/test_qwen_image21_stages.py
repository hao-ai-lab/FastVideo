# SPDX-License-Identifier: Apache-2.0
"""Qwen request and stage contracts with real batches and weight-free components.

Tensor operations run on CPU. These checks exercise stage integration; independent
checkpoint parity remains in tests/local_tests/qwen_image21/.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from fastvideo.models.dits.qwen_image21 import QwenImage21KVCacheAllocationError
from fastvideo.pipelines.basic.qwen_image21 import stages
from fastvideo.pipelines.basic.qwen_image21.stages import STATE_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


@pytest.fixture
def stage_args(monkeypatch):
    # Keep the real stages and caches; only replace runtime device/weight moves.
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stages.pinned_offload, "load", lambda model, device, pin=False: model.to(device))
    return SimpleNamespace(
        tp_size=1,
        sp_size=1,
        num_gpus=1,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pin_cpu_memory=False,
        output_type="pil",
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(arch_config=SimpleNamespace(in_channels=2)),
            vae_tiling=False,
            reference_resolution=32,
            use_kv_cache=True,
            kv_cache_device="cpu",
        ),
    )


def _batch(**updates):
    values = dict(
        data_type="image",
        prompt="draw",
        height=64,
        width=64,
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
        seed=42,
    )
    values.update(updates)
    return ForwardBatch(**values)


class TinyProcessor:
    tokenizer = SimpleNamespace(encode=lambda _text: [9])

    def __init__(self):
        self.images = []

    def apply_chat_template(self, *_args, **_kwargs):
        return [[1, 2]]

    def __call__(self, text, images=None, **kwargs):
        assert kwargs["padding_side"] == "left"
        self.images = list(images or [])
        last_id = 5 if "bad" in text[0] else 4
        ids = torch.tensor([[1, 2, 3] + [9] * len(self.images) + [last_id]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, **_kwargs):
        return input_ids.float().unsqueeze(-1).repeat(1, 1, 2)


class TinyVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(latents_mean=(0.0, 0.0), latents_std=(1.0, 1.0))
        self.encoded_pixels = []

    def disable_tiling(self):
        pass

    def encode(self, pixels):
        self.encoded_pixels.append(pixels.clone())
        latent = torch.nn.functional.avg_pool2d(pixels[:, [0, 3], 0], 16).unsqueeze(2)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latents, return_dict=False):
        image = latents[:, :1].repeat_interleave(16, 3).repeat_interleave(16, 4)
        return (torch.cat((image.repeat(1, 3, 1, 1, 1), torch.full_like(image, -0.5)), dim=1),)


class TinyScheduler:
    config = dict(base_image_seq_len=256, max_image_seq_len=8192, base_shift=0.5, max_shift=0.9)

    def set_timesteps(self, sigmas, device, mu):
        self.timesteps = torch.tensor(sigmas, device=device) * 1000

    def set_begin_index(self, index):
        assert index == 0

    def step(self, noise, timestep, latents, return_dict=False):
        return (latents - noise * 0.1,)


class TinyTransformer(nn.Module):
    """Populate real prefix caches so ownership and cleanup checks inspect data."""

    def __init__(self, fail_cache=False, model_error=None):
        super().__init__()
        self.config = SimpleNamespace(causal_condition=True)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])
        self.calls = []
        self.fail_cache = fail_cache
        self.model_error = model_error

    def forward(self, hidden_states, encoder_hidden_states, kv_cache_mode, kv_cache, **_kwargs):
        self.calls.append(SimpleNamespace(mode=kv_cache_mode, cache=kv_cache))
        condition = encoder_hidden_states[:, -1:, :]
        if kv_cache_mode == "extract":
            prefix = condition.unsqueeze(2)
            kv_cache.get_layer(0).store(prefix, prefix)
            kv_cache.prefix_len = 1
            if self.fail_cache:
                self.fail_cache = False
                raise QwenImage21KVCacheAllocationError("simulated host allocation failure")
        elif kv_cache_mode == "cached":
            condition = kv_cache.get_layer(0).get()[1].squeeze(2)
        if self.model_error is not None:
            raise self.model_error
        return (condition.expand_as(hidden_states),)


def _run(batch, args, transformer=None):
    transformer = transformer if transformer is not None else TinyTransformer()
    vae, processor, scheduler = TinyVAE(), TinyProcessor(), TinyScheduler()
    pipeline = [
        stages.QwenImage21InputStage(),
        stages.QwenImage21TextEncodingStage(TinyEncoder(), processor),
        stages.QwenImage21LatentStage(vae),
        stages.QwenImage21ScheduleStage(scheduler),
        stages.QwenImage21DenoisingStage(transformer, scheduler),
        stages.QwenImage21DecodingStage(vae),
    ]
    for stage in pipeline:
        batch = stage.forward(batch, args)
    return batch, transformer, vae, processor


def _assert_caches_cleared(transformer):
    caches = [call.cache for call in transformer.calls if call.cache is not None]
    assert caches
    for cache in caches:
        assert cache.prefix_len is None
        assert all(layer.k is None and layer.v is None for layer in cache.layer_caches)


def test_request_validation_accepts_real_forward_batch(stage_args):
    batch = _batch()
    result = stages.QwenImage21InputStage().forward(batch, stage_args)
    assert result is batch
    assert result.batch_size == 1
    assert result.extra[STATE_KEY].img_shapes == [[(1, 4, 4)]]
    assert not result.do_classifier_free_guidance
    for updates, message in [
        ({"num_frames": 2}, "num_frames"),
        ({"guidance_scale": 3}, "true_cfg_scale"),
        ({"height": 63}, "multiple of 32"),
        ({"true_cfg_scale": float("nan")}, "finite"),
        ({"num_videos_per_prompt": 2}, "num_videos_per_prompt"),
    ]:
        with pytest.raises(ValueError, match=message):
            stages.QwenImage21InputStage().forward(_batch(**updates), stage_args)
    for resolution in (0, 33):
        stage_args.pipeline_config.reference_resolution = resolution
        with pytest.raises(ValueError, match="reference_resolution"):
            stages.QwenImage21InputStage().forward(_batch(), stage_args)


def test_ordered_references_preserve_rgba_for_vae_and_composite_for_encoder(stage_args):
    references = [Image.new("RGBA", (32, 32), color) for color in [(0, 0, 0, 128), (128, 0, 0, 64)]]
    batch, transformer, vae, processor = _run(_batch(references=references), stage_args)
    assert batch.extra[STATE_KEY].img_shapes == [[(1, 2, 2), (1, 2, 2), (1, 4, 4)]]
    assert batch.image_latent.shape == (1, 8, 2)
    expected_red = torch.tensor([0.0, 128.0]) / 127.5 - 1
    expected_alpha = torch.tensor([128.0, 64.0]) / 127.5 - 1
    torch.testing.assert_close(batch.image_latent[0, ::4, 0], expected_red)
    torch.testing.assert_close(batch.image_latent[0, ::4, 1], expected_alpha)
    assert all(pixels.shape == (1, 4, 1, 32, 32) for pixels in vae.encoded_pixels)
    for source, encoded in zip(references, processor.images, strict=True):
        expected = Image.new("RGB", source.size, "white")
        expected.paste(source, mask=source.getchannel("A"))
        assert encoded.mode == "RGB"
        assert encoded.tobytes() == expected.tobytes()
    assert batch.output.shape == (1, 4, 1, 64, 64)
    torch.testing.assert_close(batch.output[:, 3], torch.full((1, 1, 64, 64), 0.25))
    _assert_caches_cleared(transformer)


def test_true_cfg_requires_explicit_negative_prompt_and_separate_caches(stage_args):
    initial = torch.zeros(1, 16, 2)
    guided, transformer, _, _ = _run(
        _batch(latents=initial.clone(), true_cfg_scale=2, negative_prompt="bad"), stage_args)
    assert guided.do_classifier_free_guidance
    torch.testing.assert_close(guided.latents, torch.full_like(initial, -0.6))
    assert [call.mode for call in transformer.calls] == ["extract", "extract", "cached", "cached"]
    positive_cache, negative_cache = transformer.calls[0].cache, transformer.calls[1].cache
    assert positive_cache is not negative_cache
    assert transformer.calls[2].cache is positive_cache
    assert transformer.calls[3].cache is negative_cache
    _assert_caches_cleared(transformer)

    unguided, transformer, _, _ = _run(_batch(latents=initial.clone(), true_cfg_scale=2), stage_args)
    assert not unguided.do_classifier_free_guidance
    assert [call.mode for call in transformer.calls] == ["extract", "cached"]
    torch.testing.assert_close(unguided.latents, torch.full_like(initial, -0.8))
    _assert_caches_cleared(transformer)


def test_host_cache_allocation_failure_falls_back_to_uncached_execution(stage_args):
    transformer = TinyTransformer(fail_cache=True)
    actual, _, _, _ = _run(_batch(), stage_args, transformer)
    assert [call.mode for call in transformer.calls] == ["extract", None, None]
    assert all(call.cache is None for call in transformer.calls[1:])
    _assert_caches_cleared(transformer)
    stage_args.pipeline_config.use_kv_cache = False
    expected, _, _, _ = _run(_batch(), stage_args)
    torch.testing.assert_close(actual.latents, expected.latents, atol=0, rtol=0)
    torch.testing.assert_close(actual.output, expected.output, atol=0, rtol=0)


def test_model_error_propagates_and_clears_request_cache(stage_args):
    error = RuntimeError("model failure")
    transformer = TinyTransformer(model_error=error)
    with pytest.raises(RuntimeError, match="model failure") as raised:
        _run(_batch(), stage_args, transformer)
    assert raised.value is error
    assert [call.mode for call in transformer.calls] == ["extract"]
    _assert_caches_cleared(transformer)


@pytest.mark.parametrize("offload", ["whole", "layerwise"])
def test_cancellation_stops_remaining_steps_and_releases_cache_and_weights(stage_args, monkeypatch, offload):
    batch = _batch(num_inference_steps=3, latents=torch.zeros(1, 16, 2))
    scheduler = TinyScheduler()
    for stage in (
        stages.QwenImage21InputStage(),
        stages.QwenImage21TextEncodingStage(TinyEncoder(), TinyProcessor()),
        stages.QwenImage21LatentStage(TinyVAE()),
        stages.QwenImage21ScheduleStage(scheduler),
    ):
        batch = stage.forward(batch, stage_args)

    class InterruptingTransformer(TinyTransformer):

        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            denoising.interrupt = True
            return result

    transformer = InterruptingTransformer()
    denoising = stages.QwenImage21DenoisingStage(transformer, scheduler)
    stage_args.dit_cpu_offload = offload == "whole"
    stage_args.dit_layerwise_offload = offload == "layerwise"
    loaded, unloaded, layerwise_released, steps = [], [], [], []
    monkeypatch.setattr(stages.pinned_offload, "load", lambda model, device, pin=False: loaded.append(model))
    monkeypatch.setattr(stages.pinned_offload, "unload", lambda model: unloaded.append(model))
    monkeypatch.setattr(denoising, "_release_layerwise", lambda: layerwise_released.append(True))
    original_step = scheduler.step

    def record_step(*args, **kwargs):
        steps.append(args[1])
        return original_step(*args, **kwargs)

    monkeypatch.setattr(scheduler, "step", record_step)
    result = denoising.forward(batch, stage_args)

    assert result is batch
    assert denoising.interrupt is True  # The executor owns the cancellation latch.
    assert [call.mode for call in transformer.calls] == ["extract"]
    assert len(steps) == 1
    assert batch.step_index == 0
    torch.testing.assert_close(batch.latents, torch.full_like(batch.latents, -0.4))
    _assert_caches_cleared(transformer)
    assert loaded == unloaded == ([transformer] if offload == "whole" else [])
    assert layerwise_released == ([True] if offload == "layerwise" else [])
