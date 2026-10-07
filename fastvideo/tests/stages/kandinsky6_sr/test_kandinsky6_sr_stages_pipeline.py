# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR pipeline on tiny random bundles (CPU): stage chain, scheduler-driven sampling, inputs and outputs."""
from __future__ import annotations

from pathlib import Path

import k6_sr_tiny
import pytest
import torch

from fastvideo.configs.pipelines.kandinsky6_sr_options import SR_OPTION_FIELDS
from fastvideo.pipelines.basic.kandinsky6_sr import sr_io, tiling
from fastvideo.pipelines.basic.kandinsky6_sr.kandinsky6_sr_pipeline import Kandinsky6SRPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

pytestmark = pytest.mark.usefixtures("distributed_setup", "tiny_resolutions")


def _batch(**kwargs) -> ForwardBatch:
    defaults = dict(data_type="video", prompt="", seed=42, sr_resolution_scale=2.0, save_video=True,
                    return_frames=False, num_inference_steps=2)
    defaults.update(kwargs)
    extra = dict(defaults.pop("extra", {}))
    extra.update({key: defaults.pop(key) for key in SR_OPTION_FIELDS if key in defaults})
    return ForwardBatch(**defaults, extra=extra)


def _load(bundle: Path, args) -> Kandinsky6SRPipeline:
    pipeline = Kandinsky6SRPipeline(str(bundle), args)
    pipeline.post_init()
    return pipeline


def _frames(result: ForwardBatch) -> torch.Tensor:
    return (result.output[0] * 255).clamp_(0, 255).to(torch.uint8)


@pytest.fixture()
def clip(tmp_path) -> Path:
    path = tmp_path / "clip.mp4"
    k6_sr_tiny.write_mp4(path, 20, 24, size=(64, 96), audio_seconds=2.0)
    return path


@pytest.fixture(params=["distilled", "flow_matching"])
def bundle_args(request, cpu_args):
    bundle = request.getfixturevalue(f"{request.param}_bundle")
    return bundle, cpu_args(bundle)


def _count_calls(monkeypatch, obj, name) -> list[int]:
    calls = []
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, name, wrapped)
    return calls


# x2.25 pre-upscales 64x96 by 1.125 to the 16 px grid (64x112), then tiles at x2.
@pytest.mark.parametrize("scale, size, expected_hw", [(2.0, (64, 96), (128, 192)), (2.25, (64, 96), (128, 224)),
                                                      (4.0, (64, 64), (256, 256))])
def test_pipeline_upscales_and_keeps_fps_and_audio(bundle_args, tmp_path, scale, size, expected_hw):
    bundle, args = bundle_args
    clip = tmp_path / "clip.mp4"
    k6_sr_tiny.write_mp4(clip, 20, 24, size=size, audio_seconds=2.0)
    pipeline = _load(bundle, args)
    assert list(pipeline._stage_name_mapping) == [
        "video_encoding_stage", "latent_preparation_stage", "denoising_stage", "decoding_stage"
    ]
    result = pipeline.forward(_batch(video_path=str(clip), sr_resolution_scale=scale), args)

    assert result.output.dtype == torch.float32 and tuple(result.output.shape) == (1, 3, 17, *expected_hw)
    assert (result.height, result.width, result.num_frames) == (*expected_hw, 17)
    assert result.fps == result.extra["output_fps"] == 24
    assert result.extra["audio_sample_rate"] == sr_io.AUDIO_SAMPLE_RATE
    assert result.extra["audio"].shape[0] == round(17 / 24 * sr_io.AUDIO_SAMPLE_RATE)
    assert result.latents is None and result.lq_latents is None  # large intermediates are not shipped back


def test_each_tile_runs_num_inference_steps_through_the_scheduler(bundle_args, clip, monkeypatch):
    bundle, args = bundle_args
    pipeline = _load(bundle, args)
    transformer = pipeline.get_module("transformer")
    scheduler = pipeline.get_module("scheduler")
    forwards = _count_calls(monkeypatch, transformer, "forward")
    steps = _count_calls(monkeypatch, scheduler, "step")
    pipeline.forward(_batch(video_path=str(clip), num_inference_steps=3), args)
    num_tiles = tiling.plan_tiles(64, 96, 512, 2, 0.2, 16).num_tiles
    assert num_tiles > 1
    assert len(forwards) == len(steps) == num_tiles * 3


def test_same_seed_is_reproducible_and_a_new_seed_changes_the_result(bundle_args, clip):
    bundle, args = bundle_args
    pipeline = _load(bundle, args)
    first = _frames(pipeline.forward(_batch(video_path=str(clip)), args))
    again = _frames(pipeline.forward(_batch(video_path=str(clip)), args))
    other = _frames(pipeline.forward(_batch(video_path=str(clip), seed=7), args))
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_tile_groups_are_seeded_by_their_first_tile(fv_args, distilled_bundle, clip):
    pipeline = _load(distilled_bundle, fv_args)
    single = _frames(pipeline.forward(_batch(video_path=str(clip)), fv_args))
    grouped = _frames(pipeline.forward(_batch(video_path=str(clip), sr_tiles_batch_size=2), fv_args))
    assert single.shape == grouped.shape
    assert not torch.equal(single, grouped)  # tile 1 draws its noise from the group seeded with seed + 0


def test_raw_latent_input_skips_the_video_and_keeps_an_audio_override(fv_args, distilled_bundle, monkeypatch):
    monkeypatch.setattr(sr_io, "read_video", lambda *a, **k: pytest.fail("the video must not be decoded"))
    latent = torch.randn(3, 4, 4, 6, generator=torch.Generator().manual_seed(5))
    audio = torch.linspace(-0.5, 0.5, 1000)
    batch = _batch()
    batch.extra.update(sr_lr_latent=latent, sr_audio=audio, sr_audio_sample_rate=16000)
    result = _load(distilled_bundle, fv_args).forward(batch, fv_args)
    assert tuple(result.output.shape[-2:]) == (4 * 16 * 2, 6 * 16 * 2)
    assert torch.equal(result.extra["audio"], audio) and result.extra["audio_sample_rate"] == 16000
    assert "sr_lr_latent" not in result.extra and "sr_audio" not in result.extra


@pytest.mark.parametrize("extra, kwargs, error", [
    ({"sr_lr_latent": torch.zeros(3, 4, 4, 6)}, {"sr_resolution_scale": 2.25}, "2.25"),
    ({"sr_lr_latent": torch.zeros(4, 4, 6)}, {}, r"\[T, C, H, W\]"),
    ({"sr_lr_latent": torch.zeros(3, 4, 4, 6)}, {"video_path": "clip.mp4"}, "not both"),
    ({}, {"sr_resolution_scale": 3.0}, "2, 4 or 2.25"),
    ({}, {"video_path": "clip.mp4", "sr_target_resolution": "8k"}, "sr_target_resolution"),
    ({}, {"video_path": ["a.mp4", "b.mp4"]}, "one video"),
    ({}, {}, "video_path"),
])
def test_bad_requests_fail_before_any_model_runs(fv_args, distilled_bundle, extra, kwargs, error):
    batch = _batch(**kwargs)
    batch.extra.update(extra)
    with pytest.raises(ValueError, match=error):
        _load(distilled_bundle, fv_args).forward(batch, fv_args)


@pytest.mark.parametrize("bundle_name", ["distilled", "flow_matching"])
def test_a_transformer_from_the_other_bundle_is_rejected(request, cpu_args, clip, monkeypatch, bundle_name):
    bundle = request.getfixturevalue(f"{bundle_name}_bundle")
    args = cpu_args(bundle)
    pipeline = _load(bundle, args)
    arch = pipeline.get_module("transformer").config.arch_config
    scheduler = pipeline.get_module("scheduler")
    # Give each transformer the other bundle's head width.
    monkeypatch.setattr(arch, "out_visual_dim",
                        arch.in_visual_dim if bundle_name == "distilled" else arch.in_visual_dim * 10)
    forwards = _count_calls(monkeypatch, pipeline.get_module("transformer"), "forward")
    with pytest.raises(ValueError, match=f"head is {arch.out_visual_dim} channels wide, but {type(scheduler).__name__}"):
        pipeline.forward(_batch(video_path=str(clip)), args)
    assert not forwards


def test_zero_steps_are_rejected(fv_args, distilled_bundle, clip):
    with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
        _load(distilled_bundle, fv_args).forward(_batch(video_path=str(clip), num_inference_steps=0), fv_args)


def test_a_missing_upscaler_scale_is_reported(fv_args, distilled_bundle, tmp_path):
    clip = tmp_path / "square.mp4"
    k6_sr_tiny.write_mp4(clip, 20, 24, size=(64, 64))
    pipeline = _load(distilled_bundle, fv_args)
    bank = pipeline.get_module("latent_upscaler")
    bank.config.scales = (2, )  # a bank that only serves x2
    del bank._models[1]
    with pytest.raises(ValueError, match="no x4"):
        pipeline.forward(_batch(video_path=str(clip), sr_resolution_scale=4.0), fv_args)


def test_warmup_request_without_a_video_completes(fv_args, distilled_bundle):
    result = _load(distilled_bundle, fv_args).forward(_batch(save_video=False), fv_args)
    assert result.output.ndim == 5


def test_output_codes_survive_video_generators_truncating_conversion(fv_args, distilled_bundle, clip):
    result = _load(distilled_bundle, fv_args).forward(_batch(video_path=str(clip)), fv_args)
    codes = result.output * 255
    fraction = codes - codes.floor()
    assert bool((((fraction - 0.5).abs() < 1e-3) | (codes == 255)).all())


def test_delivery_resize_fits_the_bucket(fv_args, distilled_bundle, clip):
    result = _load(distilled_bundle, fv_args).forward(_batch(video_path=str(clip), sr_target_resolution="96x64"),
                                                      fv_args)
    assert (result.height, result.width) == (64, 96)


def test_tiles_reach_stitch_tiles_as_float_not_pre_quantized_uint8(fv_args, distilled_bundle, clip, monkeypatch):
    # Tiles must stay float through blending and be quantized to uint8 once, inside
    # stitch_tiles -- not per tile beforehand (which would blend already-rounded values). Spy on the
    # actual call the decoding stage makes.
    import fastvideo.pipelines.stages.kandinsky6_sr as k6_sr_stages

    seen_tiles = []
    original = k6_sr_stages.stitch_tiles

    def spy(tiles, *args, **kwargs):
        seen_tiles.extend(tiles)
        return original(tiles, *args, **kwargs)

    monkeypatch.setattr(k6_sr_stages, "stitch_tiles", spy)
    _load(distilled_bundle, fv_args).forward(_batch(video_path=str(clip)), fv_args)

    assert seen_tiles, "stitch_tiles was never called"
    for tile in seen_tiles:
        assert tile.dtype.is_floating_point, f"tile reached stitch_tiles pre-quantized as {tile.dtype}"
