# SPDX-License-Identifier: Apache-2.0
"""VideoGenerator hooks used by prompt-less, input-shaped pipelines (Kandinsky6 SR)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import fastvideo.entrypoints.video_generator as video_generator
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def _generator(args: FastVideoArgs, executor=None) -> VideoGenerator:
    generator = VideoGenerator.__new__(VideoGenerator)
    generator.fastvideo_args = args
    generator.executor = executor
    return generator


def _kandinsky6_ti2va_args(tmp_path: Path) -> FastVideoArgs:
    model = tmp_path / "kandinsky6-t2va"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({"_class_name": "Kandinsky6TI2VAPipeline",
                                                        "_diffusers_version": "0.35.0"}))
    return FastVideoArgs.from_kwargs(model_path=str(model), num_gpus=1)


def test_prompt_is_optional_for_the_sr_pipeline_and_the_output_is_named_after_the_video(fv_args, tmp_path, monkeypatch):
    seen = {}
    generator = _generator(fv_args)
    monkeypatch.setattr(generator, "_generate_single_video",
                        lambda prompt, sampling_param, fastvideo_args, **kw: seen.update(prompt=prompt, **kw) or {})
    sampling_param = SamplingParam(video_path="/data/clips/holiday.mp4", output_path=str(tmp_path / "out"))
    generator._generate_video_impl(prompt=None, sampling_param=sampling_param, fastvideo_args=fv_args)
    assert seen["prompt"] == ""
    assert Path(seen["output_path"]) == tmp_path / "out" / "holiday.mp4"
    # ... and a second run does not overwrite the first result
    (tmp_path / "out" / "holiday.mp4").write_bytes(b"x")
    generator._generate_video_impl(prompt=None, sampling_param=sampling_param, fastvideo_args=fv_args)
    assert Path(seen["output_path"]).name == "holiday_1.mp4"


def test_other_pipelines_still_require_a_prompt(tmp_path):
    args = _kandinsky6_ti2va_args(tmp_path)
    with pytest.raises(ValueError, match="Either prompt or prompt_txt"):
        _generator(args)._generate_video_impl(prompt=None, sampling_param=SamplingParam(video_path="clip.mp4"),
                                              fastvideo_args=args)


class _StubExecutor:

    def __init__(self, shape):
        self.shape = shape

    def execute_forward(self, batch, args):
        out = ForwardBatch(data_type="video")
        out.output = torch.rand(*self.shape)
        return out


def _run(generator, args, monkeypatch):
    big_allocations = []
    real_empty = torch.empty

    def spy(*shape, **kwargs):
        size = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        if len(size) == 5:
            big_allocations.append(tuple(size))
            return real_empty(0)
        return real_empty(*shape, **kwargs)

    monkeypatch.setattr(video_generator.torch, "empty", spy)
    sampling_param = SamplingParam(save_video=False, return_frames=True, height=720, width=1280, num_frames=125)
    result = generator._generate_single_video(prompt="", sampling_param=sampling_param, fastvideo_args=args,
                                              output_path="unused.mp4")
    return result, big_allocations


def test_sr_does_not_preallocate_the_return_buffer_from_the_request_geometry(fv_args, monkeypatch):
    shape = (1, 3, 9, 64, 96)
    result, allocations = _run(_generator(fv_args, _StubExecutor(shape)), fv_args, monkeypatch)
    assert allocations == []  # (1, 3, 125, 720, 1280) floats would be ~1.3 GB of pinned memory for nothing
    assert tuple(result["samples"].shape) == shape and result["size"] == (64, 96, 9)  # geometry read from the output


def test_other_pipelines_keep_the_preallocation(tmp_path, monkeypatch):
    args = _kandinsky6_ti2va_args(tmp_path)
    _, allocations = _run(_generator(args, _StubExecutor((1, 3, 9, 64, 96))), args, monkeypatch)
    assert allocations == [(1, 3, 125, 720, 1280)]


class _ExtraStubExecutor(_StubExecutor):
    """Returns ``extra`` the way a worker does: the worker's own ``batch.fps`` never comes back, only ``extra``."""

    def __init__(self, shape, extra):
        super().__init__(shape)
        self.extra = extra

    def execute_forward(self, batch, args):
        out = super().execute_forward(batch, args)
        out.extra.update(self.extra)
        return out


def _saved_fps(generator, args, monkeypatch, request_fps=24) -> int:
    saved = {}
    monkeypatch.setattr(generator, "_save_video_with_audio_ffmpeg_pipe", lambda **kw: saved.update(kw) or True)
    monkeypatch.setattr(video_generator.imageio, "mimsave", lambda path, frames, fps, format: saved.update(fps=fps))
    sampling_param = SamplingParam(save_video=True, return_frames=False, fps=request_fps)
    generator._generate_single_video(prompt="", sampling_param=sampling_param, fastvideo_args=args,
                                     output_path="unused.mp4")
    return saved["fps"]


@pytest.mark.parametrize("with_audio", [True, False], ids=["audio_single_pass", "video_only"])
def test_the_mp4_is_written_at_the_fps_reported_by_the_worker(fv_args, monkeypatch, with_audio):
    extra = {"output_fps": 16}
    if with_audio:
        extra.update(audio=torch.zeros(1600), audio_sample_rate=44100)
    generator = _generator(fv_args, _ExtraStubExecutor((1, 3, 9, 64, 96), extra))
    assert _saved_fps(generator, fv_args, monkeypatch, request_fps=24) == 16


def test_other_pipelines_keep_writing_at_the_request_fps(fv_args, monkeypatch):
    generator = _generator(fv_args, _ExtraStubExecutor((1, 3, 9, 64, 96), {}))
    assert _saved_fps(generator, fv_args, monkeypatch, request_fps=30) == 30


@pytest.mark.parametrize("legacy", [False, True])
def test_sr_options_reach_batch_extra_for_both_generator_apis(fv_args, tmp_path, monkeypatch, legacy):
    generator = _generator(fv_args)
    seen = {}
    monkeypatch.setattr(generator, "_generate_single_video",
                        lambda **kwargs: seen.update(kwargs) or {})
    monkeypatch.setattr(generator, "_wrap_legacy_result", lambda result: result)
    options = {"sr_resolution_scale": 4, "sr_tiles_batch_size": 2, "sr_tile_min_overlap": 0.3,
               "sr_target_resolution": "hd", "sr_target_resize_mode": "exact"}
    if legacy:
        generator.generate_video(video_path="clip.mp4", output_path=str(tmp_path), **options)
    else:
        generator._generate_single_request(video_generator.normalize_generation_request({
            "inputs": {"video_path": "clip.mp4"}, "output": {"output_path": str(tmp_path)},
            "extensions": options,
        }))
    assert seen["_extra_overrides"] == options
    assert all(not hasattr(seen["sampling_param"], key) for key in options)


def test_legacy_generator_rejects_sr_options_for_ti2va(tmp_path):
    args = _kandinsky6_ti2va_args(tmp_path)
    with pytest.raises(ValueError, match="only supported by Kandinsky6 SR"):
        _generator(args).generate_video(prompt="a cat", sr_resolution_scale=4)
