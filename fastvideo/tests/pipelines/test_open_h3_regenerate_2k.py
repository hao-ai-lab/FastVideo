# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Open-H3-Regenerate-2K building blocks."""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fastvideo.models.upsamplers.minimax_h3_latent_upscaler import (
    MiniMaxH3LatentUpscaler,
    resolve_upscale_geometry,
    upscale_latent,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k import draft_video
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sampling import (
    mix_initial_state,
    refine_sigmas,
    shifted_grid,
)
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sharp_curve import (
    NUM_CURVE_BRANCHES,
    SharpCurveBranches,
    interpolate_curve,
)


def test_default_sigmas_match_the_released_grid():
    assert refine_sigmas(3) == pytest.approx([0.909375, 0.857143, 0.666667, 0.0], abs=1e-6)


def test_two_step_sigmas_start_at_0_7():
    assert refine_sigmas(2) == pytest.approx([0.7, 0.494118, 0.0], abs=1e-6)


def test_shifted_grid_from_the_released_start_is_the_original_four_step_grid():
    assert shifted_grid(0.909375, 4) == pytest.approx([0.909375, 0.841446, 0.732075, 0.526697, 0.0], abs=1e-6)


def test_custom_start_sigma_is_decreasing_and_ends_at_zero():
    grid = refine_sigmas(3, start_sigma=0.8)
    assert grid[0] == 0.8 and grid[-1] == 0.0
    assert all(a > b for a, b in zip(grid, grid[1:]))


@pytest.mark.parametrize("steps", [0, 4])
def test_unsupported_step_count_is_rejected(steps):
    with pytest.raises(ValueError):
        refine_sigmas(steps)


def test_initial_state_mix_weights_noise_by_sigma():
    upscaled, noise = torch.ones(2, 3), torch.zeros(2, 3)
    assert torch.allclose(mix_initial_state(upscaled, noise, 0.909375), torch.full((2, 3), 0.090625))
    assert torch.equal(mix_initial_state(upscaled, noise, 0.0), upscaled)


def test_upscale_geometry_targets_2k_from_768p_and_576p():
    assert resolve_upscale_geometry((48, 84), (1440, 2560))[0] == (90, 160)
    shape, scale = resolve_upscale_geometry((36, 64), (1440, 2560))
    assert shape == (90, 160) and scale == pytest.approx(2.5)


def test_tiny_upscaler_forward_shape_and_identity_passthrough():
    model = MiniMaxH3LatentUpscaler(channels=32, in_blocks=2, out_blocks=2).eval()
    latent = torch.randn(24, 5, 4, 6)
    out = upscale_latent(model, latent, (96, 160), dtype=torch.float32)
    assert out.shape == (24, 5, 6, 10) and out.dtype == torch.float32
    assert torch.equal(upscale_latent(model, latent, (64, 96)), latent)


def test_upscaler_checkpoint_roundtrip(tmp_path):
    model = MiniMaxH3LatentUpscaler(channels=32, in_blocks=3, out_blocks=2)
    path = str(tmp_path / "upscaler.safetensors")
    save_file(model.state_dict(), path)
    loaded = MiniMaxH3LatentUpscaler.from_checkpoint(path)
    assert len(loaded.in_blocks) == len(model.in_blocks) and len(loaded.out_blocks) == len(model.out_blocks)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_curve_interpolation_hits_table_rows_and_midpoints():
    table = torch.arange(1025 * 8, dtype=torch.float32).reshape(1025, 8)
    coords = interpolate_curve(table, torch.tensor([0.0, 1.0, 0.5, 1 / 2048]))
    assert torch.equal(coords[0], table[0]) and torch.equal(coords[1], table[1024])
    assert torch.equal(coords[2], table[512])
    assert torch.allclose(coords[3], (table[0] + table[1]) / 2)
    with pytest.raises(ValueError):
        interpolate_curve(table, torch.tensor([1.5]))


class _Projection(torch.nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(4, out_features)

    def forward(self, x):
        return self.linear(x), None


def _write_curve_release(root, out_features, count):
    (root / "lora").mkdir()
    (root / "lora" / "adaln_t_table.bin").write_bytes(np.random.rand(1025 * 8).astype(np.float32).tobytes())
    tensors = {}
    for index in range(count):
        tensors[f"transformer.blocks.{index}.proj.curve_A.weight"] = torch.randn(3, 8)
        tensors[f"transformer.blocks.{index}.proj.curve_B.weight"] = torch.randn(out_features, 3)
    save_file(tensors, str(root / "lora" / "sharp_curve_branches.safetensors"))


def test_sharp_branches_require_all_fifty(tmp_path):
    _write_curve_release(tmp_path, 6, 3)
    with pytest.raises(ValueError, match="50"):
        SharpCurveBranches(tmp_path)


def test_sharp_branch_adds_curve_term_to_projection_output(tmp_path):
    _write_curve_release(tmp_path, 6, NUM_CURVE_BRANCHES)
    curve = SharpCurveBranches(tmp_path)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for _ in range(NUM_CURVE_BRANCHES):
        block = torch.nn.Module()
        block.proj = _Projection(6)
        model.blocks.append(block)
    curve.attach(model)
    curve.attach(model)
    timesteps = torch.tensor([0.1, 0.9])
    x = torch.randn(2, 4)
    base, _ = model.blocks[0].proj.linear(x), None
    curve.set_timesteps(timesteps)
    out, extra = model.blocks[0].proj(x)
    a, b = curve.factors["blocks.0.proj"]
    expected = base + torch.nn.functional.linear(torch.nn.functional.linear(curve.coords, a), b)
    assert extra is None and torch.allclose(out, expected, atol=1e-6)


def test_sharp_branch_without_timesteps_fails_loudly(tmp_path):
    _write_curve_release(tmp_path, 6, NUM_CURVE_BRANCHES)
    curve = SharpCurveBranches(tmp_path)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for _ in range(NUM_CURVE_BRANCHES):
        block = torch.nn.Module()
        block.proj = _Projection(6)
        model.blocks.append(block)
    curve.attach(model)
    with pytest.raises(RuntimeError, match="timesteps"):
        model.blocks[0].proj(torch.randn(2, 4))


def test_ffmpeg_command_cover_scales_and_caps_at_121_frames():
    command = draft_video.ffmpeg_decode_command("clip.mp4", 1344, 768)
    video_filter = command[command.index("-vf") + 1]
    assert "fps=24" in video_filter and "scale=1344:768:force_original_aspect_ratio=increase" in video_filter
    assert "crop=1344:768" in video_filter
    assert command[command.index("-frames:v") + 1] == "121"


def test_draft_pixels_pad_the_last_frame_to_124(monkeypatch):
    width, height = draft_video.DRAFT_SIZES["576p"]
    frames = np.arange(121, dtype=np.uint8).reshape(121, 1, 1, 1) * np.ones((121, height, width, 3), np.uint8)

    class _Done:
        stdout = frames.tobytes()

    monkeypatch.setattr(draft_video.subprocess, "run", lambda *a, **k: _Done())
    pixels = draft_video.load_draft_pixels("clip.mp4", "576p")
    assert pixels.shape == (1, 3, 124, height, width) and pixels.dtype == torch.uint8
    assert torch.equal(pixels[:, :, 120], pixels[:, :, 123])


def test_draft_pixels_reject_truncated_decode(monkeypatch):

    class _Short:
        stdout = b"\x00" * 10

    monkeypatch.setattr(draft_video.subprocess, "run", lambda *a, **k: _Short())
    with pytest.raises(ValueError, match="decoded to 10 bytes"):
        draft_video.load_draft_pixels("clip.mp4")


class _ConstantVelocityTransformer:
    """Predicts a constant video velocity and records what the stage feeds it."""

    def __init__(self, velocity):
        self.velocity = velocity
        self.calls = []
        self.attn_metadata = []

    def __call__(self, **kwargs):
        from fastvideo.forward_context import get_forward_context

        self.calls.append(kwargs)
        self.attn_metadata.append(get_forward_context().attn_metadata)
        rows = kwargs["hidden_states"].shape[1]
        audio_rows = kwargs["audio_hidden_states"].shape[1]
        return torch.full((1, rows, 8), self.velocity), torch.zeros(1, audio_rows, 4)


class _RecordingCurve:

    def __init__(self):
        self.timesteps = []
        self.attached = 0
        self.targets = []

    def attach(self, transformer):
        self.attached += 1
        self.targets.append(transformer)

    def set_timesteps(self, unique):
        self.timesteps.append(unique.clone())


def _refine_args(vsa_builder=None):
    from types import SimpleNamespace

    return SimpleNamespace(pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(patch_size=(1, 2, 2))),
                           VSA_sparsity=0.8,
                           VSA_tile_size=64,
                           vsa_builder=vsa_builder)


def _curve_model(out_features):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for _ in range(NUM_CURVE_BRANCHES):
        block = torch.nn.Module()
        block.proj = _Projection(out_features)
        model.blocks.append(block)
    return model


def test_sharp_branches_reach_a_reloaded_transformer_instance(tmp_path):
    _write_curve_release(tmp_path, 6, NUM_CURVE_BRANCHES)
    curve = SharpCurveBranches(tmp_path)
    curve.set_timesteps(torch.tensor([0.1, 0.9]))
    x = torch.randn(2, 4)
    for model in (_curve_model(6), _curve_model(6)):
        curve.attach(model)
        out, _ = model.blocks[0].proj(x)
        assert not torch.allclose(out, model.blocks[0].proj.linear(x))


def test_stage_attaches_the_branches_to_each_materialized_lazy_transformer(monkeypatch):
    from fastvideo.pipelines.lazy_module import LazyModule

    lazy = LazyModule("transformer", lambda: _ConstantVelocityTransformer(0.5))
    first = _denoise(monkeypatch, steps=2, transformer=lazy)[2]
    loaded_first = first.targets[-1]
    assert lazy.release()
    second = _denoise(monkeypatch, steps=2, transformer=lazy, curve=first)[2]

    assert not isinstance(loaded_first, LazyModule) and not isinstance(second.targets[-1], LazyModule)
    assert second.targets[-1] is not loaded_first


def _denoise(monkeypatch, steps, velocity=0.5, vsa_builder=None, transformer=None, curve=None, fastvideo_args=None):
    from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages import denoising
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(denoising, "_h3_vsa_metadata_builder", lambda *args: vsa_builder)
    monkeypatch.setattr(denoising, "_h3_vsa_single_region_segments", lambda layout, patch: ("segments", ))
    layout = build_packed_sequence(torch.ones(3, dtype=torch.long), 2, 2, 2, 2, (1, 2, 2))
    batch = ForwardBatch(data_type="video")
    batch.num_inference_steps = steps
    batch.prompt_embeds = [torch.zeros(1, 3, 4)]
    batch.latents = torch.zeros(layout.video_indices.numel(), 8)
    batch.audio_latents = torch.ones(layout.audio_indices.numel(), 4)
    batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
    transformer = transformer if transformer is not None else _ConstantVelocityTransformer(velocity)
    curve = curve if curve is not None else _RecordingCurve()
    stage = denoising.Regenerate2KDenoisingStage(transformer, curve, None)
    args = fastvideo_args if fastvideo_args is not None else _refine_args(vsa_builder)
    return stage.forward(batch, args), transformer, curve


def test_euler_steps_integrate_the_constant_velocity_over_sigma_zero_to_start(monkeypatch):
    batch, transformer, curve = _denoise(monkeypatch, steps=3, velocity=0.5)
    assert len(transformer.calls) == 3 and curve.attached == 1
    assert torch.allclose(batch.latents, torch.full_like(batch.latents, 0.909375 * 0.5))


def test_video_clock_is_one_minus_sigma_and_audio_is_held_clean(monkeypatch):
    _, transformer, curve = _denoise(monkeypatch, steps=3)
    for call, unique, sigma in zip(transformer.calls, curve.timesteps, [0.909375, 6 / 7, 2 / 3]):
        assert torch.allclose(unique.sort().values, torch.tensor([1.0 - sigma, 1.0]).sort().values.float(), atol=1e-6)
        assert call["hidden_states"].dtype == torch.bfloat16 and call["audio_hidden_states"].dtype == torch.bfloat16
        assert torch.equal(call["audio_hidden_states"].float(), torch.ones_like(call["audio_hidden_states"]).float())


def test_two_step_request_runs_two_forwards(monkeypatch):
    batch, transformer, _ = _denoise(monkeypatch, steps=2, velocity=1.0)
    assert len(transformer.calls) == 2
    assert torch.allclose(batch.latents, torch.full_like(batch.latents, 0.7), atol=1e-6)


def test_registry_resolves_the_release_to_the_regenerate_pipeline(tmp_path):
    from fastvideo.configs.pipelines.open_h3_regenerate_2k import OpenH3Regenerate2KPipelineConfig
    from fastvideo.registry import get_model_info, get_pipeline_config_cls_from_name, get_preset_selection

    hub_id = "Efficient-Large-Model/Open-H3-Regenerate-2K"
    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "open_h3_regenerate_2k_lora.safetensors").write_bytes(b"")
    for path in (hub_id, str(tmp_path)):
        assert get_pipeline_config_cls_from_name(path) is OpenH3Regenerate2KPipelineConfig
        assert get_model_info(path).pipeline_cls.__name__ == "OpenH3Regenerate2KPipeline"
    preset, family = get_preset_selection(hub_id)
    assert (preset, family) == ("open_h3_regenerate_2k", "open_h3_regenerate_2k")


def test_refine_transformer_dir_accepts_a_component_dir_or_its_checkpoint_parent(tmp_path):
    from fastvideo.pipelines.basic.open_h3_regenerate_2k.open_h3_regenerate_2k_pipeline import (
        resolve_refine_transformer_dir,
    )

    assert resolve_refine_transformer_dir(None) is None
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "config.json").write_text("{}")
    assert resolve_refine_transformer_dir(str(component)) == str(component)
    assert resolve_refine_transformer_dir(str(tmp_path)) == str(component)
    with pytest.raises(ValueError, match="config.json"):
        resolve_refine_transformer_dir(str(tmp_path / "missing"))


def _preparation_stage(tmp_path, draft_latent_path=None):
    from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages.latent_preparation import (
        Regenerate2KLatentPreparationStage,
    )

    return Regenerate2KLatentPreparationStage(None,
                                              None,
                                              None,
                                              load_upscaler=lambda: None,
                                              draft_size="768p",
                                              start_sigma=None,
                                              draft_latent_path=draft_latent_path)


def _preparation_batch():
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    batch = ForwardBatch(data_type="video")
    batch.prompt_embeds = [torch.zeros(1, 3, 4)]
    batch.raw_latent_shape = (1, 24, 31, 90, 160)
    batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = torch.zeros(3, dtype=torch.long)
    return batch


def test_draft_latent_input_removes_the_draft_video_requirement(tmp_path):
    video_batch = _preparation_batch()
    assert not _preparation_stage(tmp_path).verify_input(video_batch, None).is_valid()

    latent_path = tmp_path / "draft_latent.pt"
    torch.save(torch.zeros(24, 31, 48, 84), latent_path)
    assert _preparation_stage(tmp_path, str(latent_path)).verify_input(_preparation_batch(), None).is_valid()


def test_a_pt_draft_video_path_is_a_per_request_draft_latent(tmp_path):
    stage = _preparation_stage(tmp_path)
    assert stage._draft_latent_source("/drafts/fox.mp4") is None
    assert stage._draft_latent_source("/drafts/fox_latent.PT") == "/drafts/fox_latent.PT"
    configured = _preparation_stage(tmp_path, "/drafts/configured.pt")
    assert configured._draft_latent_source("/drafts/other.pt") == "/drafts/configured.pt"


@pytest.mark.parametrize("shape", [(24, 31, 48, 84), (1, 24, 31, 48, 84)])
def test_saved_draft_latent_loads_as_a_native_four_dimensional_latent(tmp_path, shape):
    path = tmp_path / "draft_latent.pt"
    torch.save(torch.full(shape, 0.5), path)
    loaded = _preparation_stage(tmp_path, str(path))._load_draft_latent(str(path), torch.device("cpu"))
    assert loaded.shape == (24, 31, 48, 84) and loaded.dtype == torch.float32
    assert torch.all(loaded == 0.5)


def test_saved_draft_latent_rejects_a_batch_of_several_drafts(tmp_path):
    path = tmp_path / "draft.pt"
    torch.save(torch.zeros(2, 24, 31, 24, 42), path)
    with pytest.raises(ValueError, match=r"\[1, C, T, h, w\]"):
        _preparation_stage(tmp_path, str(path))._load_draft_latent(str(path), torch.device("cpu"))


def test_absolute_module_mapping_dispatches_on_its_role():
    from types import SimpleNamespace

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

    pipeline = SimpleNamespace(model_path="/release",
                               _extra_config_module_map={"transformer": "/models/my_refiner", "vae": "base_model/vae"})

    assert ComposedPipelineBase._component_load_target(pipeline, "transformer") == ("transformer", "/models/my_refiner")
    assert ComposedPipelineBase._component_load_target(pipeline, "vae") == ("base_model/vae", "/release/base_model/vae")
    assert ComposedPipelineBase._component_load_target(pipeline, "scheduler") == ("scheduler", "/release/scheduler")


def test_saved_draft_latent_rejects_the_wrong_rank(tmp_path):
    path = tmp_path / "draft_latent.pt"
    torch.save(torch.zeros(24, 31, 48), path)
    with pytest.raises(ValueError, match=r"\[C, T, h, w\]"):
        _preparation_stage(tmp_path, str(path))._load_draft_latent(str(path), torch.device("cpu"))


class _RecordingVSABuilder:

    def __init__(self):
        self.builds = []

    def build(self, **kwargs):
        self.builds.append(kwargs)
        return f"metadata-{len(self.builds)}"


def test_vsa_backend_builds_one_sparse_metadata_per_refine_step(monkeypatch):
    builder = _RecordingVSABuilder()
    _, transformer, _ = _denoise(monkeypatch, steps=3, vsa_builder=builder)
    assert [build["VSA_sparsity"] for build in builder.builds] == [0.8, 0.8, 0.8]
    assert all(build["tile_size"] == 64 and build["exempt"] for build in builder.builds)
    assert [build["current_timestep"] for build in builder.builds] == [0, 1, 2]
    assert transformer.attn_metadata == ["metadata-1", "metadata-2", "metadata-3"]


def test_dense_backend_passes_no_attention_metadata(monkeypatch):
    _, transformer, _ = _denoise(monkeypatch, steps=3)
    assert transformer.attn_metadata == [None, None, None]


class _OffloadedTransformer(_ConstantVelocityTransformer):

    def __init__(self, velocity, fail=False):
        super().__init__(velocity)
        self.fail = fail
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self

    def __call__(self, **kwargs):
        if self.fail:
            raise RuntimeError("forward failed")
        return super().__call__(**kwargs)


@pytest.mark.parametrize("fail", [False, True])
def test_full_cpu_offload_moves_the_transformer_back_to_cpu_after_the_refine(monkeypatch, fail):
    from types import SimpleNamespace

    transformer = _OffloadedTransformer(0.5, fail=fail)
    args = SimpleNamespace(dit_cpu_offload=True, dit_layerwise_offload=False, use_fsdp_inference=False)
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            _denoise(monkeypatch, steps=2, transformer=transformer, fastvideo_args=args)
    else:
        _denoise(monkeypatch, steps=2, transformer=transformer, fastvideo_args=args)

    assert transformer.moves == ["cpu", "cpu"]  # to the stage device (CPU in this test), then back to the host


@pytest.mark.parametrize("variable, value", [("FASTVIDEO_H3_ADALN_CACHE", True),
                                              ("FASTVIDEO_H3_ADALN_TABLE", "/tmp/adaln_tables.pt")])
def test_pipeline_refuses_an_adaln_cache_that_would_skip_the_sharp_branches(variable, value):
    from fastvideo import envs
    from fastvideo.pipelines.basic.open_h3_regenerate_2k.open_h3_regenerate_2k_pipeline import refuse_adaln_cache

    refuse_adaln_cache()
    with getattr(envs, variable).override(value), pytest.raises(ValueError, match="Sharp curve branches"):
        refuse_adaln_cache()
