# SPDX-License-Identifier: Apache-2.0
"""DreamX-World pipeline config and conditioning smoke tests."""

from types import SimpleNamespace
import json

import numpy as np
import torch

from fastvideo.api.presets import get_preset
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.pipeline_registry import PipelineType, import_pipeline_classes
from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import (
    DreamXCamera,
    _interpolate_camera_poses,
    build_dreamx_camera_condition,
)
from fastvideo.configs.pipelines.dreamx_world import DreamXWorld5BARPipelineConfig, DreamXWorld5BCamPipelineConfig
from fastvideo.pipelines.basic.dreamx_world.ar_denoising import DreamXWorldARCausalDenoisingStage
from fastvideo.pipelines.basic.dreamx_world.dreamx_world_ar_pipeline import DreamXWorldARPipeline
from fastvideo.pipelines.basic.dreamx_world.dreamx_world_pipeline import DreamXWorldPipeline
from fastvideo.pipelines.basic.dreamx_world.stages import (
    DREAMX_Y_CAMERA_KEY,
    DreamXWorldCameraConditioningStage,
)
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.registry import get_default_preset, get_model_info, get_pipeline_config_cls_from_name


def test_dreamx_world_5b_cam_pipeline_config_wires_first_scope_components():
    config = DreamXWorld5BCamPipelineConfig()

    assert config.flow_shift == 3.0
    assert config.ti2v_task is True
    assert config.expand_timesteps is True
    assert config.dit_config.expand_timesteps is True
    assert config.dit_config.num_layers == 30
    assert config.dit_config.add_control_adapter is True
    assert config.dit_config.cam_method == "prope"

    assert config.vae_config.load_encoder is True
    assert config.vae_config.load_decoder is True
    assert config.vae_config.z_dim == 48
    assert config.vae_config.scale_factor_temporal == 4
    assert config.vae_config.scale_factor_spatial == 16

    assert len(config.text_encoder_configs) == 1
    text_config = config.text_encoder_configs[0]
    assert text_config.prefix == "umt5"
    assert text_config.vocab_size == 256384
    assert text_config.d_model == 4096
    assert config.text_encoder_precisions == ("bf16",)


def test_dreamx_world_pipeline_registry_discovers_entrypoint():
    pipelines = import_pipeline_classes(PipelineType.BASIC)

    assert pipelines["basic"]["DreamXWorldPipeline"] is DreamXWorldPipeline




def test_dreamx_world_local_model_index_resolves_model_info(tmp_path):
    model_dir = tmp_path / "DreamX-World-5B-Cam-converted"
    model_dir.mkdir()
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (model_dir / component).mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({
        "_class_name": "DreamXWorldPipeline",
        "_diffusers_version": "0.31.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "transformer": ["diffusers", "DreamXWorldTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }) + "\n")

    info = get_model_info(str(model_dir), pipeline_type=PipelineType.BASIC, workload_type=WorkloadType.I2V)

    assert info.pipeline_cls is DreamXWorldPipeline
    assert info.pipeline_config_cls is DreamXWorld5BCamPipelineConfig

def test_dreamx_world_model_path_resolves_pipeline_config():
    assert get_pipeline_config_cls_from_name("GD-ML/DreamX-World-5B-Cam") is DreamXWorld5BCamPipelineConfig



def test_dreamx_world_default_preset_is_registered():
    preset_name = get_default_preset("GD-ML/DreamX-World-5B-Cam")
    preset = get_preset(preset_name, "dreamx_world")

    assert preset.name == "dreamx_world_5b_cam"
    assert preset.workload_type == "i2v"
    assert preset.defaults["height"] == 480
    assert preset.defaults["width"] == 832
    assert preset.defaults["num_frames"] == 161
    assert preset.defaults["num_inference_steps"] == 30
    assert preset.defaults["guidance_scale"] == 5.0




def test_dreamx_world_pipeline_initializes_official_flow_scheduler():
    pipeline = DreamXWorldPipeline.__new__(DreamXWorldPipeline)
    pipeline.modules = {}
    fastvideo_args = SimpleNamespace(pipeline_config=DreamXWorld5BCamPipelineConfig())

    pipeline.initialize_pipeline(fastvideo_args)

    scheduler = pipeline.modules["scheduler"]
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler.config.shift == 3.0

def test_dreamx_world_camera_conditioning_stage_sets_y_camera_extra():
    batch = ForwardBatch(
        data_type="t2v",
        action_list=["wj", "d"],
        action_speed_list=[4, 6],
        num_frames=17,
        height=704,
        width=1280,
        latents=torch.zeros(1, 16, 5, 44, 80),
    )
    stage = DreamXWorldCameraConditioningStage()

    out = stage.forward(batch, fastvideo_args=object())
    y_camera = out.extra[DREAMX_Y_CAMERA_KEY]
    expected = build_dreamx_camera_condition(
        ["wj", "d"],
        [4, 6],
        num_frames=17,
        height=704,
        width=1280,
        dtype=torch.float32,
        device="cpu",
    )

    assert set(y_camera) == {"viewmats", "K"}
    for key, expected_value in expected.items():
        assert y_camera[key].shape == (1, *expected_value.shape)
        torch.testing.assert_close(y_camera[key][0], expected_value)
    assert stage.verify_output(out, fastvideo_args=object()).is_valid()


def test_dreamx_world_denoising_kwargs_filter_for_y_camera():
    y_camera = {"viewmats": torch.eye(4).reshape(1, 1, 4, 4), "K": torch.eye(3).reshape(1, 1, 3, 3)}
    stage = DenoisingStage.__new__(DenoisingStage)

    def accepts_y_camera(hidden_states, encoder_hidden_states, timestep, y_camera=None):
        return y_camera

    def no_y_camera(hidden_states, encoder_hidden_states, timestep):
        return hidden_states

    assert stage.prepare_extra_func_kwargs(accepts_y_camera, {"y_camera": y_camera}) == {"y_camera": y_camera}
    assert stage.prepare_extra_func_kwargs(no_y_camera, {"y_camera": y_camera}) == {}



def test_dreamx_world_ar_pipeline_config_wires_components():
    config = DreamXWorld5BARPipelineConfig()
    assert config.is_causal is True
    assert config.flow_shift == 5.0
    assert config.dmd_denoising_steps == (1000, 750, 500, 250)
    assert config.warp_denoising_step is True
    assert config.context_noise == 0.1
    assert config.dit_config.arch_config.local_attn_size == 12
    assert config.dit_config.arch_config.sink_size == 3
    assert config.dit_config.arch_config.attn_compress == 4


def test_dreamx_world_ar_pipeline_registry_and_preset():
    from fastvideo.api.presets import get_preset
    from fastvideo.registry import get_pipeline_config_cls_from_name, get_preset_selection

    assert get_pipeline_config_cls_from_name("GD-ML/DreamX-World-5B") is DreamXWorld5BARPipelineConfig
    preset_name, family = get_preset_selection("GD-ML/DreamX-World-5B")
    assert (preset_name, family) == ("dreamx_world_5b_ar", "dreamx_world")
    preset = get_preset("dreamx_world_5b_ar", "dreamx_world")
    assert preset.defaults["num_inference_steps"] == 4
    assert DreamXWorldARPipeline.pipeline_config_cls is DreamXWorld5BARPipelineConfig


def test_dreamx_world_camera_conditioning_stage_expands_scalar_speed():
    batch = ForwardBatch(
        data_type="t2v",
        action_list=["w", "d"],
        action_speed_list=2.0,
        num_frames=17,
        height=704,
        width=1280,
        latents=torch.zeros(1, 16, 5, 44, 80),
    )
    stage = DreamXWorldCameraConditioningStage()

    out = stage.forward(batch, fastvideo_args=object())

    assert set(out.extra[DREAMX_Y_CAMERA_KEY]) == {"viewmats", "K"}


def test_dreamx_world_camera_interpolation_handles_single_camera():
    camera = DreamXCamera(
        fx=0.8,
        fy=0.8,
        cx=0.5,
        cy=0.5,
        w2c_mat=np.eye(4, dtype=np.float64),
    )

    out = _interpolate_camera_poses(
        [camera],
        src_indices=np.array([0.0]),
        tgt_indices=np.array([0.0, 1.0, 2.0]),
    )

    assert out == [camera, camera, camera]


def test_dreamx_world_ar_cache_initializes_camera_self_attention_entries():
    cam_self_attn = SimpleNamespace(num_heads=3, head_dim=5)
    transformer = SimpleNamespace(
        blocks=[SimpleNamespace(cam_self_attn=cam_self_attn), SimpleNamespace(cam_self_attn=cam_self_attn)],
        num_attention_heads=2,
        attention_head_dim=4,
    )
    stage = DreamXWorldARCausalDenoisingStage.__new__(DreamXWorldARCausalDenoisingStage)
    stage.transformer = transformer
    stage.num_transformer_blocks = 2
    stage.local_attn_size = 6

    caches = stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        frame_seq_length=7,
    )

    assert len(caches) == 2
    assert caches[0]["k"].shape == (1, 42, 2, 4)
    assert caches[0]["prope_k"].shape == (1, 42, 3, 5)
    assert caches[0]["prope_v"].shape == (1, 42, 3, 5)
    assert int(caches[0]["prope_global_end_index"].item()) == 0
    assert int(caches[0]["prope_local_end_index"].item()) == 0


def test_dreamx_world_ar_context_noise_fraction_maps_to_scheduler_timestep():
    assert DreamXWorldARCausalDenoisingStage._context_noise_timestep(0.1) == 100
    assert DreamXWorldARCausalDenoisingStage._context_noise_timestep(100) == 100



def test_dreamx_world_ar_context_update_advances_camera_cache_indices():
    class DummyTransformer:
        def __call__(self, *, hidden_states, encoder_hidden_states, timestep, y_camera, kv_cache, crossattn_cache,
                     current_start):
            del encoder_hidden_states, y_camera, crossattn_cache
            assert current_start == 0
            assert timestep.unique().tolist() == [100]
            new_tokens = timestep.shape[1]
            for cache in kv_cache:
                cache["local_end_index"] += new_tokens
                cache["global_end_index"] += new_tokens
                cache["prope_local_end_index"] += new_tokens
                cache["prope_global_end_index"] += new_tokens
                cache["k"][:, :new_tokens] = 1
                cache["prope_k"][:, :new_tokens] = 1
            return hidden_states

    cam_self_attn = SimpleNamespace(num_heads=3, head_dim=5)
    cache_transformer = SimpleNamespace(
        blocks=[SimpleNamespace(cam_self_attn=cam_self_attn)],
        num_attention_heads=2,
        attention_head_dim=4,
    )
    stage = DreamXWorldARCausalDenoisingStage.__new__(DreamXWorldARCausalDenoisingStage)
    stage.transformer = cache_transformer
    stage.num_transformer_blocks = 1
    stage.local_attn_size = 6
    caches = stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        frame_seq_length=2,
    )
    # Keep the cache allocation source separate from the callable transformer used by _update_context_cache.
    stage.transformer = DummyTransformer()

    stage._update_context_cache(
        block_latents=torch.zeros(1, 4, 3, 2, 2),
        context=[torch.zeros(2, 4)],
        camera_block={"viewmats": torch.eye(4).reshape(1, 1, 4, 4), "K": torch.eye(3).reshape(1, 1, 3, 3)},
        kv_cache=caches,
        crossattn_cache=[{}],
        start=0,
        frame_seq_length=2,
        target_dtype=torch.float32,
        autocast_enabled=False,
        context_noise=0.1,
    )

    assert int(caches[0]["local_end_index"].item()) == 6
    assert int(caches[0]["prope_local_end_index"].item()) == 6
    assert caches[0]["k"][:, :6].sum().item() == 48
    assert caches[0]["prope_k"][:, :6].sum().item() == 90
