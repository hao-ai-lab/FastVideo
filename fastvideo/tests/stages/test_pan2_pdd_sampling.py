# SPDX-License-Identifier: Apache-2.0
"""PAN2 PDD students at inference, on CPU: the export's fastvideo_inference.json and the fused-block sampler.

``build_pan2_pdd_inference_file`` writes the contract a PAN2 PDD export carries and ``read_pan2_pdd_inference_file``
reads it back, rejecting drift in any field or from the checkpoint files it repeats. ``PAN2PipelineConfig`` applies it
to a run (partition, sparse-attention settings) and to each request (step count, no guidance). ``PAN2PDDDenoisingStage``
samples a tiny student as an explicit Euler integration of its unfused fine-grid heads does.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
import torch

from fastvideo import envs
from fastvideo.api.parser import parse_config
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import GenerationRequest
from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig, PAN2VideoConfig
from fastvideo.configs.pipelines.pan2 import (PAN2PipelineConfig, build_pan2_pdd_inference_file,
                                              read_pan2_pdd_inference_file)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.pdd import PDDModalitySchedule, build_pdd_sampling_plan
from fastvideo.models.dits.pan2 import PAN2Transformer3DModel
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.basic.pan2.pdd_denoising import PAN2PDDDenoisingStage, pan2_pdd_transformer_timestep
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

# The DMPDD recipe's Grid32 student in 4-head blocks: eight fused forwards.
_TRAINED = {
    "pdd_steps": 32,
    "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "num_inference_steps": 8,
    "transformer_forwards": 8,
    "grid_max_t": 0.999,
    "video_scheduler_shift": 9.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "vsa_tile_policy": ["first", "center", "center"],
    "vsa_compress_text_keys": False,
}


@pytest.fixture
def no_backend_env(env_overrides):
    """FastVideoArgs reads FASTVIDEO_ATTENTION_BACKEND as the run's explicit backend; start without it."""
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))


def _checkpoint(root: Path, inference_file: dict | None, *, pdd_steps: int | None = 32, shift: float = 9.0) -> str:
    """A PAN2 checkpoint's JSON files: the inference file and the configs that own its repeated values."""
    files = {
        "transformer/config.json": {} if pdd_steps is None else {"pdd_steps": pdd_steps},
        "scheduler/scheduler_config.json": {"shift": shift},
    }
    if inference_file is not None:
        files["fastvideo_inference.json"] = inference_file
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    return str(root)


# ------------------------------------------------- the contract file


def test_contract_round_trips_through_a_checkpoint(tmp_path) -> None:
    contract = build_pan2_pdd_inference_file(**_TRAINED)
    assert contract["guidance_scale"] == 1.0 and contract["conditioning"] == "first_latent_frame"
    assert read_pan2_pdd_inference_file(_checkpoint(tmp_path, contract)) == contract
    assert read_pan2_pdd_inference_file(_checkpoint(tmp_path / "base", None)) is None


@pytest.mark.parametrize("field,value,match", [
    ("pdd_step_indices", [0, 8, 4, 32], "increase strictly"),
    ("pdd_step_indices", [0, 4, 8, 12, 16, 20, 24, 28], "increase strictly"),
    ("num_inference_steps", 7, "disagrees with the 8 blocks"),
    ("transformer_forwards", 16, "disagrees with the 8 blocks"),
    ("grid_max_t", 1.0, "PDD fine grid ends at 0.999"),
    ("video_scheduler_shift", 0.0, "positive and finite"),
    ("attention_backend", "NOT_A_BACKEND", "not an attention backend"),
    ("vsa_sparsity", 1.0, r"in \[0, 1\)"),
    ("vsa_tile_size", 96, "must be one of"),
    ("vsa_tile_policy", ["first", "center"], r"per \(t, h, w\) axis"),
    ("vsa_compress_text_keys", "no", "must be a bool"),
    ("vsa_variant", "vsa3", "vsa_variant='vsa3' must be one of"),
    ("vsa_dense_layers", [43, 0], "increasing order"),
    ("vsa_dense_layers", [0, 0], "increasing order"),
    ("vsa_dense_layers", [-1], "increasing order"),
    ("vsa_dense_layers", [True], "increasing order"),
    ("vsa_dense_layers", "0,43", "increasing order"),
])
def test_contract_rejects_a_drifted_trained_field(tmp_path, field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_pan2_pdd_inference_file(**{**_TRAINED, field: value})
    drifted = {**build_pan2_pdd_inference_file(**_TRAINED), field: value}
    with pytest.raises(ValueError, match=match):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path, drifted))


@pytest.mark.parametrize("field,value", [("guidance_scale", 3.0), ("conditioning", "none"),
                                         ("precision_decoding", "bfloat16"), ("model_type", "minimax_h3")])
def test_contract_rejects_another_code_fixed_value(tmp_path, field, value) -> None:
    with pytest.raises(ValueError, match="fixed"):
        build_pan2_pdd_inference_file(**_TRAINED, **{field: value})
    drifted = {**build_pan2_pdd_inference_file(**_TRAINED), field: value}
    with pytest.raises(ValueError, match="is unsupported"):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path, drifted))


def test_contract_needs_exactly_its_fields_and_the_files_it_repeats(tmp_path) -> None:
    contract = build_pan2_pdd_inference_file(**_TRAINED)
    with pytest.raises(ValueError, match=r"missing \['vsa_sparsity'\]"):
        build_pan2_pdd_inference_file(**{key: value for key, value in _TRAINED.items() if key != "vsa_sparsity"})
    with pytest.raises(ValueError, match=r"unknown \['seed'\]"):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path / "unknown", {**contract, "seed": 1}))
    with pytest.raises(ValueError, match="pdd_steps=32 disagrees with transformer/config.json pdd_steps=16"):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path / "grid", contract, pdd_steps=16))
    with pytest.raises(ValueError, match="video_scheduler_shift=9.0 disagrees with scheduler/scheduler_config.json"):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path / "shift", contract, shift=5.0))
    with pytest.raises(ValueError, match="not a PAN2 PDD inference file"):
        read_pan2_pdd_inference_file(_checkpoint(tmp_path / "other", {"schema": "fasth3-inference-contract-v1"}))


# ------------------------------------------------- run and request settings


def _run_args(model_path: str, config_settings: dict | None = None, **run_settings) -> FastVideoArgs:
    """A run's FastVideoArgs; construction resolves the checkpoint's settings."""
    return FastVideoArgs(model_path=model_path, pipeline_config=PAN2PipelineConfig(**(config_settings or {})),
                         **run_settings)


def test_a_sparse_student_fixes_the_partition_and_its_attention(tmp_path, no_backend_env) -> None:
    args = _run_args(_checkpoint(tmp_path, build_pan2_pdd_inference_file(**_TRAINED)))
    config = args.pipeline_config
    assert config.pdd_step_indices == tuple(_TRAINED["pdd_step_indices"])
    assert config.vsa_tile_policy == ("first", "center", "center") and config.vsa_compress_text_keys is False
    assert (args.attention_backend, args.VSA_tile_size, args.VSA_sparsity) == ("VIDEO_SPARSE_ATTN_H3", 128, 0.9)
    # A file without a variant was written for a VSA-H3 student.
    assert config.vsa_variant == "vsa"


def test_a_vsa2_student_samples_with_vsa2(tmp_path, no_backend_env) -> None:
    trained = {**_TRAINED, "vsa_tile_policy": ["first", "last", "last"], "vsa_compress_text_keys": True}
    args = _run_args(_checkpoint(tmp_path, build_pan2_pdd_inference_file(**trained, vsa_variant="vsa2")))
    config = args.pipeline_config
    assert (config.vsa_variant, config.vsa_tile_policy, config.vsa_compress_text_keys) == ("vsa2", ("first", "last",
                                                                                                    "last"), True)
    assert (args.attention_backend, args.VSA_tile_size) == ("VIDEO_SPARSE_ATTN_H3", 128)


def test_dense_vsa_layers_come_from_the_file(tmp_path, no_backend_env) -> None:
    dense = build_pan2_pdd_inference_file(**_TRAINED, vsa_dense_layers=[0, 43])
    assert _run_args(_checkpoint(tmp_path / "dense", dense)).pipeline_config.vsa_dense_layers == (0, 43)
    # A file without the field was written for a student with every block sparse.
    sparse = build_pan2_pdd_inference_file(**_TRAINED)
    assert _run_args(_checkpoint(tmp_path / "sparse", sparse)).pipeline_config.vsa_dense_layers == ()


def test_a_run_may_retune_sparsity_but_not_the_trained_layout(tmp_path, no_backend_env, caplog) -> None:
    path = _checkpoint(tmp_path, build_pan2_pdd_inference_file(**_TRAINED))
    with caplog.at_level(logging.WARNING):
        assert _run_args(path, VSA_sparsity=0.8).VSA_sparsity == 0.8
    assert "trained with VSA_sparsity=0.9" in caplog.text
    with pytest.raises(ValueError, match="trained with VSA_tile_size=128"):
        _run_args(path, VSA_tile_size=64)
    with pytest.raises(ValueError, match="trained with attention_backend=VIDEO_SPARSE_ATTN_H3"):
        _run_args(path, attention_backend="TORCH_SDPA")
    with pytest.raises(ValueError, match="pdd_step_indices comes from"):
        _run_args(path, {"pdd_step_indices": (0, 16, 32)})
    with pytest.raises(ValueError, match="trained with flow shift 9.0"):
        _run_args(path, {"flow_shift": 5.0})


def test_a_dense_student_keeps_the_runs_attention(tmp_path, no_backend_env) -> None:
    dense = build_pan2_pdd_inference_file(**{**_TRAINED, "attention_backend": "FLASH_ATTN", "vsa_sparsity": 0.0})
    args = _run_args(_checkpoint(tmp_path, dense))
    assert args.pipeline_config.pdd_step_indices == tuple(_TRAINED["pdd_step_indices"])
    assert (args.attention_backend, args.VSA_tile_size, args.VSA_sparsity) == (None, 256, 0.0)


def test_a_checkpoint_without_the_file_is_sampled_as_before(tmp_path, no_backend_env) -> None:
    path = _checkpoint(tmp_path, None, pdd_steps=None)
    assert _run_args(path).pipeline_config.pdd_step_indices is None
    with pytest.raises(ValueError, match="applies only to PAN2 PDD checkpoints"):
        _run_args(path, {"pdd_step_indices": (0, 16, 32)})


def test_a_request_samples_the_trained_blocks_without_guidance() -> None:
    config = PAN2PipelineConfig(pdd_step_indices=tuple(_TRAINED["pdd_step_indices"]))
    request = parse_config(GenerationRequest, {"prompt": "fox"})
    sampled = config.apply_request_constraints(request, SamplingParam(num_inference_steps=50, guidance_scale=3.0))
    assert (sampled.num_inference_steps, sampled.guidance_scale) == (8, 1.0)
    request = parse_config(GenerationRequest, {"prompt": "fox", "sampling": {"num_inference_steps": 8}})
    assert config.apply_request_constraints(request, SamplingParam()).num_inference_steps == 8
    with pytest.raises(ValueError, match="runs exactly 8 transformer forwards"):
        config.apply_request_constraints(
            parse_config(GenerationRequest, {
                "prompt": "fox",
                "sampling": {
                    "num_inference_steps": 50
                }
            }), SamplingParam())
    with pytest.raises(ValueError, match="without classifier-free guidance"):
        config.apply_request_constraints(
            parse_config(GenerationRequest, {
                "prompt": "fox",
                "sampling": {
                    "guidance_scale": 3.0
                }
            }), SamplingParam())
    base = SamplingParam(num_inference_steps=50, guidance_scale=3.0)
    assert PAN2PipelineConfig().apply_request_constraints(request, base) is base


# ------------------------------------------------- the sampler

_GRID, _LATENT_CHANNELS = 8, 4
_ARCH = dict(
    patch_size=(1, 2, 2),
    in_channels=2 * _LATENT_CHANNELS + 1,
    out_channels=_LATENT_CHANNELS,
    num_attention_heads=4,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    context_mlp_ratio=2.0,
    refiner_mlp_ratio=2.0,
    text_embed_dim=16,
    rope_axes_dim=(4, 6, 6),
)


@pytest.fixture()
def single_process_group(env_overrides):
    """A 1x1 sequence/tensor-parallel group on one CPU process, with the dense Torch SDPA backend."""
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    env_overrides.enter_context(envs.override_external("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", os.environ.get("MASTER_PORT") or "29597"))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


def _student() -> PAN2Transformer3DModel:
    arch = PAN2ArchConfig(**_ARCH, pdd_steps=_GRID)
    model = PAN2Transformer3DModel(PAN2VideoConfig(arch_config=arch), hf_config={})
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model.eval()


@pytest.mark.parametrize("task", ["i2v", "t2v"])
def test_the_sampler_integrates_the_fused_heads_of_each_block(monkeypatch, task, single_process_group) -> None:
    """Each step advances the fp32 state by the weighted sum of its block's heads at the block's first node."""
    student = _student()
    indices = (0, 2, 4, 8)  # unequal blocks of the 8-interval grid
    shift = 9.0
    generator = torch.Generator().manual_seed(5)
    latents = torch.randn(1, _LATENT_CHANNELS, 3, 4, 6, generator=generator)
    condition = torch.zeros(1, _LATENT_CHANNELS + 1, 3, 4, 6)
    if task == "i2v":
        condition[:, :_LATENT_CHANNELS, :1] = torch.randn(1, _LATENT_CHANNELS, 1, 4, 6, generator=generator)
        condition[:, _LATENT_CHANNELS:, :1] = 1.0
    prompt = torch.randn(1, 5, _ARCH["text_embed_dim"], generator=generator)

    # The integration written out: unfused heads, weighted by the plan, at each block's first node.
    plan = build_pdd_sampling_plan(indices, {"video": PDDModalitySchedule(shift)})
    weights = plan.integration_weights["video"]
    state = latents.float() * plan.node_sigmas["video"][:1].float()
    expected_timesteps = []
    for step in range(plan.num_steps):
        start, end = plan.block(step)
        u = float(plan.fine_grid[start])
        # The rational shift with the PDD grid's end M = 0.999 as its fixed point: s * u * M / (u * (s - 1) + M).
        expected_timesteps.append(shift * u * 0.999 / (u * (shift - 1) + 0.999) * 1000)
        timestep = pan2_pdd_transformer_timestep(plan.fine_grid[start:start + 1], shift)
        model_input = torch.cat([state.to(torch.bfloat16), condition.to(torch.bfloat16)], dim=1)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), set_forward_context(
                current_timestep=timestep, attn_metadata=None):
            heads = student(hidden_states=model_input,
                            encoder_hidden_states=prompt.to(torch.bfloat16),
                            timestep=timestep).float().unflatten(1, (_GRID, _LATENT_CHANNELS))
        state = state + torch.einsum("n,bnc...->bc...", weights[start:end].float(), heads[:, start:end])

    pipeline_config = PAN2PipelineConfig(dit_config=PAN2VideoConfig(arch_config=PAN2ArchConfig(**_ARCH,
                                                                                               pdd_steps=_GRID)),
                                         pdd_step_indices=indices)
    fastvideo_args = type("Args", (), {})()
    fastvideo_args.pipeline_config = pipeline_config
    fastvideo_args.dit_cpu_offload = False
    fastvideo_args.dit_layerwise_offload = False
    fastvideo_args.use_fsdp_inference = False
    fastvideo_args.VSA_tile_size = 128
    monkeypatch.setattr("fastvideo.pipelines.basic.pan2.pdd_denoising.get_local_torch_device",
                        lambda: torch.device("cpu"))
    stage = PAN2PDDDenoisingStage(transformer=student, scheduler=FlowMatchEulerDiscreteScheduler(shift=shift))
    request = ForwardBatch(data_type="video",
                           latents=latents.clone(),
                           image_latent=condition,
                           prompt_embeds=[prompt],
                           guidance_scale=1.0,
                           num_inference_steps=len(indices) - 1)
    sampled = stage.forward(request, fastvideo_args)

    assert sampled.latents.shape == latents.shape and sampled.latents.dtype == torch.float32
    # The fused head rounds once to bf16 where the written-out sum rounds each head: equal up to bf16 rounding.
    torch.testing.assert_close(sampled.latents, state, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sampled.timesteps.double(), torch.tensor(expected_timesteps, dtype=torch.float64),
                               rtol=1e-6, atol=1e-3)


def test_the_sampler_refuses_guidance_and_non_pdd_checkpoints(single_process_group) -> None:
    stage = PAN2PDDDenoisingStage(transformer=_student(), scheduler=FlowMatchEulerDiscreteScheduler(shift=9.0))
    fastvideo_args = type("Args", (), {})()
    fastvideo_args.pipeline_config = PAN2PipelineConfig()
    with pytest.raises(ValueError, match="PDD checkpoints only"):
        stage.forward(ForwardBatch(data_type="video"), fastvideo_args)
    fastvideo_args.pipeline_config = PAN2PipelineConfig(pdd_step_indices=(0, 4, 8))
    with pytest.raises(ValueError, match="without classifier-free guidance"):
        stage.forward(ForwardBatch(data_type="video", guidance_scale=3.0), fastvideo_args)
