# SPDX-License-Identifier: Apache-2.0
"""Loading a PDD-widened MiniMax-H3 ``transformer_ref`` export through the transformer loader.

A PDD export records ``pdd_steps`` in its transformer ``config.json`` and
stores both output projections widened to that many heads, next to its trained
VSA compression gates. The loader must build the widened heads from the
config, load every tensor exactly, and fail closed when the config and the
weights disagree. Runs on CPU with a tiny synthetic checkpoint.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import fastvideo.envs as envs
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.pdd import PDDModalitySchedule, PDDReplicatedLinear, pdd_fine_grid
from fastvideo.models.dits import minimax_h3
from fastvideo.models.loader import component_loader
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG, build_packed_sequence, build_row_timesteps
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.platforms import AttentionBackendEnum

PDD_STEPS = 4
TINY_ARCH = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    hidden_size=32,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=48,
    in_channels=4,
    audio_in_channels=4,
    patch_size=[1, 2, 2],
    text_dim=16,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=24,
    rope_freq_dim=2,
    rope_theta=10000.0,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)
VIDEO_PATCH_DIM = 4 * 1 * 2 * 2


def _released_name(name: str) -> str:
    """FastVideo parameter name -> released (diffusers) checkpoint name."""
    for fastvideo_name, released_name in (("time_embedder.fc_in.", "time_embedder.linear_1."),
                                          ("time_embedder.fc_out.", "time_embedder.linear_2."),
                                          (".attn.to_out.", ".attn.to_out.0."), (".ff.fc_in.", ".ff.net.0.proj."),
                                          (".ff.fc_out.", ".ff.net.2.")):
        name = name.replace(fastvideo_name, released_name)
    return name


class _VSAH3Selection:
    """What the selector returns when VIDEO_SPARSE_ATTN_H3 resolves on a CUDA host."""

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN_H3"


@pytest.fixture()
def cpu_loader(monkeypatch, env_overrides):
    """Load onto the CPU with SDPA attention, on CPU-only and CUDA hosts alike."""
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))


@pytest.fixture()
def vsa_gates(monkeypatch, cpu_loader):
    """Construct H3 attention as the VSA-H3 backend does, so the compression gates exist on a CPU host."""
    select = minimax_h3.get_attn_backend

    def select_vsa_h3(head_size, dtype, supported_attention_backends=None, **kwargs):
        if supported_attention_backends and AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3 in supported_attention_backends:
            return _VSAH3Selection
        return select(head_size, dtype, supported_attention_backends=supported_attention_backends, **kwargs)

    monkeypatch.setattr(minimax_h3, "get_attn_backend", select_vsa_h3)


@pytest.fixture()
def single_process_group(env_overrides):
    """A 1x1 sequence/tensor-parallel group, which DistributedAttention needs even on one CPU process."""
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    # Keep a launcher- or CI-assigned rendezvous: packed CI lanes share the tray's network namespace, and each
    # lease gets its own port range.
    env_overrides.enter_context(envs.override_external("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", os.environ.get("MASTER_PORT") or "29591"))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


def _write_transformer_ref(root, *, config_pdd_steps, weight_pdd_steps=PDD_STEPS, with_gates=True):
    """A synthetic BF16 export: every tensor the (widened) model defines, in the released naming."""
    arch = MiniMaxH3ArchConfig(**{**TINY_ARCH, "patch_size": tuple(TINY_ARCH["patch_size"])},
                               pdd_steps=weight_pdd_steps)
    with torch.device("meta"):
        reference = minimax_h3.MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch), hf_config={})
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for name, parameter in reference.named_parameters():
        if with_gates or "to_gate_compress" not in name:
            tensors[_released_name(name)] = (torch.randn(parameter.shape, generator=generator) * 0.1).to(torch.bfloat16)
    path = root / "transformer_ref"
    path.mkdir()
    save_file(tensors, str(path / "diffusion_pytorch_model.safetensors"))
    config = {"_class_name": "MiniMaxH3Transformer3DModel", "_diffusers_version": "0.36.0.dev0", **TINY_ARCH}
    if config_pdd_steps is not None:
        config["pdd_steps"] = config_pdd_steps
    (path / "config.json").write_text(json.dumps(config))
    return path, tensors


def _load(path):
    args = SimpleNamespace(
        pipeline_config=MiniMaxH3PipelineConfig(),
        override_transformer_cls_name=None,
        model_paths={},
        init_weights_from_safetensors=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        dit_cpu_offload=True,
        pin_cpu_memory=False,
        use_fsdp_inference=False,
        training_mode=False,
        enable_torch_compile=False,
        torch_compile_kwargs=None,
        inference_torch_compile=False,
        VSA_tile_size=128,
        lora_path=None,
        lora_strength=1.0,
        inference_mode=True,
        dit_layerwise_offload=False,
    )
    return component_loader.TransformerLoader().load(str(path), args)


def test_load_widened_heads_and_trained_gates_exactly(tmp_path, vsa_gates):
    path, tensors = _write_transformer_ref(tmp_path, config_pdd_steps=PDD_STEPS)
    model = _load(path)

    assert model.pdd_steps == PDD_STEPS
    for head, rows in (("proj_out", VIDEO_PATCH_DIM), ("audio_proj_out", TINY_ARCH["audio_in_channels"])):
        linear = getattr(model, head)
        assert isinstance(linear, PDDReplicatedLinear)
        assert linear.grid_size == PDD_STEPS and linear.head_output_size == rows
        assert linear.weight.shape == (PDD_STEPS * rows, TINY_ARCH["hidden_size"])
        # Output projections stay FP32 at inference, like the released heads.
        assert linear.weight.dtype == torch.float32
    gate_names = [name for name, _ in model.named_parameters() if "to_gate_compress" in name]
    assert gate_names == [f"transformer_blocks.{index}.attn.to_gate_compress.weight" for index in range(2)]

    loaded = dict(model.named_parameters())
    assert {_released_name(name) for name in loaded} == set(tensors)
    for name, parameter in loaded.items():
        assert torch.equal(parameter, tensors[_released_name(name)].to(parameter.dtype)), name
    with torch.no_grad():
        assert all(block.attn._gate_active() for block in model.transformer_blocks)


def test_load_export_without_gates_disables_gate_branch(tmp_path, vsa_gates):
    path, _ = _write_transformer_ref(tmp_path, config_pdd_steps=PDD_STEPS, with_gates=False)
    model = _load(path)
    with torch.no_grad():
        assert not any(block.attn._gate_active() for block in model.transformer_blocks)


@pytest.mark.parametrize("config_pdd_steps,weight_pdd_steps", [(None, PDD_STEPS), (PDD_STEPS, None), (8, PDD_STEPS)])
def test_load_mismatched_config_and_weight_grids(tmp_path, vsa_gates, config_pdd_steps, weight_pdd_steps):
    path, _ = _write_transformer_ref(tmp_path, config_pdd_steps=config_pdd_steps, weight_pdd_steps=weight_pdd_steps)
    # ReplicatedLinear.weight_loader rejects the shape; never a silent partial load.
    with pytest.raises((AssertionError, RuntimeError), match="size"):
        _load(path)


def test_fuse_pdd_block_weighted_mean_of_widened_heads(tmp_path, cpu_loader, single_process_group):
    """End to end through the DiT: a fused block equals the weighted mean of its heads' outputs."""
    path, _ = _write_transformer_ref(tmp_path, config_pdd_steps=PDD_STEPS, with_gates=False)
    model = _load(path)
    layout = build_packed_sequence(torch.full((3, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long), 2, 4, 4, 2, (1, 2, 2))
    unique, inverse = build_row_timesteps(layout, 0.3, 0.4, 0.999, 1.0)
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, int(layout.video_indices.numel()), VIDEO_PATCH_DIM, generator=generator),
        audio_hidden_states=torch.randn(1,
                                        int(layout.audio_indices.numel()),
                                        TINY_ARCH["audio_in_channels"],
                                        generator=generator),
        encoder_hidden_states=torch.randn(1, 3, TINY_ARCH["text_dim"], generator=generator),
        timestep=unique,
        timestep_indices=inverse,
        token_tags=layout.token_tags,
        position_ids=layout.position_ids,
        video_indices=layout.video_indices,
        audio_indices=layout.audio_indices,
        text_indices=layout.text_indices,
    )
    weights = {
        "video": PDDModalitySchedule(shift=12.0).integration_weights(pdd_fine_grid(PDD_STEPS)),
        "audio": PDDModalitySchedule(shift=3.0).integration_weights(pdd_fine_grid(PDD_STEPS)),
    }
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                              forward_batch=ForwardBatch(data_type="dummy")):
        heads = model(**inputs)
        with model.fuse_pdd_block(1, 3, weights, torch.float32):
            fused = model(**inputs)
    for name, all_heads, block in zip(("video", "audio"), heads, fused, strict=True):
        all_heads = all_heads.unflatten(-1, (PDD_STEPS, -1))
        alpha = weights[name][1:3] / weights[name][1:3].sum()
        expected = torch.einsum("n,brnc->brc", alpha.float(), all_heads[..., 1:3, :].float())
        assert block.shape == expected.shape
        torch.testing.assert_close(block.float(), expected, rtol=1e-4, atol=1e-5)
