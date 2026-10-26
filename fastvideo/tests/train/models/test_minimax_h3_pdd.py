# SPDX-License-Identifier: Apache-2.0
"""Contracts for the PDD-widened MiniMax H3 transformer and training plugin.

A tiny real ``MiniMaxH3Transformer3DModel`` with ``arch_config.pdd_steps`` set
checks the widened heads, the teacher-repeat load hook, the export hooks, and
that materialized heads and the fused block forward agree through the packed
``MiniMaxH3PDDModel`` adapter (FastGen ``test_multimodal_fused_pdd_block_returns_its_mean_velocity``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo import envs
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.layers.pdd import PDDModalitySchedule, PDDReplicatedLinear
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.train.methods.trajectory_matching.pdd_core import advance_block, advance_fused
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3_pdd import MiniMaxH3PDDModel
from fastvideo.train.utils.training_config import DataConfig, TrainingConfig

_GRID = 4
_VIDEO_SHAPE = (1, 2, 24, 4, 4)
_AUDIO_SHAPE = (1, 2, 32, 8)


def _tiny_arch(pdd_steps: int | None) -> MiniMaxH3ArchConfig:
    return MiniMaxH3ArchConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=64,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
        pdd_steps=pdd_steps,
    )


def _tiny_transformer(pdd_steps: int | None, device: torch.device) -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(0)
    config = MiniMaxH3Config(arch_config=_tiny_arch(pdd_steps))
    transformer = MiniMaxH3Transformer3DModel(config=config, hf_config={})
    return transformer.to(device).eval()


def _tiny_training_config():
    return SimpleNamespace(
        data=SimpleNamespace(num_latent_t=2, num_frames=5, num_height=64, num_width=64),
        distributed=SimpleNamespace(sp_size=1),
        vsa_sparsity=0.0,
        vsa_tile_size=256,
    )


def _raw_batch(seed: int = 1, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "vae_latent": torch.randn(1, 24, 2, 4, 4, generator=generator).to(device),
        "audio_latent": torch.randn(1, 2, 32, 8, generator=generator).to(device),
        "text_embedding": torch.randn(1, 4, 5120, generator=generator).to(device),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32).to(device),
    }


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 attention layers require a CUDA device for a real forward")
    return torch.device("cuda")


# ----------------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------------


def test_arch_config_validates_pdd_steps() -> None:
    assert _tiny_arch(None).pdd_steps is None
    assert _tiny_arch(8).pdd_steps == 8
    with pytest.raises(ValueError, match="pdd_steps must be an int >= 2"):
        _tiny_arch(1)


def test_widened_transformer_heads_and_hooks() -> None:
    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=_tiny_arch(_GRID)), hf_config={})
        plain = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=_tiny_arch(None)), hf_config={})

    assert transformer.pdd_steps == _GRID
    assert isinstance(transformer.proj_out, PDDReplicatedLinear)
    assert isinstance(transformer.audio_proj_out, PDDReplicatedLinear)
    assert transformer.proj_out.weight.shape == (_GRID * 24 * 4, 32)
    assert transformer.audio_proj_out.weight.shape == (_GRID * 32, 32)
    assert set(transformer.pdd_linears) == {"video", "audio"}
    # Plain transformers are byte-identical to before: no PDD attributes engage.
    assert plain.pdd_steps is None
    assert not isinstance(plain.proj_out, PDDReplicatedLinear)
    assert plain.inference_config_overrides() == {}
    assert plain._fastvideo_shard_cache_key_extra is None
    assert plain.inference_export_expected_shape("proj_out.weight", (96, 32)) == (96, 32)
    with pytest.raises(RuntimeError, match="no PDD heads"):
        _ = plain.pdd_linears

    # Loading a single-head teacher checkpoint repeats it into every head.
    narrow = torch.arange(96 * 32, dtype=torch.float32).reshape(96, 32)
    widened = transformer.preprocess_loaded_weight("proj_out.weight", narrow)
    assert torch.equal(widened, narrow.repeat(_GRID, 1))
    bias = torch.arange(32, dtype=torch.float32)
    assert torch.equal(transformer.preprocess_loaded_weight("audio_proj_out.bias", bias),
                       bias.repeat(_GRID))
    # A grid-wide checkpoint passes through; other parameters are untouched.
    assert transformer.preprocess_loaded_weight("proj_out.weight", widened) is widened
    other = torch.zeros(3)
    assert transformer.preprocess_loaded_weight("proj_in.weight", other) is other
    with pytest.raises(ValueError, match="Cannot load"):
        transformer.preprocess_loaded_weight("proj_out.weight", torch.zeros(5, 32))

    # Export contract: widened head shapes and the config key that reloads them.
    assert transformer.inference_export_expected_shape("proj_out.weight", (96, 32)) == (_GRID * 96, 32)
    assert transformer.inference_export_expected_shape("audio_proj_out.bias", (32, )) == (_GRID * 32, )
    # Continuations use the previous PDD export as ``init_from``. Its base
    # shapes are already widened and must not receive the grid factor twice.
    assert transformer.inference_export_expected_shape("proj_out.weight", (_GRID * 96, 32)) == (_GRID * 96, 32)
    assert transformer.inference_export_expected_shape("audio_proj_out.bias", (_GRID * 32, )) == (_GRID * 32, )
    with pytest.raises(ValueError, match="single-head rows or"):
        transformer.inference_export_expected_shape("audio_proj_out.bias", (2 * 32, ))
    assert transformer.inference_export_expected_shape("proj_in.weight", (32, 96)) == (32, 96)
    assert transformer.inference_config_overrides() == {"pdd_steps": _GRID}
    assert transformer._fastvideo_shard_cache_key_extra == {"pdd_steps": _GRID}


def test_fastgen_fake_score_adaln_freeze_is_exact() -> None:
    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel(
            config=MiniMaxH3Config(arch_config=_tiny_arch(None)),
            hf_config={},
        )

    adaln_names = {name for name, _ in transformer.named_parameters() if "adaln" in name.lower()}
    assert adaln_names
    before = {name: parameter.requires_grad for name, parameter in transformer.named_parameters()}
    frozen = MiniMaxH3Model._freeze_adaln_parameters(transformer)
    assert frozen == len(adaln_names)
    for name, parameter in transformer.named_parameters():
        expected = False if name in adaln_names else before[name]
        assert parameter.requires_grad is expected


# ----------------------------------------------------------------------------
# Training plugin
# ----------------------------------------------------------------------------


def _make_model(monkeypatch: pytest.MonkeyPatch, transformer: torch.nn.Module,
                device: torch.device) -> MiniMaxH3PDDModel:
    monkeypatch.setattr(MiniMaxH3Model, "device", property(lambda _self: device))
    model = MiniMaxH3PDDModel.__new__(MiniMaxH3PDDModel)
    model._trainable = True
    model.transformer = transformer
    model.training_config = _tiny_training_config()
    model.sp_group = None
    model.attention_backend = None
    model.pdd_steps = _GRID
    return model


def test_pdd_plugin_schedule_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=_tiny_arch(_GRID)), hf_config={})
    model = _make_model(monkeypatch, transformer, torch.device("cpu"))
    grid = model.pdd_fine_grid(torch.device("cpu"))
    assert grid.dtype == torch.float64 and grid.shape == (_GRID + 1, )
    assert float(grid[0]) == 0.999 and float(grid[-1]) == 0.0
    timestep = model.pdd_timestep(grid[:1])
    assert timestep.dtype == torch.float64
    assert float(timestep) == 999.0
    assert model.pdd_timestep(grid[2:3].expand(2)).tolist() == [float(grid[2]) * 1000.0] * 2
    with pytest.raises(ValueError, match="share a base time"):
        model.pdd_timestep(grid[:2])

    batch = SimpleNamespace(minimax_h3_dmd_layout=None)
    with pytest.raises(RuntimeError, match="must set TrainingBatch.minimax_h3_dmd_layout"):
        model.pdd_modality_schedules(batch)
    batch.minimax_h3_dmd_layout = model._fixed_latent_layout()
    blocks = model.pdd_modality_schedules(batch)
    assert [block.name for block in blocks] == ["video", "audio"]
    assert blocks[0].columns == slice(0, 1 * 2 * 24 * 4 * 4)
    assert blocks[1].columns == slice(1 * 2 * 24 * 4 * 4, 1 * 2 * 24 * 4 * 4 + 2 * 32 * 8)
    assert blocks[0].schedule == PDDModalitySchedule(shift=12.0)
    assert blocks[1].schedule == PDDModalitySchedule(shift=3.0)
    # The plugin's PDD sigmas are exactly the DMD adapter's continuous ones.
    sigma_video, sigma_audio = model._noise_amounts(model.pdd_timestep(grid[2:3]))
    torch.testing.assert_close(blocks[0].schedule.sigma(grid[2:3]), sigma_video, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(blocks[1].schedule.sigma(grid[2:3]), sigma_audio, rtol=0.0, atol=1e-12)


def _init_parameters(transformer: torch.nn.Module) -> None:
    """Give a freshly constructed (uninitialized-storage) DiT deterministic values."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, parameter in transformer.named_parameters():
            if parameter.ndim >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
            elif "norm" in name:
                parameter.fill_(1.0)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="MiniMax H3 attention layers require a CUDA device for a real forward")
def test_pdd_plugin_heads_and_fused_forward_agree(monkeypatch: pytest.MonkeyPatch, env_overrides,
                                                  distributed_setup) -> None:
    # H3's attention layers read the sequence-parallel group even at world
    # size one, so the forward runs inside distributed_setup's 1x1 environment.
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    _run_heads_and_fused_forward_check(monkeypatch, _device())


def _run_heads_and_fused_forward_check(monkeypatch: pytest.MonkeyPatch, device: torch.device) -> None:
    transformer = _tiny_transformer(_GRID, device)
    _init_parameters(transformer)
    model = _make_model(monkeypatch, transformer, device)
    batch = model.prepare_batch(_raw_batch(device=device), generator=torch.Generator(device=device).manual_seed(7))
    fine_grid = model.pdd_fine_grid(device)
    blocks = model.pdd_modality_schedules(batch)
    state = batch.latents
    numel = state.shape[1]
    timestep = model.pdd_timestep(fine_grid[:1])

    with torch.no_grad():
        heads = model.predict_pdd_heads(state, timestep, batch, conditional=True, cfg_uncond=None, attn_kind="dense")
    assert heads.shape == (1, _GRID, numel)
    # FastGen casts network outputs to the state dtype: heads live at the bf16 boundary.
    assert heads.dtype == state.dtype == torch.bfloat16
    assert torch.isfinite(heads).all()
    # Widened rows are independent parameters, so the heads genuinely differ.
    assert not torch.allclose(heads[:, 1], heads[:, 0])
    # A head window cuts the same rows before unpatchify.
    with torch.no_grad():
        window = model.predict_pdd_heads(state, timestep, batch, conditional=True, cfg_uncond=None, attn_kind="dense",
                                         head_window=(1, 3))
    torch.testing.assert_close(window, heads[:, 1:3])
    with pytest.raises(ValueError, match="head_window must satisfy"):
        model.predict_pdd_heads(state, timestep, batch, conditional=True, cfg_uncond=None, attn_kind="dense",
                                head_window=(3, 3))

    # The fused forward equals the normalized weighted mean of the
    # materialized heads over the block, per modality.
    with torch.no_grad():
        fused = model.predict_pdd_fused(state, timestep, 1, 3, fine_grid, batch, precision_decoding=torch.float32,
                                        conditional=True, cfg_uncond=None, attn_kind="dense")
    assert fused.shape == (1, numel)
    assert fused.dtype == state.dtype
    expected = torch.zeros(1, numel, device=device, dtype=torch.float32)
    for block in blocks:
        weights = block.schedule.integration_weights(fine_grid)[1:3]
        alpha = (weights / weights.sum()).to(torch.float32)
        expected[:, block.columns] = torch.einsum("n,bnk->bk", alpha, heads[:, 1:3, block.columns].float())
    torch.testing.assert_close(fused.float(), expected.to(state.dtype).float(), rtol=2e-2, atol=2e-2)

    # Materialized and fused block advances land on the same state.
    step_idx = torch.tensor([1], device=device)
    end_idx = torch.tensor([3], device=device)
    with torch.no_grad():
        materialized = advance_block(state, heads, step_idx, end_idx, fine_grid, blocks, precision_decoding=torch.float32)
        fused_state = advance_fused(state, fused, 1, 3, fine_grid, blocks, precision_decoding=torch.float32)
    torch.testing.assert_close(fused_state.float(), materialized.float(), rtol=5e-2, atol=5e-2)
    # Fusion is transient: the widened forward is restored afterwards.
    assert all(linear._fusion_state is None for linear in transformer.pdd_linears.values())


def test_pdd_plugin_loads_widened_heads_for_its_own_role_only(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class _Loaded(torch.nn.Module):
        pdd_steps = _GRID
        patch_size = (1, 2, 2)

    def _load_module_from_path(**kwargs):
        observed["pdd_steps"] = kwargs["training_config"].pipeline_config.dit_config.arch_config.pdd_steps
        return _Loaded()

    monkeypatch.setattr("fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path", _load_module_from_path)
    training_config = TrainingConfig(
        data=DataConfig(preprocessed_data_type="text_only", seed=42),
        pipeline_config=MiniMaxH3PipelineConfig(),
    )

    model = MiniMaxH3PDDModel(init_from="unused", training_config=training_config, pdd_steps=_GRID)

    assert observed["pdd_steps"] == _GRID
    # The shared config keeps the released single-head architecture.
    assert training_config.pipeline_config.dit_config.arch_config.pdd_steps is None
    assert model.training_config is training_config
    assert model._dit_arch_overrides() == {"pdd_steps": _GRID}
    with pytest.raises(ValueError, match="pdd_steps must be an int >= 2"):
        MiniMaxH3PDDModel(init_from="unused", training_config=training_config, pdd_steps=1)

    # An exported checkpoint's config.json can override the grid through
    # update_model_arch; the role refuses a transformer on a different grid.
    class _OtherGrid(torch.nn.Module):
        pdd_steps = 2 * _GRID
        patch_size = (1, 2, 2)

    monkeypatch.setattr("fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path",
                        lambda **kwargs: _OtherGrid())
    with pytest.raises(RuntimeError, match=f"loaded a transformer with pdd_steps={2 * _GRID}"):
        MiniMaxH3PDDModel(init_from="unused", training_config=training_config, pdd_steps=_GRID)
