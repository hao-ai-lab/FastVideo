# SPDX-License-Identifier: Apache-2.0
"""Target-only H3 DMD with real tiny CPU transformer forwards and backward."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import MiniMaxH3DMDModel


def _model(monkeypatch):
    import fastvideo.attention.layer as attention_layer
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setattr(attention_layer, "get_sp_parallel_rank", lambda: 0)
    monkeypatch.setattr(attention_layer, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(attention_layer, "sequence_model_parallel_all_to_all_4D", lambda tensor, **_: tensor)
    monkeypatch.setattr(MiniMaxH3Model, "device", property(lambda _: torch.device("cpu")))
    arch = MiniMaxH3ArchConfig(num_attention_heads=2, attention_head_dim=16, hidden_size=32,
                              num_layers=1, num_refiner_layers=1, ffn_dim=64, freq_dim=16,
                              time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)
    transformer = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=arch), hf_config={})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in transformer.named_parameters():
            if parameter.ndim == 1 and "norm" in name:
                parameter.fill_(1.0)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    transformer.to(torch.bfloat16)
    model = MiniMaxH3DMDModel.__new__(MiniMaxH3DMDModel)
    model._trainable = True
    model.transformer = transformer
    model.training_config = SimpleNamespace(
        data=SimpleNamespace(num_latent_t=2, num_frames=5, num_height=64, num_width=64,
                             native_shape_bucketing=False),
        distributed=SimpleNamespace(sp_size=1), vsa_sparsity=0.8, vsa_tile_size=64,
        model=SimpleNamespace(minimax_h3_activation_offload_min_tokens=0))
    model.noise_scheduler = MiniMaxH3Scheduler(shift=10.0)
    model.audio_noise_scheduler = MiniMaxH3Scheduler(shift=3.0)
    model.attention_backend = AttentionBackendEnum.TORCH_SDPA
    model.sp_group = None
    return model


def _raw():
    generator = torch.Generator().manual_seed(3)
    return {"vae_latent": torch.randn(1, 24, 2, 4, 4, generator=generator),
            "audio_latent": torch.randn(1, 2, 32, 8, generator=generator),
            "text_embedding": torch.randn(1, 4, 5120, generator=generator),
            "text_attention_mask": torch.tensor([[1, 1, 0, 0]])}


@pytest.mark.parametrize("conditional", [False, True])
def test_capacity_preserves_real_dmd_outputs_losses_and_gradients(monkeypatch, conditional):
    model = _model(monkeypatch)
    raw = _raw()
    prepare = lambda: model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    real = prepare()
    model.training_config.data.fixed_sequence_capacities = [256]
    padded = prepare()
    assert torch.equal(real.latents, padded.latents)
    assert model.modality_slices_for_batch(real) == model.modality_slices_for_batch(padded)
    assert padded.minimax_h3_text_attn_metadata.packed_sequence.query_lengths == (2,)
    assert padded.attn_metadata.packed_sequence.query_lengths == (real.minimax_h3_layout.sequence_length,)
    assert padded.minimax_h3_layout.sequence_length == 256
    timestep = torch.tensor([500], dtype=torch.long)
    kwargs = dict(conditional=conditional, cfg_uncond={"text": "zero"}, attn_kind="vsa")
    seen = []
    handle = model.transformer.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].shape[1]))
    expected = model.predict_noise(real.latents, timestep, real, **kwargs)
    expected_loss = expected.float().square().mean()
    expected_loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.transformer.named_parameters() if p.grad is not None}
    model.transformer.zero_grad(set_to_none=True)
    actual = model.predict_noise(padded.latents, timestep, padded, **kwargs)
    actual_loss = actual.float().square().mean()
    actual_loss.backward()
    handle.remove()
    assert seen == [real.minimax_h3_layout.sequence_length, 256]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_loss, expected_loss)
    for name, parameter in model.transformer.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0.02, atol=2e-4)


def test_release_clocks_and_native_geometry_do_not_use_fixed_config(monkeypatch):
    model = _model(monkeypatch)
    model.training_config.data.native_shape_bucketing = True
    raw = _raw()
    raw.pop("vae_latent")
    raw.pop("audio_latent")
    raw["_shape_bucket_id"] = "bucket=128x64-5f"
    batch = model.prepare_batch(raw, generator=torch.Generator().manual_seed(4), latents_source="zeros")
    layout = batch.minimax_h3_dmd_layout
    assert layout.video_shape == (1, 2, 24, 4, 8)
    noise = torch.randn(batch.latents.shape, generator=torch.Generator().manual_seed(8))
    for timestep, maximum in [(torch.tensor([500]), 1.0), (torch.tensor([500.0]), 0.999)]:
        video_sigma, audio_sigma = model._noise_amounts(timestep)
        base = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(video_sigma, 10 * base * maximum / (base * 9 + maximum), rtol=0, atol=0)
        torch.testing.assert_close(audio_sigma, 3 * base * maximum / (base * 2 + maximum), rtol=0, atol=0)
        mixed = model.add_noise_for_batch(batch.latents.float(), noise, timestep, batch)
        recovered = model.extract_eps_for_batch(mixed, batch.latents.float(), timestep, batch)
        torch.testing.assert_close(recovered, noise, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("tile_size", [64, 128])
def test_capacity_does_not_change_target_vsa_geometry(monkeypatch, tile_size):
    model = _model(monkeypatch)
    model.attention_backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    model.training_config.vsa_tile_size = tile_size
    raw = _raw()
    real = model.prepare_batch(raw, generator=torch.Generator().manual_seed(1))
    model.training_config.data.fixed_sequence_capacities = [256]
    padded = model.prepare_batch(raw, generator=torch.Generator().manual_seed(1))
    for field in ("variable_block_sizes", "untile_combined_index", "block_mask_plan"):
        assert torch.equal(getattr(real.attn_metadata_vsa, field), getattr(padded.attn_metadata_vsa, field))
    assert padded.attn_metadata_vsa.total_seq_length == real.minimax_h3_layout.sequence_length
    assert padded.attn_metadata_vsa.pad_tokens == 256 - real.minimax_h3_layout.sequence_length
    raw["_packed_joint_token_count"] = real.minimax_h3_layout.sequence_length - 1
    with pytest.raises(ValueError, match="accounting disagrees"):
        model.prepare_batch(raw, generator=torch.Generator().manual_seed(1))


def test_target_adapter_rejects_cross_document_envelopes(monkeypatch):
    model = _model(monkeypatch)
    with pytest.raises(ValueError, match="one document"):
        model.prepare_batch({"_packed_samples": [_raw(), _raw()]}, generator=torch.Generator())


def test_native_text_only_preprocessor_keeps_full_text_with_safety_bound(monkeypatch):
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_text_only
    import fastvideo.train.models.minimax_h3.minimax_h3 as adapter
    import fastvideo.train.utils.dataloader as dataloader
    calls = []
    monkeypatch.setattr(adapter, "get_sp_group", lambda: None)
    monkeypatch.setattr(dataloader, "build_parquet_t2v_train_dataloader",
                        lambda data, **kwargs: calls.append((data, kwargs)) or "loader")
    model = MiniMaxH3DMDModel.__new__(MiniMaxH3DMDModel)
    data = SimpleNamespace(native_shape_bucketing=True, preprocessed_data_type="text_only",
                           max_presentation_tokens=65536)
    model.init_preprocessors(SimpleNamespace(data=data))
    assert model.dataloader == "loader"
    assert calls == [(data, {"text_len": None, "parquet_schema": pyarrow_schema_text_only,
                             "max_presentation_tokens": 65536})]


@pytest.mark.parametrize("trainable", [False, True])
def test_trainable_refiner_does_not_reuse_pre_update_text(monkeypatch, trainable):
    model = _model(monkeypatch)
    transformer = model.transformer
    transformer._fastvideo_training_role = trainable
    text = _raw()["text_embedding"].to(torch.bfloat16)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        before = transformer._refined_text(text)
        again = transformer._refined_text(text)
        assert (before is again) is not trainable
        # Same text storage after an optimizer update must use the new weights.
        transformer.context_embedder.weight[0].add_(0.2)
        after = transformer._refined_text(text)
    if trainable:
        assert not torch.equal(after, before)
        assert transformer._text_cache is None
    else:
        # Frozen roles and inference retain the established inference cache.
        assert after is before
