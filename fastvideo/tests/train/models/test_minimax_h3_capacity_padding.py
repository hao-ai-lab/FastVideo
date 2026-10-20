# SPDX-License-Identifier: Apache-2.0
"""Real tiny H3/PDD forwards on CPU; only the single-rank transport is stubbed."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo import envs
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.tests.train.models.test_minimax_h3_pdd import (
    _GRID, _init_parameters, _make_model, _raw_batch, _tiny_transformer,
)
from fastvideo.tests.train.models.h3_reference_fixtures import (
    _raw_batch as _ref_raw_batch, _training_config as _ref_training_config,
)
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3_pdd import MiniMaxH3PDDModel
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_pdd import MiniMaxH3Ref2VPDDModel


@pytest.fixture(autouse=True)
def _dense_attention_backend(env_overrides):
    # Layers built outside a loader bind the environment's backend.
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))


def _cpu_model(monkeypatch, reference):
    import fastvideo.attention.layer as attention_layer

    monkeypatch.setattr(attention_layer, "get_sp_parallel_rank", lambda: 0)
    monkeypatch.setattr(attention_layer, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(attention_layer, "sequence_model_parallel_all_to_all_4D", lambda tensor, **_: tensor)
    transformer = _tiny_transformer(_GRID, torch.device("cpu"))
    _init_parameters(transformer)
    # CPU linear algebra requires matching input/weight dtypes (no CUDA autocast).
    transformer.to(torch.bfloat16)
    model = _make_model(monkeypatch, transformer, torch.device("cpu"))
    model.training_config.model = SimpleNamespace(minimax_h3_activation_offload_min_tokens=0)
    if reference:
        ref = MiniMaxH3Ref2VPDDModel.__new__(MiniMaxH3Ref2VPDDModel)
        ref.__dict__.update(model.__dict__)
        model = ref
        model.training_config = _ref_training_config()
        model.noise_scheduler = MiniMaxH3Scheduler(shift=12.0)
        model.audio_noise_scheduler = MiniMaxH3Scheduler(shift=3.0)
    model.attention_backend = AttentionBackendEnum.TORCH_SDPA
    return model


@pytest.mark.parametrize("reference", [False, True])
def test_pdd_capacity_preserves_heads_losses_gradients_and_fused_forward(monkeypatch, reference):
    model = _cpu_model(monkeypatch, reference)
    raw = _ref_raw_batch() if reference else _raw_batch()
    prepare = lambda: model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    real = prepare()
    model.training_config.data.fixed_sequence_capacities = [256]
    padded = prepare()
    assert padded.minimax_h3_layout.sequence_length == 256
    assert padded.minimax_h3_layout.pad_tokens == 256 - real.minimax_h3_layout.sequence_length
    assert padded.minimax_h3_layout.valid.sum() == real.minimax_h3_layout.sequence_length
    assert torch.equal(padded.latents, real.latents)
    assert model.pdd_modality_schedules(padded) == model.pdd_modality_schedules(real)
    assert padded.attn_metadata.packed_sequence.total_query_tokens == real.minimax_h3_layout.sequence_length
    assert padded.minimax_h3_text_attn_metadata.packed_sequence.pad_tokens == 0
    if reference:
        assert torch.equal(padded.minimax_h3_condition_video_rows, real.minimax_h3_condition_video_rows)
        assert padded.minimax_h3_layout.reference_segments == real.minimax_h3_layout.reference_segments

    grid = model.pdd_fine_grid(torch.device("cpu"))
    timestep = model.pdd_timestep(grid[:1])
    # The repeated block sees one fixed shape even though generated outputs
    # and the text refiner retain their own real-only shapes.
    seen = []
    handle = model.transformer.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].shape[1]))
    heads = model.predict_pdd_heads(real.latents, timestep, real, attn_kind="dense")
    expected_loss = heads.float().square().mean()
    expected_loss.backward()
    expected_grads = {name: p.grad.clone() for name, p in model.transformer.named_parameters() if p.grad is not None}
    model.transformer.zero_grad(set_to_none=True)
    # DMD/PDD callers request vsa even for a dense role; the padded fallback
    # must retain dense masking instead of accidentally using no metadata.
    actual = model.predict_pdd_heads(padded.latents, timestep, padded, attn_kind="vsa")
    actual_loss = actual.float().square().mean()
    actual_loss.backward()
    handle.remove()
    assert seen == [real.minimax_h3_layout.sequence_length, 256]
    torch.testing.assert_close(actual, heads, rtol=0, atol=0)
    assert torch.equal(actual_loss, expected_loss)
    for name, p in model.transformer.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(p.grad, expected_grads[name], rtol=0.02, atol=2e-4)
    with torch.no_grad():
        fused = model.predict_pdd_fused(real.latents, timestep, 0, 2, grid, real,
                                       precision_decoding=torch.float32, attn_kind="dense")
        padded_fused = model.predict_pdd_fused(padded.latents, timestep, 0, 2, grid, padded,
                                              precision_decoding=torch.float32, attn_kind="dense")
    torch.testing.assert_close(padded_fused, fused, rtol=0, atol=0)


@pytest.mark.parametrize("reference", [False, True])
def test_exact_capacity_uses_the_same_dense_capacity_route(monkeypatch, reference):
    model = _cpu_model(monkeypatch, reference)
    raw = _ref_raw_batch() if reference else _raw_batch()
    real = model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    non_text = real.minimax_h3_layout.sequence_length - real.encoder_hidden_states.shape[1]
    text_length = 256 - non_text
    raw["text_embedding"] = torch.zeros(1, text_length, 5120)
    raw["text_attention_mask"] = torch.ones(1, text_length)
    if reference:
        raw["text_token_tags"] = torch.ones(1, text_length)
    model.training_config.data.fixed_sequence_capacities = [256]
    exact = model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    assert exact.minimax_h3_layout.pad_tokens == 0
    assert exact.minimax_h3_layout.valid.all()
    assert exact.attn_metadata.packed_sequence.sequence_capacity == 256
    assert exact.attn_metadata.packed_sequence.query_lengths == (256,)
    assert exact.sequence_padding_ratio == 1.0


def test_padded_transformer_fails_closed_without_attention_metadata(monkeypatch):
    model = _cpu_model(monkeypatch, False)
    model.training_config.data.fixed_sequence_capacities = [256]
    batch = model.prepare_batch(_raw_batch(), generator=torch.Generator().manual_seed(7))
    batch.attn_metadata = None
    grid = model.pdd_fine_grid(torch.device("cpu"))
    with pytest.raises(ValueError, match="matching masked attention metadata"):
        model.predict_pdd_heads(batch.latents, model.pdd_timestep(grid[:1]), batch, attn_kind="dense")


@pytest.mark.parametrize("model_type", [MiniMaxH3PDDModel, MiniMaxH3Ref2VPDDModel])
def test_pdd_rejects_cross_sample_envelopes_but_not_capacity(model_type):
    model = model_type.__new__(model_type)
    assert model._supports_cross_sample_sequence_packing is False
    with pytest.raises(ValueError, match="one document"):
        MiniMaxH3Model.prepare_batch(model, {"_packed_samples": [{}, {}]}, generator=torch.Generator())


@pytest.mark.parametrize("tile_size", [64, 128, 256])
def test_ref2va_capacity_applies_after_all_reference_regions(monkeypatch, tile_size):
    model = _cpu_model(monkeypatch, True)
    raw = _ref_raw_batch()
    real = model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    real_count = real.minimax_h3_layout.sequence_length
    model.training_config.data.fixed_sequence_capacities = [256]
    raw.update(_fixed_sequence_capacity=256, _packed_joint_token_count=real_count)
    padded = model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
    assert padded.sequence_packing_num_tokens == real_count
    assert padded.sequence_padding_ratio == 256 / real_count
    model.attention_backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    model.training_config.vsa_sparsity = 0.9
    model.training_config.vsa_tile_size = tile_size
    model._maybe_build_vsa_metadata(padded)
    assert padded.attn_metadata_vsa.pad_tokens == padded.minimax_h3_layout.pad_tokens
    assert padded.attn_metadata_vsa.total_seq_length == real_count
    raw["_packed_joint_token_count"] = real_count - 1
    with pytest.raises(ValueError, match="accounting disagrees"):
        model.prepare_batch(raw, generator=torch.Generator().manual_seed(7))
