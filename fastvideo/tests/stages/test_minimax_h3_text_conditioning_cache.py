# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_conditioning as conditioning
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.pipelines.lazy_module import LazyModule


class TinyConditioner(MiniMaxH3Qwen3VLConditioner):

    def __init__(self, offset=0.0):
        torch.nn.Module.__init__(self)
        self._conditioning_revision = 0
        self.weight = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(8, 4) + offset)
        self.calls = 0
        self.eval()

    def _is_omitted_checkpoint_key(self, name):
        return False

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return torch.nn.functional.embedding(input_ids, self.weight)


@pytest.fixture
def make_stage(monkeypatch):
    monkeypatch.setattr(conditioning, "get_world_size", lambda: 1)
    monkeypatch.setattr(conditioning, "get_local_torch_device", lambda: torch.device("cpu"))

    def create(budget=256, owner=None):
        tokenizer = lambda prompt, **kwargs: {"input_ids": [ord(char) % 8 for char in prompt]}
        tokenizer.convert_tokens_to_ids = lambda token: 3
        return conditioning.MiniMaxH3ConditioningStage(
            TinyConditioner() if owner is None else owner,
            tokenizer,
            None,
            manage_cpu_offload=False,
            text_conditioning_cache_max_bytes=budget,
        )

    return create


def run(stage, prompt="ab", seed=0):
    batch = SimpleNamespace(prompt=prompt, seed=seed, references=None, extra={MINIMAX_H3_KEYFRAMES_KEY: []})
    return stage.forward(batch, SimpleNamespace(text_encoder_cpu_offload=False))


def test_default_recomputes_and_enabled_reuses_across_seeds(make_stage):
    disabled = make_stage(0)
    run(disabled, seed=1)
    run(disabled, seed=2)
    assert disabled.conditioner.calls == 2 and not disabled._text_cache

    stage = make_stage()
    first = run(stage, seed=1)
    expected = first.prompt_embeds[0].clone()
    first.prompt_embeds[0].add_(100)
    second = run(stage, seed=2)
    torch.testing.assert_close(second.prompt_embeds[0], expected, rtol=0, atol=0)
    second.prompt_embeds[0].zero_()
    torch.testing.assert_close(run(stage).prompt_embeds[0], expected, rtol=0, atol=0)
    assert stage.conditioner.calls == 1
    assert stage._text_cache_bytes == 48
    assert all(value.device.type == "cpu" for entry in stage._text_cache.values() for value in entry)


@pytest.mark.parametrize("entrypoint,fail", [("load_weights", False), ("load_weights", True),
                                           ("load_state_dict", False), ("load_state_dict", True)])
def test_native_root_reload_invalidates_before_partial_failure(make_stage, entrypoint, fail):
    stage = make_stage()
    old = run(stage).prompt_embeds[0]
    owner = stage.conditioner
    changed = owner.weight.detach().clone() + 1
    before = owner.conditioning_revision
    if entrypoint == "load_weights":
        weights = [("weight", changed)]
        if fail:
            weights.append(("unexpected", torch.zeros(1)))
        if fail:
            with pytest.raises(ValueError, match="Unexpected"):
                owner.load_weights(weights)
        else:
            owner.load_weights(weights)
    else:
        state = {"weight": changed}
        if fail:
            state["unexpected"] = torch.zeros(1)
            with pytest.raises(RuntimeError, match="Unexpected"):
                owner.load_state_dict(state)
        else:
            owner.load_state_dict(state)
    assert owner.conditioning_revision == before + 1
    torch.testing.assert_close(run(stage).prompt_embeds[0], old + 1, rtol=0, atol=0)
    assert owner.calls == 2


def test_tokens_dtype_and_replaced_encoder_invalidate(make_stage):
    stage = make_stage()
    original = run(stage).prompt_embeds[0]
    stage.tokenizer = lambda prompt, **kwargs: {"input_ids": [2, 3]}
    run(stage)
    assert stage.conditioner.calls == 2
    stage.conditioner.double()
    assert run(stage).prompt_embeds[0].dtype == torch.float64
    assert stage.conditioner.calls == 3
    previous = weakref.ref(stage.conditioner)
    stage.conditioner = TinyConditioner(10.0)
    gc.collect()
    assert previous() is None
    output = run(stage).prompt_embeds[0]
    torch.testing.assert_close(output, stage.conditioner.weight[[2, 3]].unsqueeze(0), rtol=0, atol=0)
    assert not torch.equal(original, output) and stage.conditioner.calls == 1


def test_lazy_release_never_reuses_an_old_encoder_result(make_stage):
    owners = []

    def load():
        owner = TinyConditioner(10.0 * len(owners))
        owners.append(weakref.ref(owner))
        return owner

    proxy = LazyModule("text_encoder", load)
    stage = make_stage(owner=proxy)
    first = run(stage).prompt_embeds[0]
    assert proxy.release()
    assert owners[0]() is None
    torch.testing.assert_close(run(stage).prompt_embeds[0], first + 10, rtol=0, atol=0)
    assert proxy.materialize().calls == 1


def test_byte_budget_lru_and_oversized_result(make_stage):
    stage = make_stage(96)
    run(stage, "ab")
    run(stage, "cd")
    run(stage, "ab")
    run(stage, "ef")
    run(stage, "cd")
    assert stage.conditioner.calls == 4 and len(stage._text_cache) == 2
    assert stage._text_cache_bytes == 96
    oversized = make_stage(47)
    run(oversized)
    run(oversized)
    assert oversized.conditioner.calls == 2 and oversized._text_cache_bytes == 0


def test_training_bypasses_and_clears_reuse(make_stage):
    stage = make_stage()
    run(stage)
    stage.conditioner.train()
    run(stage)
    run(stage)
    assert stage.conditioner.calls == 3 and not stage._text_cache


def test_autocast_context_does_not_reuse_other_compute_settings(make_stage):
    stage = make_stage()
    run(stage)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        run(stage)
        run(stage)
    assert stage.conditioner.calls == 2
    run(stage)
    assert stage.conditioner.calls == 3


def test_keyframe_and_reference_inputs_bypass_text_reuse(make_stage, monkeypatch):
    stage = make_stage()
    original = stage._encode_tokens
    calls = []
    monkeypatch.setattr(stage, "_encode_tokens", lambda *args, **kwargs: calls.append(kwargs) or original(*args, **kwargs))
    def image_processor(images, **kwargs):
        return {"pixel_values": torch.zeros(len(images), 3),
                "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long)}

    image_processor.merge_size = 1
    stage.processor = SimpleNamespace(image_processor=image_processor)
    for _ in range(2):
        batch = SimpleNamespace(prompt="ab", references=None, extra={MINIMAX_H3_KEYFRAMES_KEY: [object()]})
        stage.forward(batch, SimpleNamespace(text_encoder_cpu_offload=False))
    for _ in range(2):
        batch = SimpleNamespace(prompt="ab", references=[object()], extra={MINIMAX_H3_KEYFRAMES_KEY: []})
        stage.forward(batch, SimpleNamespace(text_encoder_cpu_offload=False))
    assert len(calls) == 4 and not stage._text_cache


def test_non_single_rank_and_invalid_budgets_are_rejected(make_stage, monkeypatch):
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            make_stage(value)
    monkeypatch.setattr(conditioning, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="single-rank"):
        make_stage()
    monkeypatch.setattr(conditioning, "get_world_size", lambda: 1)
    stage = make_stage()
    monkeypatch.setattr(conditioning, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="single-rank"):
        run(stage)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_native_qwen_conditioner_cuda_reuse_and_reload(distributed_setup):
    from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
        MiniMaxH3Qwen3VLArchConfig, MiniMaxH3Qwen3VLConfig)
    from fastvideo.distributed import get_local_torch_device

    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config = MiniMaxH3Qwen3VLArchConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_hidden_layers_override=2, output_hidden_state_index=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 3, 3], "rope_type": "default"},
        vision_depth=1, vision_hidden_size=32, vision_intermediate_size=64, vision_num_heads=2,
        vision_patch_size=2, vision_temporal_patch_size=1, vision_spatial_merge_size=1,
        vision_out_hidden_size=32, vision_num_position_embeddings=4, vision_deepstack_visual_indexes=(),
    )
    device = get_local_torch_device()
    with torch.device(device):
        owner = MiniMaxH3Qwen3VLConditioner(config).float().eval()
    generator = torch.Generator(device="cpu").manual_seed(20261027)
    weights = [(name, torch.randn(parameter.shape, generator=generator) * 0.02)
               for name, parameter in owner.named_parameters()]
    owner.load_weights(weights)
    tokenizer = lambda prompt, **kwargs: {"input_ids": [1, 2, 3, 4]}
    stage = conditioning.MiniMaxH3ConditioningStage(
        owner, tokenizer, None, manage_cpu_offload=False, text_conditioning_cache_max_bytes=4096)
    fresh = conditioning.MiniMaxH3ConditioningStage(owner, tokenizer, None, manage_cpu_offload=False)
    calls = []
    handle = owner.register_forward_hook(lambda *args: calls.append(True))
    try:
        cold = run(stage, seed=1).prompt_embeds[0]
        warm = run(stage, seed=2).prompt_embeds[0]
        assert cold.device.type == "cuda" and torch.isfinite(cold).all()
        assert len(calls) == 1
        torch.testing.assert_close(cold, warm, rtol=0, atol=0)
        torch.testing.assert_close(run(fresh).prompt_embeds[0], warm, rtol=0, atol=0)
        assert len(calls) == 2
        owner.load_weights([(name, value + 0.01) for name, value in weights])
        reloaded = run(stage).prompt_embeds[0]
        assert len(calls) == 3
        torch.testing.assert_close(run(fresh).prompt_embeds[0], reloaded, rtol=0, atol=0)
        assert len(calls) == 4
        assert 0 < stage._text_cache_bytes <= 4096
        assert all(value.device.type == "cpu" for entry in stage._text_cache.values() for value in entry)
    finally:
        handle.remove()
