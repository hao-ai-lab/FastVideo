# SPDX-License-Identifier: Apache-2.0
"""QwenPromptRefinerRole unit tests with a tiny local causal LM."""

from __future__ import annotations

import torch
import pytest

from fastvideo.train.roles.qwen_refiner import QwenPromptRefinerRole


def _build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "<eos>": 2,
    }
    words = (
        ["You", "are", "an", "expert", "prompt", "engineer", "Rewrite", "the", "user", "'s", "to", "be", "more", "detailed", "cinematic", "and", "visually", "grounded", "while", "preserving", "its", "meaning", "Respond", "with", "rewritten", "inside", "<answer>", "</answer>", "tags", "only", ".", "a", "cat", "dog", "runs", "fast", "slow", "cinematic", "lighting", "red", "ball", "jumps", "over", "fence"])
    vocab.update({w: i + 3 for i, w in enumerate(words)})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}")
    return fast


def _build_tiny_model(vocab_size: int):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(config)
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = 2
    return model


def _build_role(**overrides) -> QwenPromptRefinerRole:
    tokenizer = _build_tiny_tokenizer()
    model = _build_tiny_model(len(tokenizer))
    kwargs = dict(
        init_from="tiny-local",
        trainable=True,
        lora={"enable": True, "rank": 4, "alpha": 8},
        model_kind="causal_lm",
        torch_dtype="float32",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    kwargs.update(overrides)
    return QwenPromptRefinerRole(**kwargs)


def test_refiner_role_lora_only_trainable():
    role = _build_role()
    trainable = role.trainable_parameters()
    assert trainable, "expected LoRA parameters to be trainable"
    for name, param in role.model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"non-LoRA parameter trainable: {name}"
    assert set(role.checkpoint_modules()) == {"model"}
    assert role.checkpoint_modules()["model"] is role.model


def test_refiner_requires_lora_config():
    tokenizer = _build_tiny_tokenizer()
    model = _build_tiny_model(len(tokenizer))
    with pytest.raises(ValueError, match="lora"):
        QwenPromptRefinerRole(
            init_from="tiny-local",
            trainable=True,
            lora=None,
            model_kind="causal_lm",
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


def test_sequence_logprobs_and_reference_parity():
    role = _build_role()
    prompts = ["a cat", "a dog runs"]
    completions = ["<answer> cinematic cat </answer>", "<answer> dog </answer>"]

    sums, counts = role.sequence_logprobs(prompts, completions)
    assert sums.shape == (2, )
    assert counts.tolist() == [4, 3]
    assert torch.isfinite(sums).all()

    # Zero-init LoRA B: adapter-disabled reference matches the adapter.
    ref_sums, _ = role.sequence_logprobs(prompts, completions, use_adapter=False)
    assert torch.allclose(sums, ref_sums, atol=1e-5)

    # Perturb LoRA B: current policy diverges; reference still matches base.
    with torch.no_grad():
        for name, param in role.model.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.1)
    new_sums, _ = role.sequence_logprobs(prompts, completions)
    new_ref_sums, _ = role.sequence_logprobs(prompts, completions, use_adapter=False)
    assert not torch.allclose(new_sums, ref_sums, atol=1e-4)
    assert torch.allclose(new_ref_sums, ref_sums, atol=1e-5)


def test_sequence_logprobs_requires_grad():
    role = _build_role()
    sums, _ = role.sequence_logprobs(["a cat"], ["<answer> x </answer>"],
                                     requires_grad=True)
    assert sums.requires_grad
    (-sums.mean()).backward()
    grads = [p.grad for p in role.trainable_parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_generate_refinements_count_and_type():
    role = _build_role()
    outputs = role.generate_refinements(
        ["a cat", "a dog"],
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        seed=1234,
    )
    assert len(outputs) == 2
    assert all(isinstance(text, str) for text in outputs)
