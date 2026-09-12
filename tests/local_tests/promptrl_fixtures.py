# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for PromptRL tests: tiny refiner + fake Wan student."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.methods.rl.promptrl.rewards import RewardResult
from fastvideo.train.methods.rl.promptrl.sde import (
    SDETransition,
    sde_step_from_model_output,
    transition_log_prob,
)
from fastvideo.train.roles.base import TrainRoleBase
from fastvideo.train.roles.qwen_refiner import QwenPromptRefinerRole


def build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[UNK]": 1, "<eos>": 2}
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


def build_tiny_model(vocab_size: int, *, seed: int = 1234):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
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


def build_tiny_refiner(*, trainable: bool = True, seed: int = 1234) -> QwenPromptRefinerRole:
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_model(len(tokenizer), seed=seed)
    return QwenPromptRefinerRole(
        init_from="tiny-local",
        trainable=trainable,
        lora={"enable": True, "rank": 4, "alpha": 8},
        model_kind="causal_lm",
        torch_dtype="float32",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )


class TinyVelocityNet(torch.nn.Module):
    """Frozen base linear + trainable LoRA-style delta on the channel dim."""

    def __init__(self, channels: int, *, seed: int = 99):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base = torch.nn.Linear(channels, channels)
        self.base.requires_grad_(False)
        self.lora_a = torch.nn.Parameter(torch.zeros(channels, channels))
        self.lora_b = torch.nn.Parameter(torch.zeros(channels, channels))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            torch.nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.delta_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Latents are [B, T, C, H, W]; linears act on the channel dim.
        moved = x.movedim(2, -1)
        out = self.base(moved)
        if self.delta_enabled:
            out = out + moved @ self.lora_a.T @ self.lora_b.T
        return out.movedim(-1, 2)


class FakeWanStudent(TrainRoleBase):
    """Duck-typed WanPromptRLModel with a tiny trainable velocity net.

    Latents follow the training convention ``[B, T, C, H, W]``; the
    velocity net acts on the channel dim.
    """

    def __init__(self, *, channels: int = 3, latent_t: int = 2, seed: int = 99):
        self._trainable = True
        self.channels = channels
        self.latent_t = latent_t
        self.transformer = TinyVelocityNet(channels, seed=seed)
        self.dataloader: Any = None
        self.encoded_prompts: list[list[str]] = []
        self._device = torch.device("cpu")
        self.backward_calls = 0

    # -- TrainRoleBase --

    @property
    def device(self) -> torch.device:
        return self._device

    def checkpoint_modules(self):
        return {"transformer": self.transformer}

    def init_preprocessors(self, training_config) -> None:
        pass

    # -- WanPromptRLModel duck interface --

    def encode_prompts(self, prompts: list[str]):
        self.encoded_prompts.append(list(prompts))
        generator = torch.Generator().manual_seed(abs(hash(tuple(prompts))) % (2**31))
        embeds = torch.randn(len(prompts), 4, 6, generator=generator)
        mask = torch.ones(len(prompts), 4)
        return embeds, mask

    def latent_shape(self, batch_size: int):
        return (int(batch_size), self.latent_t, self.channels, 4, 4)

    def build_rollout_batch(self, embeds, mask, *, latent_shape):
        return SimpleNamespace(
            conditional_dict={
                "encoder_hidden_states": embeds,
                "encoder_attention_mask": mask,
            },
            timesteps=None,
        )

    def backward(self, loss, ctx, *, grad_accum_rounds):
        del ctx
        self.backward_calls += 1
        (loss / max(1, int(grad_accum_rounds))).backward()

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, attn_kind="dense"):
        return self.transformer(noisy_latents.float())

    @contextlib.contextmanager
    def adapter_disabled(self):
        previous = self.transformer.delta_enabled
        self.transformer.delta_enabled = False
        try:
            yield
        finally:
            self.transformer.delta_enabled = previous

    @torch.no_grad()
    def rollout(
        self,
        batch,
        *,
        batch_size=1,
        generator=None,
        num_steps=20,
        sde_steps=8,
        noise_scale=0.8,
        flow_shift=None,
        store_transitions=True,
    ):
        shape = self.latent_shape(batch_size)
        sigmas = torch.linspace(1.0, 0.0, int(num_steps) + 1).tolist()
        timesteps = [s * 1000.0 for s in sigmas[:-1]]
        sigma_max = sigmas[1] if len(sigmas) > 1 else sigmas[0]
        current = torch.randn(shape, generator=generator)
        transitions: list[SDETransition] = []
        sde_start = int(num_steps) - int(sde_steps)
        for step_idx in range(int(num_steps)):
            model_output = self.predict_noise(current, None, batch, conditional=True)
            if store_transitions and step_idx >= sde_start:
                prev, log_prob, _ = sde_step_from_model_output(
                    model_output,
                    current,
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    noise_scale=noise_scale,
                    sigma_max=sigma_max,
                    generator=generator,
                )
                transitions.append(
                    SDETransition(
                        sample=current.detach().clone(),
                        prev_sample=prev.detach().clone(),
                        timestep=timesteps[step_idx],
                        sigma=sigmas[step_idx],
                        sigma_next=sigmas[step_idx + 1],
                        old_log_prob=log_prob.detach(),
                    ))
                current = prev
            else:
                current = current + (sigmas[step_idx + 1] - sigmas[step_idx]) * model_output
        return SimpleNamespace(
            latents=current.detach(),
            transitions=transitions,
            sigmas=sigmas,
            timesteps=timesteps,
        )


    def transition_logprobs(
        self,
        transitions,
        batch,
        *,
        noise_scale,
        sigma_max,
        use_adapter=True,
        requires_grad=False,
    ):
        adapter_context = (contextlib.nullcontext() if use_adapter else self.adapter_disabled())
        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        results = []
        with adapter_context, grad_context:
            for transition in transitions:
                model_output = self.predict_noise(transition.sample, None, batch, conditional=True)
                log_prob, mean = transition_log_prob(
                    model_output,
                    transition.sample,
                    transition.prev_sample,
                    sigma=transition.sigma,
                    sigma_next=transition.sigma_next,
                    noise_scale=noise_scale,
                    sigma_max=sigma_max,
                )
                results.append((log_prob, mean))
        return results

    @torch.no_grad()
    def decode_latents(self, latents):
        # Fake media [B, C, T, H, W] in [0, 1] with RGB channels.
        media = torch.sigmoid(latents.float())
        if media.shape[2] >= 3:
            return media
        repeats = (3 + media.shape[2] - 1) // media.shape[2]
        return media.repeat(1, 1, repeats, 1, 1)[:, :, :3]


class FakeRewardProvider:
    """Deterministic per-slot scores; records scoring prompts."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (
            lambda sample: 1.0 + 0.25 * int(sample.sample_id.rsplit("-", 1)[-1]))
        self.scored: list[tuple[str, str]] = []

    def score(self, samples):
        results = []
        for sample in samples:
            self.scored.append((sample.sample_id, sample.original_prompt))
            score = float(self.score_fn(sample))
            results.append(
                RewardResult(
                    score=score,
                    details={
                        "visual_quality": score,
                        "text_alignment": score,
                        "physical_consistency": score,
                    },
                    sample_id=sample.sample_id,
                    request_id=sample.request_id,
                ))
        return results
