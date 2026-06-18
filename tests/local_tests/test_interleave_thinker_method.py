from types import SimpleNamespace
from typing import Any, Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.methods.rl.interleave_thinker import InterleaveThinkerRLMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import (
    CheckpointConfig,
    DataConfig,
    DistributedConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    TrackerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


def _response(success=True):
    success_text = "true" if success else "false"
    return f"""
    <think>reasoning</think>
    <answer>{{"previous_step_success": {success_text}, "refine_prompt": "improve prompt"}}</answer>
    """


class _FakeInterleaveActor(ModelBase):

    def __init__(self):
        super().__init__(trainable=True)
        self.transformer = torch.nn.Module()
        self.noise_scheduler = SimpleNamespace(num_train_timesteps=0)
        self.generate_calls = []
        self.train_calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def init_preprocessors(self, training_config):
        self.training_config = training_config

    def generate_interleave_responses(self, batch, **kwargs):
        self.generate_calls.append((batch, kwargs))
        rollouts = []
        rewards = {
            "prompt-a": [0.0, 1.0],
            "prompt-b": [0.25, 0.75],
        }
        for prompt, values in rewards.items():
            for generation_idx, reward_value in enumerate(values):
                rollouts.append({
                    "group_key": prompt,
                    "origin_prompt": prompt,
                    "previous_prompt": f"{prompt} previous",
                    "response": _response(success=True),
                    "ground_truth": {
                        "success": True,
                        "semantics": 0.0,
                        "quality": 0.0,
                    },
                    "edit_scores": {
                        "edited_image_reward_semantic": reward_value,
                        "edited_image_reward_quality": 0.0,
                    },
                    "generation_index": generation_idx,
                })
        return rollouts

    def train_interleave_rollouts(self, **kwargs):
        self.train_calls.append(kwargs)
        advantages = kwargs["advantages"]
        return (
            {
                "total_loss": advantages.pow(2).mean()
            },
            {
                "actor/updates": 1.0
            },
        )

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        raise NotImplementedError

    def add_noise(self, clean_latents, noise, timestep):
        del clean_latents, noise, timestep
        raise NotImplementedError

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, cfg_uncond=None, attn_kind="dense"):
        del noisy_latents, timestep, batch, conditional, cfg_uncond, attn_kind
        raise NotImplementedError

    def backward(self, loss, ctx, *, grad_accum_rounds):
        del loss, ctx, grad_accum_rounds
        raise NotImplementedError


def _cfg(method_overrides=None):
    method = {
        "num_generations": 2,
        "num_batches_per_step": 1,
        "format_weight": 0.0,
        "judge_accuracy_weight": 0.0,
        "semantic_weight": 1.0,
        "quality_weight": 0.0,
        "terminal_progress": False,
    }
    method.update(method_overrides or {})
    return SimpleNamespace(
        method=method,
        validation={},
        training=TrainingConfig(
            distributed=DistributedConfig(),
            data=DataConfig(seed=123, train_batch_size=1),
            optimizer=OptimizerConfig(learning_rate=0.0),
            loop=TrainingLoopConfig(max_train_steps=10, gradient_accumulation_steps=3),
            checkpoint=CheckpointConfig(),
            tracker=TrackerConfig(trackers=[]),
            model=ModelTrainingConfig(),
        ),
    )


def test_interleave_thinker_managed_step_scores_advantages_and_calls_actor_update():
    actor = _FakeInterleaveActor()
    method = InterleaveThinkerRLMethod(cfg=_cfg(), role_models={"student": actor})
    batch = {"origin_prompt": ["prompt-a", "prompt-b"]}

    loss_map, outputs, metrics = method.managed_train_step(iter([batch]), iteration=7)

    assert outputs == {}
    assert torch.isclose(loss_map["total_loss"], torch.tensor(0.9996), atol=1.0e-3)
    assert metrics["actor/updates"] == 1.0
    assert metrics["interleave/num_rollouts"] == 4.0
    assert metrics["interleave/num_groups"] == 2.0
    assert torch.isclose(metrics["interleave/reward/overall"], torch.tensor(0.5))

    assert len(actor.generate_calls) == 1
    _, generate_kwargs = actor.generate_calls[0]
    assert generate_kwargs["num_generations"] == 2
    assert generate_kwargs["temperature"] == 1.0
    assert generate_kwargs["top_p"] == 1.0

    train_kwargs = actor.train_calls[0]
    advantages = train_kwargs["advantages"]
    assert advantages.shape == (4,)
    assert torch.isclose(advantages[:2].sum(), torch.tensor(0.0), atol=1.0e-5)
    assert torch.isclose(advantages[2:].sum(), torch.tensor(0.0), atol=1.0e-5)
    assert train_kwargs["gradient_accumulation_steps"] == 3


def test_interleave_thinker_method_can_use_offline_response_batches():
    actor = _FakeInterleaveActor()
    actor.generate_interleave_responses = None
    method = InterleaveThinkerRLMethod(cfg=_cfg(), role_models={"student": actor})
    batch = {
        "origin_prompt": ["prompt-a"],
        "ground_truth": [{
            "success": True
        }],
        "responses": [[_response(True), _response(True)]],
        "edit_scores": [[{
            "edited_image_reward_semantic": 0.0
        }, {
            "edited_image_reward_semantic": 1.0
        }]],
    }

    loss_map, _, metrics = method.managed_train_step(iter([batch]), iteration=1)

    assert "total_loss" in loss_map
    assert metrics["interleave/num_rollouts"] == 2.0
    assert len(actor.train_calls) == 1


def test_interleave_thinker_config_parses_public_yaml():
    cfg = load_run_config("examples/train/configs/rl/interleave_thinker/critic_grpo.yaml")

    assert cfg.models["student"]["_target_"] == (
        "fastvideo.train.models.interleave_thinker.InterleaveThinkerCriticModel")
    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.interleave_thinker.InterleaveThinkerRLMethod"
    assert cfg.method["num_generations"] == 8
    assert cfg.training.data.data_path.endswith("critic_rl.jsonl")
    assert cfg.training.optimizer.learning_rate == 2.0e-6


def test_existing_finetune_method_uses_default_optimizer_path():
    assert FineTuneMethod.manages_optimization(FineTuneMethod.__new__(FineTuneMethod)) is False
