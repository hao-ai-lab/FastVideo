from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.methods.rl.interleave_thinker import InterleaveThinkerRLMethod
from fastvideo.train.methods.rl.rewards import score_interleave_thinker_rewards
from fastvideo.train.models.base import RoleModelBase
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


class _FakeInterleaveActor(RoleModelBase):

    def __init__(self, *, trainable=True):
        super().__init__(trainable=trainable)
        self.transformer = torch.nn.Linear(1, 1)
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


class _FakeReferenceActor(_FakeInterleaveActor):

    def __init__(self):
        super().__init__(trainable=False)
        self.reference_calls = []
        self.transformer.train()

    def reference_logprobs_for_interleave_rollouts(self, rollouts):
        self.reference_calls.append([dict(rollout) for rollout in rollouts])
        return [[-0.1, -0.2] for _ in rollouts]


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
    method = InterleaveThinkerRLMethod(
        cfg=_cfg({
            "clip_range": 0.15,
            "kl_coef": 0.05,
            "micro_batch_size_per_device_for_update": 2,
        }),
        role_models={"student": actor},
    )
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
    assert train_kwargs["clip_range"] == 0.15
    assert train_kwargs["kl_coef"] == 0.05
    assert train_kwargs["update_micro_batch_size"] == 2
    assert train_kwargs["rollouts"][0].get("reference_logprobs") is None


def test_interleave_thinker_namespaces_groups_and_samples_across_input_batches():
    actor = _FakeInterleaveActor()
    method = InterleaveThinkerRLMethod(
        cfg=_cfg({
            "num_batches_per_step": 2,
        }),
        role_models={"student": actor},
    )
    batch = {"origin_prompt": ["prompt-a", "prompt-b"]}

    _, _, metrics = method.managed_train_step(iter([batch, batch]), iteration=4)

    assert metrics["interleave/num_rollouts"] == 8.0
    assert metrics["interleave/num_groups"] == 4.0
    train_rollouts = actor.train_calls[0]["rollouts"]
    assert [rollout["sample_index"] for rollout in train_rollouts] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [rollout["group_key"] for rollout in train_rollouts] == [
        "batch:0:prompt-a",
        "batch:0:prompt-a",
        "batch:0:prompt-b",
        "batch:0:prompt-b",
        "batch:1:prompt-a",
        "batch:1:prompt-a",
        "batch:1:prompt-b",
        "batch:1:prompt-b",
    ]


def test_interleave_thinker_adapts_exported_callable_reward_rows_to_tensors():
    actor = _FakeInterleaveActor()
    method = InterleaveThinkerRLMethod(
        cfg=_cfg({
            "reward_scorer": score_interleave_thinker_rewards,
        }),
        role_models={"student": actor},
    )

    _, _, metrics = method.managed_train_step(
        iter([{
            "origin_prompt": ["prompt-a", "prompt-b"]
        }]),
        iteration=5,
    )

    assert torch.isclose(metrics["interleave/reward/overall"], torch.tensor(0.75))
    assert torch.isclose(metrics["interleave/reward/format_reward"], torch.tensor(1.0))


def test_interleave_thinker_method_attaches_reference_logprobs_to_rollouts():
    actor = _FakeInterleaveActor()
    reference = _FakeReferenceActor()
    method = InterleaveThinkerRLMethod(
        cfg=_cfg({
            "kl_coef": 0.01
        }),
        role_models={
            "student": actor,
            "reference": reference,
        },
    )
    batch = {"origin_prompt": ["prompt-a", "prompt-b"]}

    _, _, metrics = method.managed_train_step(iter([batch]), iteration=3)

    assert metrics["interleave/reference_logprob_rollouts"] == 4.0
    assert len(reference.reference_calls) == 1
    assert len(reference.reference_calls[0]) == 4
    assert reference.transformer.training is False
    assert all(not param.requires_grad for param in reference.transformer.parameters())
    train_rollouts = actor.train_calls[0]["rollouts"]
    assert [rollout["reference_logprobs"] for rollout in train_rollouts] == [[-0.1, -0.2]] * 4


def test_interleave_thinker_method_requires_frozen_reference_model():
    with pytest.raises(ValueError, match="models.reference.trainable=false"):
        InterleaveThinkerRLMethod(
            cfg=_cfg(),
            role_models={
                "student": _FakeInterleaveActor(),
                "reference": _FakeInterleaveActor(trainable=True),
            },
        )


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


def test_interleave_thinker_method_instantiates_configured_edit_scorer():
    actor = _FakeInterleaveActor()
    actor.generate_interleave_responses = None
    method = InterleaveThinkerRLMethod(
        cfg=_cfg({
            "edit_scorer": {
                "_target_": "fastvideo.train.methods.rl.rewards.ConstantInterleaveEditScorer",
                "semantic_reward": 0.9,
                "quality_reward": 0.1,
            }
        }),
        role_models={"student": actor},
    )
    batch = {
        "origin_prompt": ["prompt-a"],
        "previous_prompt": ["prompt-a previous"],
        "ground_truth": [{
            "success": True
        }],
        "response": [_response(True)],
    }

    _, _, metrics = method.managed_train_step(iter([batch]), iteration=2)

    assert torch.isclose(metrics["interleave/reward/edited_image_reward_semantic"], torch.tensor(0.9))
    assert torch.isclose(metrics["interleave/reward/edited_image_reward_quality"], torch.tensor(0.1))


def test_interleave_thinker_config_parses_public_yaml():
    cfg = load_run_config("examples/train/configs/rl/interleave_thinker/critic_grpo.yaml")

    assert cfg.models["student"]["_target_"] == (
        "fastvideo.train.models.interleave_thinker.InterleaveThinkerCriticModel")
    assert cfg.models["student"]["init_from"] == "InterleaveThinker/Critic-SFT-8B"
    assert cfg.models["student"]["dataset_kind"] == "critic_rl"
    assert cfg.models["student"]["image_dir"] == "data/InterleaveThinker/Train-Data"
    assert cfg.models["student"]["lora"]["enable"] is True
    assert cfg.models["reference"]["_target_"] == (
        "fastvideo.train.models.interleave_thinker.InterleaveThinkerCriticModel")
    assert cfg.models["reference"]["init_from"] == "InterleaveThinker/Critic-SFT-8B"
    assert cfg.models["reference"]["trainable"] is False
    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.interleave_thinker.InterleaveThinkerRLMethod"
    assert cfg.method["edit_scorer"]["_target_"] == (
        "fastvideo.train.methods.rl.rewards.GeminiNanoBananaEditScorer")
    assert cfg.method["num_generations"] == 8
    assert cfg.method["clip_range"] == 0.2
    assert cfg.method["kl_coef"] == 0.01
    assert cfg.method["micro_batch_size_per_device_for_update"] == 1
    assert cfg.training.data.data_path.endswith("critic_rl.jsonl")
    assert cfg.training.optimizer.learning_rate == 2.0e-6


def test_existing_finetune_method_uses_default_optimizer_path():
    assert FineTuneMethod.manages_optimization(FineTuneMethod.__new__(FineTuneMethod)) is False
