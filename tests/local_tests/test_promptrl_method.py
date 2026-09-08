# SPDX-License-Identifier: Apache-2.0
"""PromptRLMethod CPU integration tests with fake flow model + rewards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.rl.promptrl.method import PromptRLMethod
from fastvideo.train.methods.rl.promptrl.distributed import RewardConsistencyError
from fastvideo.train.methods.rl.promptrl.prompts import PromptDataset
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    TrainingConfig,
    TrainingLoopConfig,
)
from tests.local_tests.promptrl_fixtures import (
    FakeRewardProvider,
    FakeWanStudent,
    build_tiny_refiner,
)


class FailingRewardProvider:
    def score(self, samples):
        del samples
        raise RuntimeError("rank-local reward timeout")


def _make_cfg(method_overrides: dict | None = None) -> SimpleNamespace:
    method: dict = {
        "mode": "prompt_only",
        "group_size": 2,
        "retained_originals": 1,
        "rollout": {
            "num_steps": 4,
            "sde_steps": 2,
            "noise_scale": 0.5,
        },
        "refiner": {
            "max_new_tokens": 6,
            "temperature": 1.0,
        },
        "refiner_optimizer": {
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
        },
        "generator_optimizer": {
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
        },
    }
    if method_overrides:
        method.update(method_overrides)
    return SimpleNamespace(
        training=TrainingConfig(
            distributed=DistributedConfig(),
            data=DataConfig(seed=42),
            loop=TrainingLoopConfig(max_train_steps=10),
        ),
        method=method,
        validation={},
    )


def _make_method(
    mode: str,
    *,
    provider: FakeRewardProvider | None = None,
    refiner_seed: int = 1234,
    student_seed: int = 99,
    method_overrides: dict | None = None,
) -> tuple[PromptRLMethod, FakeWanStudent, FakeRewardProvider]:
    overrides = {"mode": mode}
    if method_overrides:
        overrides.update(method_overrides)
    cfg = _make_cfg(overrides)
    student = FakeWanStudent(seed=student_seed)
    refiner = build_tiny_refiner(seed=refiner_seed)
    method = PromptRLMethod(cfg=cfg, role_models={"student": student, "refiner": refiner})
    method.set_prompt_dataset(
        PromptDataset.from_rows([{
            "prompt": "a cat",
            "id": "p1",
            "reward_tag": "animals",
        }, {
            "prompt": "a dog runs",
            "id": "p2",
            "reward_tag": "animals",
        }]))
    reward_provider = provider or FakeRewardProvider()
    method.set_reward_provider(reward_provider)
    method.cuda_generator = torch.Generator().manual_seed(7)
    return method, student, reward_provider


def _params_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module, name_substr: str) -> bool:
    return any(
        not torch.equal(before[name], param.detach())
        for name, param in module.named_parameters()
        if name_substr in name)


def test_train_start_initializes_refiner_role(monkeypatch):
    from fastvideo.train.methods.base import TrainingMethod

    method, _, _ = _make_method("prompt_only")
    calls: list[str] = []
    monkeypatch.setattr(
        TrainingMethod,
        "on_train_start",
        lambda self: calls.append("student"),
    )
    monkeypatch.setattr(
        method.refiner,
        "on_train_start",
        lambda: calls.append("refiner"),
    )

    method.on_train_start()

    assert calls == ["student", "refiner"]


class TestPromptOnlyStep:
    def test_one_step_trains_only_refiner(self):
        method, student, provider = _make_method("prompt_only")
        refiner_before = _params_snapshot(method.refiner.model)
        student_before = _params_snapshot(student.transformer)

        loss_map, _, metrics = method.managed_train_step(None, 0)

        assert torch.isfinite(loss_map["total_loss"])
        # Both group slots scored, always against the original prompt.
        assert sorted(provider.scored) == [("slot-0", "a cat"), ("slot-1", "a cat")]
        # Refiner LoRA updated; the whole student stayed frozen.
        assert _changed(refiner_before, method.refiner.model, "lora_")
        assert not _changed(student_before, student.transformer, "")
        # Core observability metrics.
        for key in ("reward/group_mean", "reward/group_std", "reward/visual_quality",
                    "refine/valid_rate", "refine/refined_vs_original_gap", "lm/loss",
                    "lm/kl", "latency/rollout_sec", "latency/reward_sec",
                    "latency/backward_sec", "mem/peak_gpu_mb"):
            assert key in metrics, key
        assert "wan/loss" not in metrics  # prompt-only skips the Wan loss

    def test_valid_refinement_used_for_generation(self):
        method, student, _ = _make_method("prompt_only")
        method.refiner.generate_refinements = (  # type: ignore[method-assign]
            lambda prompts, **kwargs: ["<answer> refined cinematic cat </answer>"])
        method.managed_train_step(None, 0)
        # slot-1 (refined) generates from the refined prompt; slot-0
        # (retained original) from the original.
        assert student.encoded_prompts[0] == ["a cat"]
        assert student.encoded_prompts[1] == ["refined cinematic cat"]

    def test_malformed_completion_falls_back_to_original(self):
        method, student, _ = _make_method("prompt_only")
        method.refiner.generate_refinements = (  # type: ignore[method-assign]
            lambda prompts, **kwargs: ["garbage without tags"])
        method.managed_train_step(None, 0)
        assert student.encoded_prompts[1] == ["a cat"]

    def test_zero_variance_rewards_produce_no_update(self):
        constant_provider = FakeRewardProvider(score_fn=lambda sample: 2.0)
        method, student, _ = _make_method("prompt_only", provider=constant_provider)
        # Make both slots format-valid so the composite is flat.
        method.refiner.generate_refinements = (  # type: ignore[method-assign]
            lambda prompts, **kwargs: ["<answer> x </answer>"])
        refiner_before = _params_snapshot(method.refiner.model)
        _, _, metrics = method.managed_train_step(None, 0)
        assert metrics["reward/group_std"] == pytest.approx(0.0)
        assert metrics["reward/advantage_abs_mean"] == pytest.approx(0.0)
        assert not _changed(refiner_before, method.refiner.model, "lora_")

    def test_reward_provider_failure_is_raised_after_gather(self):
        method, _, _ = _make_method("prompt_only", provider=FailingRewardProvider())
        with pytest.raises(RewardConsistencyError, match="rank 0 reward failure"):
            method.managed_train_step(None, 0)


class TestJointStep:
    def test_one_step_trains_both_adapters(self):
        method, student, provider = _make_method("joint")
        refiner_before = _params_snapshot(method.refiner.model)
        student_before = _params_snapshot(student.transformer)

        loss_map, _, metrics = method.managed_train_step(None, 0)

        assert torch.isfinite(loss_map["total_loss"])
        assert sorted(provider.scored) == [("slot-0", "a cat"), ("slot-1", "a cat")]
        assert _changed(refiner_before, method.refiner.model, "lora_")
        assert _changed(student_before, student.transformer, "lora_")
        assert student.backward_calls > 0
        # Frozen student base must not move.
        assert not _changed(student_before, student.transformer, "base.")
        for key in ("wan/loss", "wan/kl", "wan/ratio", "wan/clip_fraction",
                    "grad_norm/refiner", "grad_norm/student"):
            assert key in metrics, key

    def test_joint_rollout_stores_only_sde_transitions(self):
        method, _, _ = _make_method("joint")
        captured = {}
        original_rollout = method.student.rollout

        def spy_rollout(batch, **kwargs):
            result = original_rollout(batch, **kwargs)
            captured.setdefault("transitions", []).append(len(result.transitions))
            return result

        method.student.rollout = spy_rollout  # type: ignore[method-assign]
        method.managed_train_step(None, 0)
        # 4-step rollout with sde_steps=2 -> exactly 2 stored transitions.
        assert captured["transitions"] == [2, 2]



class TestCheckpointResume:
    """Exact checkpoint/resume continuation with deterministic rewards."""

    @pytest.fixture()
    def gloo_world(self, tmp_path, monkeypatch):
        """Single-rank gloo world + CPU RNG snapshot shims.

        The production RNG snapshot path is CUDA-only; on CPU-only
        builds we snapshot the CPU generators with the same semantics.
        """
        import torch.distributed as dist
        from fastvideo.train.utils.checkpoint import (
            CheckpointManager,
            _resolve_resume_checkpoint,
        )

        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)

        def cpu_save(self, checkpoint_dir):
            torch.save(
                {
                    "torch_rng": torch.get_rng_state(),
                    "gen": self.method.cuda_generator.get_state(),
                },
                checkpoint_dir / "rng_state_rank0.pt",
            )

        def cpu_load(self, checkpoint_path):
            resolved = _resolve_resume_checkpoint(
                checkpoint_path, output_dir=self.output_dir)
            if resolved is None:
                return
            rng = torch.load(resolved / "rng_state_rank0.pt", weights_only=False)
            torch.set_rng_state(rng["torch_rng"])
            self.method.cuda_generator.set_state(rng["gen"])

        monkeypatch.setattr(CheckpointManager, "_save_rng_snapshot", cpu_save)
        monkeypatch.setattr(CheckpointManager, "load_rng_snapshot", cpu_load)
        yield
        dist.destroy_process_group()

    def _checkpoint_manager(self, method, output_dir):
        from fastvideo.train.utils.checkpoint import (
            CheckpointConfig,
            CheckpointManager,
        )

        return CheckpointManager(
            method=method,
            dataloader=method.student.dataloader,
            output_dir=str(output_dir),
            config=CheckpointConfig(save_steps=1, keep_last=0),
            callbacks=None,
            raw_config=None,
        )

    def test_exact_continuation(self, tmp_path, gloo_world):
        method_a, _, _ = _make_method("joint")
        manager_a = self._checkpoint_manager(method_a, tmp_path / "out")

        method_a.managed_train_step(None, 0)
        method_a.managed_train_step(None, 1)
        manager_a.save(2)
        method_a.managed_train_step(None, 2)
        expected = {
            "refiner": _params_snapshot(method_a.refiner.model),
            "student": _params_snapshot(method_a.student.transformer),
        }

        # Fresh instance resumes from checkpoint-2 and takes step 2.
        method_b, _, _ = _make_method("joint")
        manager_b = self._checkpoint_manager(method_b, tmp_path / "out")
        method_b.seed_optimizer_state_for_resume()
        resumed = manager_b.maybe_resume(
            resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-2"))
        assert resumed == 2
        manager_b.load_rng_snapshot(str(tmp_path / "out" / "checkpoint-2"))
        method_b.managed_train_step(None, 2)

        for name, param in method_b.refiner.model.named_parameters():
            assert torch.allclose(expected["refiner"][name], param.detach(), atol=1e-7), name
        for name, param in method_b.student.transformer.named_parameters():
            assert torch.allclose(expected["student"][name], param.detach(), atol=1e-7), name

    def test_rng_snapshot_restores_rollout_stream(self, tmp_path, gloo_world):
        """Resume reproduces the exact rollout noise of the live run."""
        method_a, _, _ = _make_method("prompt_only")
        manager_a = self._checkpoint_manager(method_a, tmp_path / "out")
        method_a.managed_train_step(None, 0)
        manager_a.save(1)
        # Capture the next noise draw the live run would produce.
        live_noise = torch.randn(4, generator=method_a.cuda_generator)

        method_b, _, _ = _make_method("prompt_only")
        manager_b = self._checkpoint_manager(method_b, tmp_path / "out")
        manager_b.maybe_resume(resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-1"))
        manager_b.load_rng_snapshot(str(tmp_path / "out" / "checkpoint-1"))
        resumed_noise = torch.randn(4, generator=method_b.cuda_generator)
        assert torch.equal(live_noise, resumed_noise)


class TestMilestoneHandoff:
    def test_joint_initializes_from_prompt_only_adapter(self, tmp_path):
        """Prompt-only refiner checkpoint -> joint run; Wan LoRA at zero."""
        method_prompt, _, _ = _make_method("prompt_only")
        method_prompt.managed_train_step(None, 0)
        written = method_prompt.export_bundle(str(tmp_path / "bundle"))
        assert "refiner" in written

        # New joint refiner loads the prompt-only adapter at construction.
        refiner = build_tiny_refiner()
        refiner.load_adapter(written["refiner"])
        trained = {name: param.detach().clone()
                   for name, param in method_prompt.refiner.model.named_parameters()
                   if "lora_" in name}
        loaded = {name: param.detach().clone()
                  for name, param in refiner.model.named_parameters() if "lora_" in name}
        for name in trained:
            assert torch.allclose(trained[name], loaded[name], atol=1e-6), name

        # Wan LoRA starts at exact zero (lora_b zero-init).
        student = FakeWanStudent()
        assert torch.equal(student.transformer.lora_b, torch.zeros_like(student.transformer.lora_b))
