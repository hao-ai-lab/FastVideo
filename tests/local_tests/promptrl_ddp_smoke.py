# SPDX-License-Identifier: Apache-2.0
"""Two-GPU PromptRL refiner DDP smoke test.

Run with:

    torchrun --standalone --nproc-per-node=2 \
        -m tests.local_tests.promptrl_ddp_smoke
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.train.methods.rl.promptrl.method import PromptRLMethod
from fastvideo.train.roles.qwen_refiner import QwenPromptRefinerRole
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    TrainingConfig,
    TrainingLoopConfig,
)
from tests.local_tests.promptrl_fixtures import (
    FakeWanStudent,
    build_tiny_model,
    build_tiny_tokenizer,
)


def _build_method(local_rank: int) -> PromptRLMethod:
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_model(len(tokenizer), seed=1234)
    refiner = QwenPromptRefinerRole(
        init_from="tiny-local",
        trainable=True,
        lora={"enable": True, "rank": 4, "alpha": 8},
        model_kind="causal_lm",
        torch_dtype="float32",
        model=model,
        tokenizer=tokenizer,
        device=f"cuda:{local_rank}",
    )
    cfg = SimpleNamespace(
        training=TrainingConfig(
            distributed=DistributedConfig(sp_size=1),
            data=DataConfig(seed=42),
            loop=TrainingLoopConfig(max_train_steps=1),
        ),
        method={
            "mode": "prompt_only",
            "group_size": 2,
            "retained_originals": 1,
            "refiner_optimizer": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
            "generator_optimizer": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
        },
        validation={},
    )
    return PromptRLMethod(
        cfg=cfg,
        role_models={
            "student": FakeWanStudent(),
            "refiner": refiner,
        },
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
    )
    try:
        method = _build_method(local_rank)
        method.on_train_start()
        assert method.refiner._ddp_model is not None

        logprobs, _ = method.refiner.sequence_logprobs(
            ["a cat"],
            ["<answer> cinematic cat </answer>"],
            requires_grad=True,
        )
        (-logprobs.mean()).backward()
        gradients = [
            parameter.grad
            for parameter in method.refiner.trainable_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        ]
        assert gradients
        gradient = gradients[0]
        assert torch.isfinite(gradient).all()
        gathered = [torch.empty_like(gradient) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, gradient)
        assert all(
            torch.allclose(gathered[0], candidate, atol=1e-6, rtol=1e-5)
            for candidate in gathered[1:]
        )
        if dist.get_rank() == 0:
            print({
                "world_size": dist.get_world_size(),
                "ddp_wrapped": True,
                "gradient_norm": float(gradient.float().norm()),
            })
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
