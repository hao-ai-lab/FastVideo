# SPDX-License-Identifier: Apache-2.0
"""Execute FastVideo's DMD2 method with public Wan weights and real video data."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
import torch_npu

from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
from fastvideo.train.utils.builder import build_from_config
from fastvideo.train.utils.activation_memory import MemoryProfile
from fastvideo.train.utils.config import RunConfig
from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)

from profile_wan import policy_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--training-sample', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument(
        '--policy',
        choices=['save', 'recompute', 'ops', 'offload', 'mixed', 'block_skip', 'prefix', 'suffix', 'uniform'],
        default='recompute')
    parser.add_argument('--save-count', type=int, default=15)
    parser.add_argument('--student-policy', choices=['save', 'recompute', 'offload'])
    parser.add_argument('--critic-policy', choices=['save', 'recompute', 'offload'])
    parser.add_argument('--critic-save-count', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=2)
    parser.add_argument('--profile')
    parser.add_argument('--calibrate-from')
    parser.add_argument('--budget-gib', type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.npu.set_device(local_rank)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank))
    maybe_init_distributed_environment_and_model_parallel(
        1, 1, distributed_init_method='env://' if world > 1 else 'tcp://127.0.0.1:29883')
    sample = torch.load(args.training_sample, map_location='cpu', weights_only=True)
    config = TrainingConfig(
        model_path=args.model,
        dit_precision='fp32',
        distributed=DistributedConfig(num_gpus=world, hsdp_replicate_dim=1, hsdp_shard_dim=world),
        pipeline_config=WanT2V480PConfig(),
        data=DataConfig(data_path=args.data,
                        train_batch_size=1,
                        dataloader_num_workers=0,
                        training_cfg_rate=0,
                        seed=2026,
                        num_height=sample['height'],
                        num_width=sample['width'],
                        num_latent_t=sample['latents'].shape[2],
                        num_frames=len(sample['frame_indices'])),
        optimizer=OptimizerConfig(learning_rate=1e-5, betas=(0.0, 0.999), weight_decay=0.01),
        loop=TrainingLoopConfig(max_train_steps=args.steps + args.warmup),
        model=ModelTrainingConfig(
            activation_memory_profile=args.profile or '',
            activation_memory_budget_gib=args.budget_gib,
            activation_memory_plan_output=str(Path(args.output).with_suffix('.plan.json')) if args.profile else ''),
    )
    count = json.loads((Path(args.model) / 'transformer/config.json').read_text())['num_layers']
    holder = torch.nn.Module()
    holder.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(count)])
    roles = {}
    for role in ('student', 'teacher', 'critic'):
        roles[role] = {
            '_target_': 'fastvideo.train.models.wan.WanModel',
            'init_from': args.model,
            'trainable': role != 'teacher',
            'attention_backend': 'TORCH_SDPA',
        }
        if role != 'teacher' and not args.profile:
            save_count = (args.critic_save_count
                          if role == 'critic' and args.critic_save_count is not None else args.save_count)
            role_policy = getattr(args, f'{role}_policy') or args.policy
            roles[role]['activation_memory_plan'] = policy_for(holder,
                                                               role_policy,
                                                               checkpoint_every=args.checkpoint_every,
                                                               save_count=save_count)
    cfg = RunConfig(models=roles,
                    training=config,
                    callbacks={},
                    raw={},
                    method={
                        '_target_': 'fastvideo.train.methods.distribution_matching.dmd2.DMD2Method',
                        'rollout_mode': 'data_latent',
                        'generator_update_interval': 1,
                        'real_score_guidance_scale': 5.0,
                        'dmd_denoising_steps': [1000, 750, 500, 250],
                        'fake_score_learning_rate': 1e-5,
                        'fake_score_betas': [0.0, 0.999],
                        'fake_score_lr_scheduler': 'constant',
                    })
    _, method, dataloader, _ = build_from_config(cfg)
    method.student.negative_prompt_embeds = sample['negative_prompt_embeds'].to('npu', torch.bfloat16)
    negative_mask = torch.zeros(sample['negative_prompt_embeds'].shape[:2], device='npu')
    negative_mask[:, 0] = 1
    method.student.negative_prompt_attention_mask = negative_mask
    method.on_train_start()
    for role in ('student', 'critic'):
        getattr(method, role).transformer.train()
    method.teacher.transformer.eval()
    stream = itertools.cycle(dataloader)
    iteration = 0

    def step(observe: Callable[[str], None] | None = None) -> dict[str, Any]:
        nonlocal iteration
        optimizers = method.get_optimizers(iteration)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses, outputs, metrics = method.single_train_step(next(stream), iteration)
        method.backward(losses, outputs)
        if observe is not None:
            observe('clipping')
        norms = {
            role: clip_grad_norm_if_needed(module, 1.0)
            for role, module in method.get_grad_clip_targets(iteration).items()
        }
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise RuntimeError('Non-finite DMD2 loss')
        for optimizer in optimizers:
            if observe is not None:
                role = next(name for name, candidate in method._optimizer_dict.items() if candidate is optimizer)
                observe(f'optimizer.{role}')
            optimizer.step()
        for scheduler in method.get_lr_schedulers(iteration):
            scheduler.step()
        record = {
            'iteration': iteration,
            'losses': {
                key: float(value.detach())
                for key, value in losses.items()
            },
            'gradient_norms': norms,
            'metrics': metrics
        }
        del losses, outputs
        iteration += 1
        return record

    if args.calibrate_from:
        from role_profile import calibrate_dmd2

        if args.profile or args.policy != 'recompute' or args.student_policy or args.critic_policy or world != 1:
            raise ValueError('Role calibration requires one rank and full recomputation')
        template = MemoryProfile.load(args.calibrate_from)
        if template.metadata['input_shape'] != list(sample['latents'].shape):
            raise ValueError('Block and role calibration must use the same latent shape')
        if template.metadata['text_tokens'] != sample['prompt_embeds'].shape[1]:
            raise ValueError('Block and role calibration must use the same text length')
        started = time.perf_counter()
        profile = calibrate_dmd2(method,
                                 step,
                                 template,
                                 warmup=args.warmup,
                                 repetitions=args.steps,
                                 training_sample=args.training_sample)
        profile.metadata['calibration_ms'] = (time.perf_counter() - started) * 1000
        profile.metadata['allocator_config'] = os.environ.get('PYTORCH_NPU_ALLOC_CONF', '')
        profile.save(args.output)
        print(json.dumps({'profile': args.output, 'events': len(profile.events), 'blocks': len(profile.blocks)}))
        cleanup_dist_env_and_memory()
        return
    for _ in range(args.warmup):
        step()
    records = []
    for _ in range(args.steps):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        started = time.perf_counter()
        record = step()
        torch.npu.synchronize()
        record.update(step_ms=(time.perf_counter() - started) * 1000,
                      peak_allocated_bytes=torch.npu.max_memory_allocated(),
                      peak_reserved_bytes=torch.npu.max_memory_reserved())
        records.append(record)
    torch.npu.synchronize()
    result = {
        'executor': 'FastVideo DMD2Method / WanModel / FSDP2',
        'world_size': world,
        'model': args.model,
        'training_sample': args.training_sample,
        'steps': records,
        'warmup_steps': args.warmup,
        'teacher_trainable': method.teacher._trainable,
        'policy': args.policy if not args.profile else 'budget',
        'profile': args.profile,
        'manual_policies': {
            role: config.get('activation_memory_plan')
            for role, config in roles.items() if role != 'teacher'
        },
        'master_dtype': 'float32',
        'autocast_dtype': 'bfloat16',
        'torch': torch.__version__,
        'torch_npu': torch_npu.__version__,
        'allocator_config': os.environ.get('PYTORCH_NPU_ALLOC_CONF', ''),
        'peak_allocated_bytes': max(record['peak_allocated_bytes'] for record in records)
    }
    if rank == 0:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
