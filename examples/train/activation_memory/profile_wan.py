# SPDX-License-Identifier: Apache-2.0
"""Profile and execute memory budgets through FastVideo's native Wan adapter."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.tensor import DTensor

from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.forward_context import set_forward_context
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
from fastvideo.models.wan.transformer import WanTransformerBlock
from fastvideo.pipelines import TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.wan import WanModel
from fastvideo.train.utils.activation_memory import (
    BlockCosts,
    MemoryProfile,
    PolicyCost,
    apply_activation_memory_plan,
    plan_activation_memory,
)
from fastvideo.train.utils.training_config import DistributedConfig, TrainingConfig

def policy_for(model, policy, checkpoint_every=2, save_count=None):
    count = len(model.blocks)
    if policy in {'prefix', 'suffix', 'uniform'}:
        if save_count is None or not 0 <= save_count <= count:
            raise ValueError('Manual layouts require --save-count between zero and the number of blocks')
        if policy == 'prefix':
            saved = set(range(save_count))
        elif policy == 'suffix':
            saved = set(range(count - save_count, count))
        else:
            saved = {int((index + 0.5) * count / save_count) for index in range(save_count)}
        return {f'blocks.{index}': 'save' if index in saved else 'recompute' for index in range(count)}
    if policy == 'block_skip':
        return {f'blocks.{index}': 'recompute' if index % checkpoint_every == 0 else 'save'
                for index in range(len(model.blocks))}
    if policy == 'mixed':
        return {f'blocks.{index}': ('save', 'recompute', 'offload')[index % 3]
                for index in range(len(model.blocks))}
    return {f'blocks.{index}': policy for index in range(len(model.blocks))}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)




def sync() -> None:
    torch.npu.synchronize()


def combine_rank_profiles(profiles: list[MemoryProfile]) -> MemoryProfile:
    """Use the largest rank-local storage and time for a common FSDP plan."""
    reference = profiles[0]
    blocks = []
    for index, block in enumerate(reference.blocks):
        choices = []
        for choice_index, choice in enumerate(block.choices):
            alternatives = [profile.blocks[index].choices[choice_index] for profile in profiles]
            choices.append(
                PolicyCost(choice.policy, max(item.time_ms for item in alternatives),
                           tuple(map(max, zip(*(item.device_bytes for item in alternatives), strict=True))),
                           tuple(map(max, zip(*(item.host_bytes for item in alternatives), strict=True)))))
        blocks.append(BlockCosts(block.role, block.block, tuple(choices)))
    metadata = {**reference.metadata, 'world_size': len(profiles), 'rank_profiles': [p.metadata for p in profiles]}
    return MemoryProfile(reference.events,
                         tuple(map(max, zip(*(profile.fixed_device_bytes for profile in profiles), strict=True))),
                         tuple(blocks), metadata)


class TrainingStep:

    def __init__(self, model: WanModel, sample: dict[str, Any], seed: int) -> None:
        self.model = model
        clean = sample['latents'].permute(0, 2, 1, 3, 4).to('npu')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(clean.shape, generator=generator).to('npu')
        self.noisy, self.target = (clean + noise) / 2, noise - clean
        self.batch = TrainingBatch()
        self.batch.timesteps = torch.tensor([500.], device='npu')
        self.batch.attn_metadata = None
        self.batch.conditional_dict = {
            'encoder_hidden_states': sample['prompt_embeds'].to('npu', torch.bfloat16),
            'encoder_attention_mask': torch.ones(sample['prompt_embeds'].shape[:2], device='npu'),
        }
        self.optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=1e-5, foreach=False)

    def forward(self) -> torch.Tensor:
        return self.model.predict_noise(self.noisy, self.batch.timesteps, self.batch, conditional=True)

    def backward(self, prediction: torch.Tensor) -> torch.Tensor:
        loss = (prediction.float() - self.target).square().mean()
        self.model.backward(loss, (self.batch.timesteps, self.batch.attn_metadata), grad_accum_rounds=1)
        return loss

    def __call__(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.forward()
        loss = self.backward(prediction)
        self.optimizer.step()
        return float(loss.detach())


def measure_block(args: argparse.Namespace, model: WanModel, step: TrainingStep) -> dict[str, dict[str, Any]]:
    wrapped = model.transformer.blocks[0]
    block = wrapped._checkpoint_wrapped_module
    captured = []

    def capture(module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        state = {
            name: (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
            for name, value in module.named_parameters()
        }
        captured.append((values, state))

    handle = block.register_forward_pre_hook(capture)
    with torch.no_grad():
        step.forward()
    handle.remove()
    block_inputs, state = captured[0]
    config = json.loads((Path(args.model) / 'transformer/config.json').read_text())
    isolated = WanTransformerBlock(
        dim=config['num_attention_heads'] * config['attention_head_dim'],
        ffn_dim=config['ffn_dim'],
        num_heads=config['num_attention_heads'],
        qk_norm=config['qk_norm'],
        cross_attn_norm=config['cross_attn_norm'],
        eps=config['eps'],
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
    )
    isolated.load_state_dict(state, assign=True)
    isolated.to('npu')
    parameter_pointers = {parameter.untyped_storage().data_ptr() for parameter in isolated.parameters()}
    shared_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors(block_inputs[1:])}
    input_bytes = block_inputs[0].untyped_storage().nbytes()

    def values() -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone().requires_grad_(True) if index < 3 else value
                     for index, value in enumerate(block_inputs))

    def run(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.autocast('npu', dtype=torch.bfloat16), set_forward_context(current_timestep=step.batch.timesteps,
                                                                              attn_metadata=None):
            return module(*inputs)

    def backward(output: torch.Tensor) -> None:
        with set_forward_context(current_timestep=step.batch.timesteps, attn_metadata=None):
            output.float().square().mean().backward()

    costs: dict[str, dict[str, Any]] = {}
    for policy in ('save', 'recompute', 'offload'):
        holder = torch.nn.Module()
        holder.add_module('block', isolated)
        apply_activation_memory_plan(holder, {'block': policy})
        saved: dict[int, int] = {}
        retained_parameters: dict[int, int] = {}
        host_bytes = 0

        def pack(tensor: torch.Tensor,
                 saved: dict[int, int] = saved,
                 retained_parameters: dict[int, int] = retained_parameters) -> torch.Tensor:
            nonlocal host_bytes
            host_bytes += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            if storage.data_ptr() in parameter_pointers:
                # FSDP releases unsharded BF16 weights after forward unless
                # autograd retains them. The FP32 masters stay in fixed storage.
                if tensor.dtype == torch.bfloat16:
                    retained_parameters[storage.data_ptr()] = storage.nbytes()
            elif storage.data_ptr() not in shared_pointers:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        if policy == 'save':
            inputs = values()
            # Shared conditions are retained by the surrounding model graph.
            shared_pointers.update(tensor.untyped_storage().data_ptr() for tensor in inputs[1:3])
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                output = run(holder.block, inputs)
            backward(output)
            del inputs, output
        records = []
        for repetition in range(args.warmup + args.steps):
            for parameter in isolated.parameters():
                parameter.grad = torch.zeros_like(parameter)
            inputs = values()
            sync()
            baseline = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            started = time.perf_counter()
            output = run(holder.block, inputs)
            sync()
            forward_peak = torch.npu.max_memory_allocated() - baseline + input_bytes
            torch.npu.reset_peak_memory_stats()
            backward(output)
            sync()
            records.append({
                'time_ms': (time.perf_counter() - started) * 1000,
                'forward_peak': forward_peak,
                'backward_peak': torch.npu.max_memory_allocated() - baseline + input_bytes
            })
            del inputs, output
        records = records[args.warmup:]
        costs[policy] = {
            'time_ms':
            sum(row['time_ms'] for row in records) / len(records),
            'forward_peak':
            max(row['forward_peak'] for row in records),
            'backward_peak':
            max(row['backward_peak'] for row in records),
            'saved_bytes':
            sum(saved.values()) +
            sum(retained_parameters.values()) if policy == 'save' else input_bytes if policy == 'recompute' else 0,
            'retained_parameter_bytes':
            sum(retained_parameters.values()),
            'forward_parameter_dtypes':
            sorted({str(parameter.dtype)
                    for parameter in isolated.parameters()}),
            'host_bytes':
            costs['save']['host_bytes'] if policy == 'offload' else host_bytes,
            'repetitions':
            records,
        }
        del holder
    del isolated, state
    captured.clear()
    block_inputs = ()
    gc.collect()
    torch.npu.empty_cache()
    return costs


def observe_events(args: argparse.Namespace, model: WanModel, step: TrainingStep) -> tuple[tuple[str, ...], list[int]]:
    """Attribute continuous execution intervals, including FSDP work between hooks."""
    count = len(model.transformer.blocks)
    events = tuple([f'student.forward.{index}' for index in range(count)] + ['student.head_and_loss'] +
                   [f'student.backward.{index}' for index in reversed(range(count))] + ['optimizer'])
    for _ in range(args.warmup):
        step()
    observed, handles = [0] * len(events), []
    active: int | None = None

    def transition(event: int | None) -> None:
        nonlocal active
        sync()
        if active is not None:
            observed[active] = max(observed[active], torch.npu.max_memory_allocated())
        active = event
        torch.npu.reset_peak_memory_stats()

    def begin(event: int) -> Callable[..., None]:
        def hook(module: torch.nn.Module, *unused: Any) -> None:
            transition(event)
        return hook

    for index, block in enumerate(model.transformer.blocks):
        handles.extend([
            block.register_forward_pre_hook(begin(index)),
            block.register_full_backward_pre_hook(begin(2 * count - index))
        ])
    handles.append(model.transformer.blocks[-1].register_forward_hook(begin(count)))
    for _ in range(args.steps):
        step.optimizer.zero_grad(set_to_none=True)
        transition(0)
        prediction = step.forward()
        step.backward(prediction)
        transition(len(events) - 1)
        step.optimizer.step()
        transition(None)
        del prediction
    for handle in handles:
        handle.remove()
    return events, observed


def calibrate(args: argparse.Namespace, model: WanModel, step: TrainingStep) -> MemoryProfile:
    costs = measure_block(args, model, step)
    count = len(model.transformer.blocks)
    events, observed = observe_events(args, model, step)
    blocks = []
    for index in range(count):
        end = 2 * count - index
        choices = []
        for policy, cost in costs.items():
            device, host = [0] * len(events), [0] * len(events)
            for event in range(index, end + 1):
                device[event] = cost['saved_bytes']
                if policy == 'offload':
                    host[event] = cost['host_bytes']
            device[index], device[end] = max(device[index], cost['forward_peak']), max(device[end], cost['backward_peak'])
            choices.append(PolicyCost(policy, cost['time_ms'], tuple(device), tuple(host)))
        blocks.append(BlockCosts('student', f'blocks.{index}', tuple(choices)))
    reference = [sum(next(choice for choice in block.choices if choice.policy == 'recompute').device_bytes[event]
                     for block in blocks) for event in range(len(events))]
    fixed = tuple(max(0, actual - selected) for actual, selected in zip(observed, reference, strict=True))
    return MemoryProfile(events, fixed, tuple(blocks), {
        'executor': 'FastVideo WanModel / FSDP2', 'model': args.model,
        'input_shape': list(step.noisy.permute(0, 2, 1, 3, 4).shape),
        'text_tokens': step.batch.conditional_dict['encoder_hidden_states'].shape[1],
        'training_sample': args.training_sample, 'block_measurements': costs,
        'calibration_peaks': observed, 'master_dtype': 'float32', 'autocast_dtype': 'bfloat16',
        'optimizer': 'AdamW foreach=False', 'world_size': 1,
        'event_accounting': 'continuous pre-hook-to-pre-hook intervals',
        'calibration_references': [{'policy': 'recompute', 'calibration_peaks': observed}]
    })


def refine_memory_profile(args: argparse.Namespace, model: WanModel, step: TrainingStep,
                          policies: dict[str, str]) -> MemoryProfile:
    profile = MemoryProfile.load(args.reference_profile)
    shape = list(step.noisy.permute(0, 2, 1, 3, 4).shape)
    if profile.metadata['input_shape'] != shape:
        raise ValueError('Reference profiling must use the same input shape')
    events, observed = observe_events(args, model, step)
    if profile.events != events:
        raise ValueError('Reference profiling must use the same execution events')
    selected = [sum(next(choice for choice in block.choices if choice.policy == policies[block.block]).device_bytes[event]
                    for block in profile.blocks) for event in range(len(events))]
    fixed = tuple(max(previous, actual - cost) for previous, actual, cost in
                  zip(profile.fixed_device_bytes, observed, selected, strict=True))
    references = [*profile.metadata['calibration_references'], {'policies': policies, 'calibration_peaks': observed}]
    return MemoryProfile(events, fixed, profile.blocks, {
        **profile.metadata, 'calibration_references': references,
        'reference_profile': args.reference_profile,
        'memory_model': 'per-event envelope of measured reference residuals'
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['profile', 'benchmark'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--training-sample', required=True)
    parser.add_argument('--policy', default='recompute')
    parser.add_argument('--save-count', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=2)
    parser.add_argument('--profile')
    parser.add_argument('--reference-profile')
    parser.add_argument('--budget-gib', type=float)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--save-model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.npu.set_device(local_rank)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank))
    maybe_init_distributed_environment_and_model_parallel(
        1, 1, distributed_init_method='env://' if world > 1 else 'tcp://127.0.0.1:29885')
    config = TrainingConfig(model_path=args.model,
                            dit_precision='fp32',
                            distributed=DistributedConfig(num_gpus=world, hsdp_replicate_dim=1, hsdp_shard_dim=world),
                            pipeline_config=WanT2V480PConfig())
    count = json.loads((Path(args.model) / 'transformer/config.json').read_text())['num_layers']
    planning_ms = 0.0
    if args.profile:
        started = time.perf_counter()
        plan = plan_activation_memory(MemoryProfile.load(args.profile), int(args.budget_gib * 2**30))
        planning_ms = (time.perf_counter() - started) * 1000
        policies = plan.policies['student']
        if rank == 0:
            plan.save(Path(args.output).with_suffix('.plan.json'))
    else:
        holder = torch.nn.Module()
        holder.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(count)])
        policies = policy_for(holder,
                              'recompute' if args.command == 'profile' and not args.reference_profile else args.policy,
                              checkpoint_every=args.checkpoint_every,
                              save_count=args.save_count)
    model = WanModel(init_from=args.model,
                     training_config=config,
                     trainable=True,
                     attention_backend='TORCH_SDPA',
                     activation_memory_plan=policies)
    model.transformer.train()
    sample = torch.load(args.training_sample, map_location='cpu', weights_only=True)
    step = TrainingStep(model, sample, args.seed + rank)
    if args.command == 'profile':
        started = time.perf_counter()
        profile = (refine_memory_profile(args, model, step, policies)
                   if args.reference_profile else calibrate(args, model, step))
        elapsed_ms = (time.perf_counter() - started) * 1000
        profile.metadata['incremental_calibration_ms'] = elapsed_ms
        profile.metadata['calibration_ms'] = elapsed_ms + (profile.metadata['calibration_ms']
                                                          if args.reference_profile else 0)
        profile.metadata.update(world_size=world, rank=rank)
        if world > 1:
            rank_profiles = [profile] * world
            dist.all_gather_object(rank_profiles, profile)
            profile = combine_rank_profiles(rank_profiles)
        if rank == 0:
            profile.save(args.output)
            print(json.dumps({'profile': args.output, 'blocks': len(profile.blocks), 'events': len(profile.events)}))
    else:
        for _ in range(args.warmup):
            step()
        records = []
        for _ in range(args.steps):
            if world > 1:
                dist.barrier()
            sync()
            torch.npu.reset_peak_memory_stats()
            started = time.perf_counter()
            loss = step()
            sync()
            records.append({
                'step_ms': (time.perf_counter() - started) * 1000,
                'loss': loss,
                'peak_allocated_bytes': torch.npu.max_memory_allocated(),
                'peak_reserved_bytes': torch.npu.max_memory_reserved()
            })
        exported = None
        if args.save_model:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_model_state_dict
            from fastvideo.train.utils.inference_checkpoint import export_inference_checkpoint_from_dcp

            checkpoint = Path(args.save_model) / 'dcp'
            dcp.save({'roles.student.transformer': get_model_state_dict(model.transformer)},
                     checkpoint_id=str(checkpoint))
            if rank == 0:
                exported = export_inference_checkpoint_from_dcp(dcp_checkpoint=checkpoint,
                                                                output_dir=args.save_model,
                                                                step=args.warmup + args.steps,
                                                                module=model.transformer,
                                                                base_model_dir=args.model,
                                                                dtype=torch.bfloat16,
                                                                raw_config={
                                                                    'training_sample': args.training_sample,
                                                                    'policy': policies,
                                                                    'master_dtype': 'float32',
                                                                    'optimizer': 'AdamW foreach=False',
                                                                    'learning_rate': 1e-5,
                                                                    'world_size': world,
                                                                    'seed': args.seed,
                                                                    'activation_memory_profile': args.profile,
                                                                    'activation_memory_budget_gib': args.budget_gib
                                                                })
        rank_measurements = [records]
        if world > 1:
            rank_measurements = [[] for _ in range(world)]
            dist.all_gather_object(rank_measurements, records)
            records = [{
                key: max(rows[index][key] for rows in rank_measurements)
                for key in ('step_ms', 'peak_allocated_bytes', 'peak_reserved_bytes')
            } | {
                'loss': rank_measurements[0][index]['loss']
            } for index in range(args.steps)]
        if rank == 0:
            Path(args.output).write_text(
                json.dumps(
                    {
                        'executor': 'FastVideo WanModel / FSDP2',
                        'exported_model': str(exported) if exported else None,
                        'model': args.model,
                        'training_sample': args.training_sample,
                        'measurements': records,
                        'rank_measurements': rank_measurements,
                        'world_size': world,
                        'batch_per_rank': 1,
                        'policies': policies,
                        'planning_ms': planning_ms,
                        'warmup_steps': args.warmup,
                        'torch': torch.__version__,
                        'torch_npu': torch_npu.__version__,
                    },
                    indent=2) + '\n')
    cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
