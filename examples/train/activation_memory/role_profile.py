# SPDX-License-Identifier: Apache-2.0
"""Measure role lifetimes in FastVideo's production data-latent DMD2 method."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from fastvideo.train.utils.activation_memory import BlockCosts, MemoryProfile, PolicyCost


def calibrate_dmd2(method: Any, step: Callable[..., Any], template: MemoryProfile, *,
                   warmup: int, repetitions: int, training_sample: str) -> MemoryProfile:
    costs = template.metadata['block_measurements']
    count = len(method.student.transformer.blocks)
    events = tuple([f'student.forward.{index}' for index in range(count)] +
                   ['score.critic', 'score.teacher.conditional', 'score.teacher.unconditional', 'rollout.student'] +
                   [f'critic.forward.{index}'
                    for index in range(count)] + [f'student.backward.{index}' for index in reversed(range(count))] +
                   [f'critic.backward.{index}'
                    for index in reversed(range(count))] + ['clipping', 'optimizer.critic', 'optimizer.student'])
    indices = {event: index for index, event in enumerate(events)}
    blocks = []
    for role in ('student', 'critic'):
        for index in range(count):
            start, end = indices[f'{role}.forward.{index}'], indices[f'{role}.backward.{index}']
            choices = []
            for policy, cost in costs.items():
                device, host = [0] * len(events), [0] * len(events)
                for event in range(start, end + 1):
                    device[event] = cost['saved_bytes']
                    if policy == 'offload':
                        host[event] = cost['host_bytes']
                device[start] = max(device[start], cost['forward_peak'])
                device[end] = max(device[end], cost['backward_peak'])
                choices.append(PolicyCost(policy, cost['time_ms'], tuple(device), tuple(host)))
            blocks.append(BlockCosts(role, f'blocks.{index}', tuple(choices)))
    for _ in range(warmup):
        step()
    observed = [0] * len(events)
    phase: str | None = None
    mode: str | None = None

    def transition(name: str) -> None:
        nonlocal phase
        torch.npu.synchronize()
        if phase is not None and phase in indices:
            observed[indices[phase]] = max(observed[indices[phase]], torch.npu.max_memory_allocated())
        phase = name
        torch.npu.reset_peak_memory_stats()

    originals, handles = [], []

    def instrument_predict(role: str, original: Callable[..., Any]) -> Callable[..., Any]:

        def predict(*args: Any, **kwargs: Any) -> Any:
            nonlocal mode
            mode = f'{role}.forward' if torch.is_grad_enabled() else 'score'
            if torch.is_grad_enabled():
                transition(f'{role}.forward.0')
            elif role == 'student':
                transition('rollout.student')
            elif role == 'critic':
                transition('score.critic')
            else:
                condition = 'conditional' if kwargs['conditional'] else 'unconditional'
                transition(f'score.teacher.{condition}')
            return original(*args, **kwargs)

        return predict

    def instrument_backward(role: str, original: Callable[..., Any]) -> Callable[..., Any]:

        def backward(*args: Any, **kwargs: Any) -> Any:
            nonlocal mode
            mode = f'{role}.backward'
            transition(f'{role}.backward.{count - 1}')
            return original(*args, **kwargs)

        return backward

    def before(role: str, direction: str, index: int) -> Callable[..., None]:

        def begin(module: Any, *unused: Any) -> None:
            if mode == f'{role}.{direction}':
                transition(f'{role}.{direction}.{index}')

        return begin

    for role in ('student', 'critic', 'teacher'):
        model = getattr(method, role)
        originals.append((model, model.predict_noise, model.backward))
        model.predict_noise = instrument_predict(role, model.predict_noise)
        if role != 'teacher':
            model.backward = instrument_backward(role, model.backward)
            for index, block in enumerate(model.transformer.blocks):
                handles.extend([
                    block.register_forward_pre_hook(before(role, 'forward', index)),
                    block.register_full_backward_pre_hook(before(role, 'backward', index))
                ])
    for _ in range(repetitions):
        step(transition)
    transition('complete')
    for handle in handles:
        handle.remove()
    for model, predict, backward in originals:
        model.predict_noise, model.backward = predict, backward
    reference = [
        sum(next(choice for choice in block.choices if choice.policy == 'recompute').device_bytes[event]
            for block in blocks) for event in range(len(events))
    ]
    fixed = tuple(max(0, actual - selected) for actual, selected in zip(observed, reference, strict=True))
    return MemoryProfile(
        events, fixed, tuple(blocks), {
            **template.metadata,
            'executor': 'FastVideo DMD2Method / WanModel / FSDP2',
            'method': 'DMD2 data_latent',
            'training_sample': training_sample,
            'optimizer': 'AdamW with FastVideo optimizer configuration',
            'calibration_peaks': observed,
            'event_accounting': 'continuous pre-hook intervals including FSDP transitions',
            'teacher_trainable': False,
        })
