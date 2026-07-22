#!/usr/bin/env python3
"""Scratch gate: let fused AdamW apply the already-computed clip scale."""

from __future__ import annotations

import json
import sys
from typing import Any

import torch
import torch.distributed as dist

import benchmark_fastvideo_train_pack_d016 as benchmark
import fastvideo.train.callbacks.grad_clip as grad_clip_module
from fastvideo.train.callbacks.grad_clip import GradNormClipCallback
from fastvideo.train.methods.base import TrainingMethod
import fastvideo.training.training_utils as training_utils


_COUNTS = {
    "fused_clips": 0,
    "fallback_clips": 0,
    "scaled_optimizer_steps": 0,
    "cleared_scales": 0,
    "ordinary_cuda_scales": 0,
}
_SCALED_OPTIMIZERS: dict[int, torch.optim.Optimizer] = {}
_VALIDATED_PAIRS: set[tuple[int, int]] = set()
_ACTIVE_OPTIMIZER: torch.optim.Optimizer | None = None


def _uses_fused_adamw(optimizer: torch.optim.Optimizer | None) -> bool:
    return (
        isinstance(optimizer, torch.optim.AdamW)
        and bool(optimizer.param_groups)
        and all(group.get("fused") is True for group in optimizer.param_groups)
        and getattr(optimizer, "grad_scale", None) is None
    )


def _validate_pair(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    key = (id(module), id(optimizer))
    if key in _VALIDATED_PAIRS:
        return
    module_params = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    optimizer_params = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    if module_params != optimizer_params:
        raise RuntimeError(
            "fused clip target and optimizer parameter sets differ: "
            f"target={len(module_params)} optimizer={len(optimizer_params)}"
        )
    _VALIDATED_PAIRS.add(key)


def _install_fused_clip() -> None:
    original_step = TrainingMethod.optimizers_schedulers_step
    original_scale_grads = training_utils._clip_grads_with_norm_

    def _defer_scale(
        parameters: torch.Tensor | list[torch.Tensor],
        max_norm: float,
        total_norm: torch.Tensor,
        foreach: bool | None = None,
    ) -> None:
        del parameters, foreach
        if _ACTIVE_OPTIMIZER is None:
            raise RuntimeError("missing active optimizer while computing a target norm")
        if isinstance(total_norm, torch.distributed.tensor.DTensor):
            raise RuntimeError("clip norm remained a DTensor instead of becoming a full tensor")
        if total_norm.ndim != 0 or total_norm.device.type != "cuda":
            raise RuntimeError(f"expected an ordinary CUDA scalar norm, got {total_norm.shape} on {total_norm.device}")
        clip_coef = float(max_norm) / (total_norm + 1e-6)
        grad_scale = torch.clamp(clip_coef, max=1.0).reciprocal()
        if isinstance(grad_scale, torch.distributed.tensor.DTensor) or grad_scale.ndim != 0 or grad_scale.device.type != "cuda":
            raise RuntimeError(f"expected an ordinary CUDA scalar grad_scale, got {type(grad_scale)}")
        _ACTIVE_OPTIMIZER.grad_scale = grad_scale
        _COUNTS["fused_clips"] += 1
        _COUNTS["ordinary_cuda_scales"] += 1

    def _before_step(self: GradNormClipCallback, method: TrainingMethod, iteration: int = 0) -> None:
        global _ACTIVE_OPTIMIZER
        if self._pending_grad_norms:
            raise RuntimeError("Previous gradient norms were not logged after the optimizer step")
        if self._max_grad_norm <= 0.0:
            return

        active = {id(optimizer): optimizer for optimizer in method.get_optimizers(iteration)}
        optimizer_by_role = method._optimizer_dict
        tracker = getattr(method, "tracker", None)
        for name, module in method.get_grad_clip_targets(iteration).items():
            optimizer = optimizer_by_role.get(name)
            if optimizer is not None and id(optimizer) in active and _uses_fused_adamw(optimizer):
                _validate_pair(module, optimizer)
                _SCALED_OPTIMIZERS[id(optimizer)] = optimizer
                _ACTIVE_OPTIMIZER = optimizer
                training_utils._clip_grads_with_norm_ = _defer_scale
                try:
                    grad_norm = grad_clip_module.clip_grad_norm_if_needed(module, self._max_grad_norm)
                finally:
                    training_utils._clip_grads_with_norm_ = original_scale_grads
                    _ACTIVE_OPTIMIZER = None
                if getattr(optimizer, "grad_scale", None) is None:
                    raise RuntimeError("fused clip did not install optimizer.grad_scale")
            else:
                _COUNTS["fallback_clips"] += 1
                grad_norm = grad_clip_module.clip_grad_norm_if_needed(module, self._max_grad_norm)

            if self._log_grad_norms and tracker is not None and grad_norm is not None:
                self._pending_grad_norms.append((tracker, name, grad_norm, iteration))

    def _step(self: TrainingMethod, iteration: int) -> None:
        scaled = dict(_SCALED_OPTIMIZERS)
        try:
            original_step(self, iteration)
            _COUNTS["scaled_optimizer_steps"] += len(scaled)
        finally:
            for optimizer in scaled.values():
                if hasattr(optimizer, "grad_scale"):
                    del optimizer.grad_scale
                    _COUNTS["cleared_scales"] += 1
            _SCALED_OPTIMIZERS.clear()

    GradNormClipCallback.on_before_optimizer_step = _before_step
    TrainingMethod.optimizers_schedulers_step = _step


def _assert_tiny_exact_parity() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("tiny fused AdamW parity requires CUDA")
    device = torch.device("cuda")
    control = [torch.nn.Parameter(torch.tensor([1.25], device=device)),
               torch.nn.Parameter(torch.tensor([-0.75], device=device))]
    candidate = [torch.nn.Parameter(parameter.detach().clone()) for parameter in control]
    kwargs = dict(lr=3e-4, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.01, fused=True)
    control_optimizer = torch.optim.AdamW(control, **kwargs)
    candidate_optimizer = torch.optim.AdamW(candidate, **kwargs)

    gradients = ((3.0, 4.0), (-4.0, 3.0), (3.0, -4.0), (-3.0, -4.0))
    candidate_grad_contract: str | None = None
    for step, values in enumerate(gradients, 1):
        for parameter, value in zip(control, values, strict=True):
            parameter.grad = torch.tensor([value], device=device)
        for parameter, value in zip(candidate, values, strict=True):
            parameter.grad = torch.tensor([value], device=device)

        control_norm = training_utils._get_total_norm([parameter.grad for parameter in control])
        candidate_norm = training_utils._get_total_norm([parameter.grad for parameter in candidate])
        torch.testing.assert_close(control_norm, candidate_norm, rtol=0, atol=0)
        # Norm is exactly five. Exercise the recipe's max_norm=1 clipping and
        # the unclipped path on alternating steps.
        max_norm = 1.0 if step % 2 else 10.0
        training_utils._clip_grads_with_norm_(control, max_norm, control_norm)
        expected_effective_grads = [parameter.grad.detach().clone() for parameter in control]
        raw_candidate_grads = [parameter.grad.detach().clone() for parameter in candidate]
        clip_coef = float(max_norm) / (candidate_norm + 1e-6)
        candidate_optimizer.grad_scale = torch.clamp(clip_coef, max=1.0).reciprocal()

        control_optimizer.step()
        candidate_optimizer.step()
        del candidate_optimizer.grad_scale

        for index, (control_parameter, candidate_parameter) in enumerate(zip(control, candidate, strict=True)):
            torch.testing.assert_close(control_parameter.grad, expected_effective_grads[index], rtol=0, atol=0)
            torch.testing.assert_close(control_parameter, candidate_parameter, rtol=0, atol=0)
            control_state = control_optimizer.state[control_parameter]
            candidate_state = candidate_optimizer.state[candidate_parameter]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(control_state[key], candidate_state[key], rtol=0, atol=0)
            assert control_parameter.dtype == torch.float32
            assert control_state["exp_avg"].dtype == torch.float32
            assert control_state["exp_avg_sq"].dtype == torch.float32

        if step % 2:
            wrote_scaled = all(torch.equal(parameter.grad, expected)
                               for parameter, expected in zip(candidate, expected_effective_grads, strict=True))
            preserved_raw = all(torch.equal(parameter.grad, raw)
                                for parameter, raw in zip(candidate, raw_candidate_grads, strict=True))
            if wrote_scaled == preserved_raw:
                raise RuntimeError("candidate grads were neither uniquely scaled nor uniquely preserved")
            observed = "scaled_writeback" if wrote_scaled else "raw_preserved"
            if candidate_grad_contract is not None and candidate_grad_contract != observed:
                raise RuntimeError(f"candidate gradient contract changed: {candidate_grad_contract} -> {observed}")
            candidate_grad_contract = observed

    torch.cuda.synchronize()
    print("FUSED_CLIP_PARITY " + json.dumps({
        "device": torch.cuda.get_device_name(),
        "steps": len(gradients),
        "params_moments_steps_bit_exact": True,
        "registered_parameter_dtype": "torch.float32",
        "moment_dtype": "torch.float32",
        "candidate_grad_contract": candidate_grad_contract,
    }, sort_keys=True), flush=True)


def _report_counts() -> None:
    if not dist.is_initialized():
        return
    keys = list(_COUNTS)
    counts = torch.tensor([_COUNTS[key] for key in keys], device="cuda", dtype=torch.int64)
    dist.all_reduce(counts)
    world_size = dist.get_world_size()
    totals = dict(zip(keys, counts.cpu().tolist(), strict=True))
    expected = benchmark.EXPECTED_STEPS * world_size
    if totals != {
        "fused_clips": expected,
        "fallback_clips": 0,
        "scaled_optimizer_steps": expected,
        "cleared_scales": expected,
        "ordinary_cuda_scales": expected,
    }:
        raise RuntimeError(f"unexpected fused clip counts: {totals}")
    if dist.get_rank() == 0:
        print("FUSED_CLIP_COUNTS " + json.dumps({
            "world_size": world_size,
            "per_rank_steps": benchmark.EXPECTED_STEPS,
            "totals": totals,
        }, sort_keys=True), flush=True)


def main() -> None:
    if "--parity-only" in sys.argv:
        sys.argv.remove("--parity-only")
        _assert_tiny_exact_parity()
        return
    _install_fused_clip()
    benchmark.main()
    _report_counts()


if __name__ == "__main__":
    main()
