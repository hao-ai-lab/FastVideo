# SPDX-License-Identifier: Apache-2.0
"""Layer-wise reconstruction for the FastH3 OmniRef QAD student, streamed per block.

Supervised units R, per transformer block: the attention sub-layer (q/k/v/gate/out NVFP4 projections, QK
RMSNorm, RoPE and the sparse FP4 VSA-128 attention, as deployed) and the FFN sub-layer (NVFP4 fc_in/fc_out).
Each student unit receives the frozen teacher's input to that unit (stop-gradient) and matches the teacher
unit's output, so quantization error is corrected locally and cannot accumulate into a drift away from the
teacher.

Locality makes this cheap: inside one teacher forward, right after teacher block ``b`` runs, student block
``b`` runs on the captured inputs and is backpropagated at once. There is no end-to-end student forward or
backward, no activation recompute, and memory holds one block's activations.

Attention is sparse (VSA-128, the teacher's trained selection), not dense as in the paper's warmup: our
teacher was trained sparse and the student is deployed sparse, so dense attention would supervise a
configuration that never runs.

The loss per unit is the relative squared error ``||s - t||^2 / ||t||^2`` (late FFN outputs reach ~1e6, so an
unnormalized sum would only train those), averaged over R. With a ``GroupedRelError`` it is computed per row group
(target video, target audio, conditioning) and weighted: over the whole packed sequence audio is ~1.4% of the rows
and massive text tokens dominate ``||t||^2``, so an ungrouped ratio barely trains audio.
"""
from __future__ import annotations

from typing import Any

import torch


def ensure_fsdp_root_initialized(module: torch.nn.Module) -> None:
    """Make ``module`` its FSDP2 tree's root before any of its sharded blocks runs on its own.

    FSDP2 picks the root lazily, at the first forward of any sharded module. If a block runs first (as in
    reconstruction, which calls student blocks directly), that block claims the root and the model's next full
    forward fails with "FSDP state has already been lazily initialized". Idempotent; a no-op without FSDP2.
    """
    get_state = getattr(module, "_get_fsdp_state", None)
    if get_state is not None:
        get_state()._lazy_init()


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


GROUPS = ("video", "audio", "condition")  # target video rows, target audio rows, text + reference rows


def packed_row_groups(layout: Any, sequence_length: int) -> torch.Tensor:
    """Group id per packed row: 0 target video, 1 target audio, 2 conditioning (text, references)."""
    groups = torch.full((sequence_length, ), 2, dtype=torch.long)
    groups[layout.video_indices[layout.num_condition_video_rows:]] = 0
    groups[layout.audio_indices[layout.num_condition_audio_rows:]] = 1
    return groups


def local_row_groups(groups: torch.Tensor, device: torch.device) -> torch.Tensor:
    """This sequence-parallel rank's rows of ``groups`` as the transformer shards them; padding is -1."""
    from fastvideo.distributed import get_sp_world_size, model_parallel_is_initialized
    from fastvideo.distributed.communication_op import sequence_model_parallel_shard

    shifted = (groups + 1).to(device)
    if model_parallel_is_initialized() and get_sp_world_size() > 1:
        shifted, _ = sequence_model_parallel_shard(shifted, dim=0)  # pads with 0 -> -1 after the shift
    return shifted - 1


class GroupedRelError:
    """Weighted mean over row groups of ``||s - t||^2 / ||t||^2`` with sequence-parallel-global sums.

    Each rank backpropagates ``sp_size * w_g * num_local / den_global``. Summed over ranks, ``w_g * num_local /
    den_global`` is the global ratio, so a group living on one rank (e.g. target audio) is weighted exactly as
    if unsharded. The ``sp_size`` factor matches the x0 output loss: FSDP averages gradients over every rank,
    SP ranks included, and the output loss is identical on every SP rank, where the transformer's output
    all-gather backward (a SUM reduce-scatter) gives each rank ``sp_size`` times its share of the gradient.
    Reconstruction has no gather, so without the factor the averaged gradient would be ``1 / sp_size`` of the
    global ratio's. Groups absent from the sequence are dropped from the weight normalization.
    """

    def __init__(self, groups: torch.Tensor, weights: dict[str, float]) -> None:
        self.groups = groups
        self.weights = torch.tensor([float(weights.get(name, 0.0)) for name in GROUPS], device=groups.device)

    def __call__(self, student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(local loss to backprop, global per-group relative error [3], detached)``.

        The local loss is this rank's ``sp_size``-scaled share (see the class docstring); the detached global
        per-group errors are what to log.
        """
        from fastvideo.distributed import get_sp_group, get_sp_world_size, model_parallel_is_initialized

        teacher = teacher.float().reshape(-1, teacher.shape[-1])
        err = (student.float().reshape(-1, teacher.shape[-1]) - teacher).pow(2).sum(-1)
        ref = teacher.pow(2).sum(-1)
        one_hot = (self.groups[:, None] == torch.arange(len(GROUPS), device=err.device)[None, :]).float()
        num = err @ one_hot  # [3], differentiable
        totals = torch.stack((num.detach(), ref @ one_hot))  # [2, 3]
        sp_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_size > 1:
            totals = get_sp_group().all_reduce(totals)
        present = totals[1] > 0
        den = totals[1].clamp_min(1e-30)
        weights = torch.where(present, self.weights, torch.zeros_like(self.weights))
        norm = weights.sum().clamp_min(1e-30)
        loss = sp_size * (weights * num / den).sum() / norm
        return loss, torch.where(present, totals[0] / den, torch.full_like(den, float("nan")))


class _Replace:
    """Forward pre-hook replacing a sub-layer's first input with the teacher's, plus an output capture."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def pre(self, module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (self.value, *args[1:])

    def post(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.output = output


class _Capture:

    def __init__(self) -> None:
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def __call__(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.input, self.output = args[0], output


class BlockReconstruction:
    """Stream reconstruction losses for every block during one teacher forward.

    ``scale`` multiplies each block's loss before its backward (e.g. ``1 / (units x rungs x accum)``).
    Use as a context manager around the teacher's ``forward_rung``, with grad disabled for the teacher.
    """

    def __init__(self,
                 teacher: torch.nn.Module,
                 student: torch.nn.Module,
                 scale: float,
                 criterion: GroupedRelError | None = None,
                 measure_only: bool = False) -> None:
        ensure_fsdp_root_initialized(teacher)
        ensure_fsdp_root_initialized(student)
        self.teacher_blocks = list(teacher.transformer_blocks)
        self.student_blocks = list(student.transformer_blocks)
        if len(self.teacher_blocks) != len(self.student_blocks):
            raise ValueError("teacher and student block counts differ")
        self.scale = scale
        self.criterion = criterion
        self.measure_only = measure_only
        self.attn_rel: list[torch.Tensor] = []
        self.ff_rel: list[torch.Tensor] = []
        self.group_rel: list[torch.Tensor] = []  # [3] per unit, global, detached
        self._handles: list[Any] = []

    def _weighted(self, rel: torch.Tensor) -> torch.Tensor:
        """Group-weighted mean of per-group relative errors ``[..., 3]`` (NaN = absent group, dropped)."""
        assert self.criterion is not None
        weights = self.criterion.weights.expand_as(rel) * (~rel.isnan())
        return (weights * rel.nan_to_num()).sum(-1) / weights.sum(-1).clamp_min(1e-30)

    def _unit_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(loss to backprop, sequence-parallel-global relative error to log, detached)``."""
        if self.criterion is None:
            teacher_f = teacher.float()
            loss = (student.float() - teacher_f).pow(2).mean() / teacher_f.pow(2).mean().clamp_min(1e-20)
            return loss, loss.detach()
        loss, per_group = self.criterion(student, teacher)
        self.group_rel.append(per_group)
        return loss, self._weighted(per_group)

    def __enter__(self) -> BlockReconstruction:
        for index, block in enumerate(self.teacher_blocks):
            inner = _unwrap(block)
            attn, ff = _Capture(), _Capture()
            self._handles.append(inner.attn.register_forward_hook(attn))
            self._handles.append(inner.ff.register_forward_hook(ff))
            self._handles.append(block.register_forward_hook(self._after_block(index, attn, ff)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _after_block(self, index: int, attn: _Capture, ff: _Capture):

        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            self._student_block(index, args, attn, ff)
            attn.input = attn.output = ff.input = ff.output = None

        return hook

    def _student_block(self, index: int, block_args: tuple[Any, ...], attn: _Capture, ff: _Capture) -> None:
        block = self.student_blocks[index]
        inner = _unwrap(block)
        replace_attn, replace_ff = _Replace(), _Replace()
        replace_attn.value, replace_ff.value = attn.input, ff.input
        handles = [
            inner.attn.register_forward_pre_hook(replace_attn.pre),
            inner.attn.register_forward_hook(replace_attn.post),
            inner.ff.register_forward_pre_hook(replace_ff.pre),
            inner.ff.register_forward_hook(replace_ff.post)
        ]
        try:
            with torch.set_grad_enabled(not self.measure_only):
                # The block's own hidden input only feeds norm1 and the residual, whose results are unused; the
                # teacher attention input has the right shape. The call goes through the block, and the block
                # input and output join the graph (zero weight), so FSDP's pre-/post-backward hooks re-gather the
                # block's weights for backward and reduce-scatter its gradients.
                hidden = attn.input.detach().requires_grad_(not self.measure_only)
                block_out = block(hidden, *block_args[1:])
                attn_loss, attn_rel = self._unit_loss(replace_attn.output, attn.output)
                ff_loss, ff_rel = self._unit_loss(replace_ff.output, ff.output)
                if not self.measure_only:
                    ((attn_loss + ff_loss) * self.scale + 0.0 * block_out.float().sum()).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.attn_rel.append(attn_rel)
        self.ff_rel.append(ff_rel)

    def loss(self) -> torch.Tensor:
        """Mean (group-weighted) relative error over all units of this forward, sequence-parallel-global, detached."""
        if self.criterion is not None and self.group_rel:
            return self._weighted(torch.stack(self.group_rel)).mean()  # [units, 3], NaN for absent groups
        return (torch.stack(self.attn_rel).sum() + torch.stack(self.ff_rel).sum()) / (2 * len(self.attn_rel))

    def metrics(self, prefix: str) -> dict[str, torch.Tensor]:
        attn, ff = torch.stack(self.attn_rel), torch.stack(self.ff_rel)
        third = max(1, len(attn) // 3)
        groups = {}
        if self.group_rel:
            per_group = torch.stack(self.group_rel).nanmean(dim=0)
            groups = {f"{prefix}/{name}_rel": per_group[i] for i, name in enumerate(GROUPS)}
        return {
            **groups, f"{prefix}/attn_rel": attn.mean(),
            f"{prefix}/ff_rel": ff.mean(),
            f"{prefix}/attn_rel_first_third": attn[:third].mean(),
            f"{prefix}/attn_rel_last_third": attn[-third:].mean(),
            f"{prefix}/ff_rel_last_third": ff[-third:].mean()
        }


__all__ = ["BlockReconstruction", "GROUPS", "GroupedRelError", "local_row_groups", "packed_row_groups"]
