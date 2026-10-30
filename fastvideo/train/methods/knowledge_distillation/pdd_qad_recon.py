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


def theseus_widths(num_blocks: int) -> tuple[int, ...]:
    """Binary-merge segment widths ``1, 2, 4, ...`` below ``num_blocks``, then the whole stack."""
    widths, width = [], 1
    while width < num_blocks:
        widths.append(width)
        width *= 2
    return (*widths, num_blocks)


def theseus_schedule(total_steps: int, widths: tuple[int, ...], gamma: float) -> list[int]:
    """Segment width of each warm-up step: level ``k`` gets ``~gamma**k`` of ``total_steps`` (at least one each).

    Short-to-long, as JustQuant's Theseus QAD (arXiv:2609.33601): narrow, teacher-forced targets first, then
    progressively longer student paths, ending on the whole stack.
    """
    if total_steps < len(widths):
        raise ValueError(f"{total_steps} warm-up steps cannot cover {len(widths)} segment widths {widths}")
    weights = [gamma**k for k in range(len(widths))]
    raw = [total_steps * w / sum(weights) for w in weights]
    steps = [max(1, int(r)) for r in raw]
    # Hand out what flooring lost to the levels with the largest remainders (the long ones on ties).
    for index in sorted(range(len(widths)), key=lambda i: (raw[i] - int(raw[i]), i), reverse=True):
        if sum(steps) >= total_steps:
            break
        steps[index] += 1
    while sum(steps) > total_steps:  # the at-least-one floor overshot: shave the longest levels
        index = max(i for i in range(len(steps)) if steps[i] > 1)
        steps[index] -= 1
    return [width for width, count in zip(widths, steps, strict=True) for _ in range(count)]


class SegmentReconstruction:
    """Theseus warm-up: segments of ``width`` consecutive blocks, each from the teacher's input, during one forward.

    When teacher block ``g + width - 1`` finishes, student blocks ``g .. g + width - 1`` run chained from the
    teacher's (stop-gradient) input to block ``g`` with the teacher's block arguments, match the teacher's output of
    the segment's last block, and are backpropagated at once. Segments are teacher-forced at their boundaries, so a
    width-1 segment is block-wise distillation and the full width is the whole stack. ``scale`` multiplies each
    segment's loss before its backward. Use around the teacher's ``forward_rung`` with grad disabled.
    """

    def __init__(self,
                 teacher: torch.nn.Module,
                 student: torch.nn.Module,
                 width: int,
                 scale: float,
                 criterion: GroupedRelError | None = None) -> None:
        ensure_fsdp_root_initialized(teacher)
        ensure_fsdp_root_initialized(student)
        self.teacher_blocks = list(teacher.transformer_blocks)
        self.student_blocks = list(student.transformer_blocks)
        if len(self.teacher_blocks) != len(self.student_blocks):
            raise ValueError("teacher and student block counts differ")
        if not 1 <= width <= len(self.teacher_blocks):
            raise ValueError(f"segment width {width} outside 1..{len(self.teacher_blocks)}")
        self.width = width
        self.scale = scale
        self.criterion = criterion
        self.rel: list[torch.Tensor] = []
        self.group_rel: list[torch.Tensor] = []
        self._start: torch.Tensor | None = None
        self._args: list[tuple[Any, ...]] = []
        self._handles: list[Any] = []

    @property
    def num_segments(self) -> int:
        return -(-len(self.teacher_blocks) // self.width)

    def __enter__(self) -> SegmentReconstruction:
        for index, block in enumerate(self.teacher_blocks):
            self._handles.append(block.register_forward_hook(self._after_block(index)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._start, self._args = None, []

    def _after_block(self, index: int):

        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if index % self.width == 0:
                self._start, self._args = args[0], []
            self._args.append(args[1:])
            if index % self.width == self.width - 1 or index == len(self.teacher_blocks) - 1:
                self._student_segment(index - len(self._args) + 1, output)
                self._start, self._args = None, []

        return hook

    def _student_segment(self, first: int, teacher_out: torch.Tensor) -> None:
        assert self._start is not None
        with torch.set_grad_enabled(True):
            hidden = self._start.detach()
            for offset, block_args in enumerate(self._args):
                hidden = self.student_blocks[first + offset](hidden, *block_args)
            if self.criterion is None:
                target = teacher_out.float()
                loss = (hidden.float() - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-20)
                rel = loss.detach()
            else:
                loss, per_group = self.criterion(hidden, teacher_out)
                self.group_rel.append(per_group)
                weights = self.criterion.weights * (~per_group.isnan())
                rel = (weights * per_group.nan_to_num()).sum() / weights.sum().clamp_min(1e-30)
            (loss * self.scale).backward()
        self.rel.append(rel)

    def loss(self) -> torch.Tensor:
        """Mean (group-weighted) relative error over this forward's segments, sequence-parallel-global, detached."""
        return torch.stack(self.rel).mean()

    def metrics(self, prefix: str) -> dict[str, torch.Tensor]:
        out = {f"{prefix}/rel": self.loss(), f"{prefix}/last_segment_rel": self.rel[-1]}
        if self.group_rel:
            per_group = torch.stack(self.group_rel).nanmean(dim=0)
            out.update({f"{prefix}/{name}_rel": per_group[i] for i, name in enumerate(GROUPS)})
        return out


__all__ = [
    "BlockReconstruction", "GROUPS", "GroupedRelError", "SegmentReconstruction", "local_row_groups",
    "packed_row_groups", "theseus_schedule", "theseus_widths"
]
