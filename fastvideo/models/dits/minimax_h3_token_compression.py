# SPDX-License-Identifier: Apache-2.0
"""Middle-block spatial token compression for FastH3 (QAD training and inference; hook based).

Blocks ``[start, end)`` run on fewer video tokens. Text and audio rows are untouched; a kept video token keeps its
original RoPE coordinates and AdaLN row; after the last compressed block every video token receives the update of
its nearest kept token: ``H_out = H + expand(F(P H) - P H)`` where ``P`` is the compression operator. With
compression off nothing changes (no hook acts), so the student equals the teacher bit for bit.

Rather than replacing the block list, this installs forward hooks, so module names, FSDP sharding, activation
checkpointing and QAD linears are untouched:

* block ``start`` pre-hook compresses the hidden states and swaps AdaLN rows / RoPE / sequence length;
* every block in the range gets the compressed AdaLN rows / RoPE;
* the inner attention module of each compressed block swaps the VSA metadata for the compressed video grid (a hook
  on the *inner* module, so activation-checkpoint recompute during backward swaps it too);
* block ``end - 1`` post-hook expands and restores.

Single sequence-parallel rank only (the compression indexes the whole packed sequence).
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fastvideo.forward_context import get_forward_context

MODES = ("subsample", "pair_x", "pool2x2")


@dataclass(frozen=True)
class CompressionSpec:
    start: int
    end: int
    mode: str = "subsample"
    # Compress only the first ``first_steps`` of the sampling forwards (None = all); the low-noise steps that set
    # fine detail then run on the full token grid.
    first_steps: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"compression mode must be one of {MODES}, got {self.mode!r}")
        if not 0 <= self.start < self.end:
            raise ValueError(f"need 0 <= start < end, got [{self.start}, {self.end})")

    @classmethod
    def parse(cls, text: str) -> CompressionSpec | None:
        """``"off"``/empty -> None; ``"mode:start:end"`` (e.g. ``subsample:3:41``)."""
        text = text.strip().lower()
        if text in ("", "0", "off", "none"):
            return None
        parts = text.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"token compression must be 'off' or 'mode:start:end[:first_steps]', got {text!r}")
        return cls(int(parts[1]), int(parts[2]), parts[0], int(parts[3]) if len(parts) == 4 else None)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> CompressionSpec | None:
        if not cfg or not cfg.get("enabled", True):
            return None
        first = cfg.get("first_steps")
        return cls(int(cfg["start"]), int(cfg["end"]), str(cfg.get("mode", "subsample")),
                   None if first is None else int(first))


def compressed_grid(mode: str, grid: tuple[int, int, int]) -> tuple[int, int, int]:
    frames, height, width = grid
    if mode == "pair_x":
        return frames, height, (width + 1) // 2
    return frames, (height + 1) // 2, (width + 1) // 2


def token_reduction(mode: str, grid: tuple[int, int, int]) -> float:
    sub = compressed_grid(mode, grid)
    return (grid[0] * grid[1] * grid[2]) / (sub[0] * sub[1] * sub[2])


def _pad_to_even(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    for dim in dims:
        if x.shape[dim] % 2:
            x = torch.cat((x, x.narrow(dim, x.shape[dim] - 1, 1)), dim=dim)
    return x


def grid_ops(mode: str, grid: tuple[int, int, int]) -> tuple[Callable, Callable, torch.Tensor]:
    """``(compress, expand, keep_index)`` for a video token grid ``(T, H, W)``.

    ``compress``: rows ``[B, T*H*W, C]`` -> ``[B, T*h*w, C]``; ``expand``: delta ``[B, T*h*w, C]`` -> ``[B, T*H*W, C]``;
    ``keep_index``: for every compressed token the video row whose RoPE / AdaLN row it keeps (CPU long).
    """
    frames, height, width = grid
    sub = compressed_grid(mode, grid)
    index = torch.arange(frames * height * width).reshape(frames, height, width)

    def as_grid(rows: torch.Tensor) -> torch.Tensor:
        return rows.reshape(rows.shape[0], frames, height, width, rows.shape[-1])

    if mode == "subsample":
        keep = index[:, ::2, ::2].reshape(-1)

        def compress(rows: torch.Tensor) -> torch.Tensor:
            return as_grid(rows)[:, :, ::2, ::2].reshape(rows.shape[0], -1, rows.shape[-1])

    elif mode == "pair_x":
        keep = index[:, :, ::2].reshape(-1)

        def compress(rows: torch.Tensor) -> torch.Tensor:
            g = _pad_to_even(as_grid(rows), (3, ))
            pair = g.float().unflatten(3, (-1, 2)).mean(dim=4)
            return pair.to(rows.dtype).reshape(rows.shape[0], -1, rows.shape[-1])

    else:  # pool2x2
        keep = index[:, ::2, ::2].reshape(-1)

        def compress(rows: torch.Tensor) -> torch.Tensor:
            g = _pad_to_even(as_grid(rows), (2, 3))
            pooled = g.float().unflatten(3, (-1, 2)).unflatten(2, (-1, 2)).mean(dim=(3, 5))
            return pooled.to(rows.dtype).reshape(rows.shape[0], -1, rows.shape[-1])

    repeat_h = 1 if mode == "pair_x" else 2

    def expand(delta: torch.Tensor) -> torch.Tensor:
        d = delta.reshape(delta.shape[0], *sub, delta.shape[-1])
        if repeat_h == 2:
            d = d.repeat_interleave(2, dim=2)[:, :, :height]
        d = d.repeat_interleave(2, dim=3)[:, :, :, :width]
        return d.reshape(delta.shape[0], -1, delta.shape[-1])

    return compress, expand, keep


def _unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


@dataclass
class _Forward:
    """Per-forward state (one transformer call)."""
    full: torch.Tensor | None = None
    compressed_in: torch.Tensor | None = None
    adaln: torch.Tensor | None = None
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None
    seq_len: int = 0
    num_prefix: int = 0


class TokenCompression:
    """Installs the hooks; ``scope(grid, compressed_metadata)`` arms them for one forward (and its backward)."""

    def __init__(self, transformer: nn.Module, spec: CompressionSpec) -> None:
        self.spec = spec
        self.blocks = list(transformer.transformer_blocks)
        if spec.end > len(self.blocks):
            raise ValueError(f"compression range [{spec.start}, {spec.end}) exceeds {len(self.blocks)} blocks")
        self._handles: list[Any] = []
        self._grid: tuple[int, int, int] | None = None
        self._ops: tuple[Callable, Callable, torch.Tensor] | None = None
        self._metadata: Any = None
        self._saved_metadata: Any = None
        self._state = _Forward()
        for index in range(spec.start, spec.end):
            block = self.blocks[index]
            self._handles.append(block.register_forward_pre_hook(self._pre(index)))
            if index == spec.end - 1:
                self._handles.append(block.register_forward_hook(self._post))
            attn = _unwrap(block).attn
            self._handles.append(attn.register_forward_pre_hook(self._attn_pre))
            self._handles.append(attn.register_forward_hook(self._attn_post))

    @property
    def active(self) -> bool:
        return self._grid is not None

    @contextlib.contextmanager
    def scope(self, grid: tuple[int, int, int], compressed_metadata: Any) -> Iterator[None]:
        """Arm the hooks for forwards (and their backward recompute) inside the block."""
        self._grid, self._metadata = grid, compressed_metadata
        self._ops = grid_ops(self.spec.mode, grid)
        try:
            yield
        finally:
            self._grid = self._ops = self._metadata = None
            self._state = _Forward()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- hooks
    def _pre(self, index: int):

        def hook(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not self.active:
                return None
            hidden, temb, adaln, rotary, _ = args
            if index == self.spec.start:
                self._enter(hidden, adaln, rotary)
                hidden = self._state.compressed_in
            state = self._state
            return hidden, temb, state.adaln, state.rotary, state.seq_len

        return hook

    def _enter(self, hidden: torch.Tensor, adaln: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
        assert self._grid is not None and self._ops is not None
        compress, _, keep = self._ops
        num_video = self._grid[0] * self._grid[1] * self._grid[2]
        num_prefix = hidden.shape[1] - num_video
        if num_prefix < 0 or adaln.shape[0] != hidden.shape[1]:
            raise ValueError("token compression needs the whole packed sequence on one rank "
                             f"(hidden {tuple(hidden.shape)}, adaln {tuple(adaln.shape)}, video {num_video})")
        rows = torch.cat((torch.arange(num_prefix), keep + num_prefix)).to(hidden.device)
        self._state = _Forward(full=hidden,
                               compressed_in=torch.cat((hidden[:, :num_prefix], compress(hidden[:, num_prefix:])), dim=1),
                               adaln=adaln.index_select(0, rows),
                               rotary=(rotary[0].index_select(0, rows), rotary[1].index_select(0, rows)),
                               seq_len=int(rows.numel()),
                               num_prefix=num_prefix)

    def _post(self, module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor | None:
        if not self.active:
            return None
        assert self._ops is not None
        state = self._state
        n = state.num_prefix
        delta = self._ops[1](output[:, n:] - state.compressed_in[:, n:])
        restored = torch.cat((output[:, :n], state.full[:, n:] + delta.to(state.full.dtype)), dim=1)
        self._state = _Forward()
        return restored

    def _attn_pre(self, module: nn.Module, args: tuple[Any, ...], kwargs: Any = None) -> None:
        if not self.active:
            return
        context = get_forward_context()
        self._saved_metadata = context.attn_metadata
        context.attn_metadata = self._metadata

    def _attn_post(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        if not self.active:
            return
        get_forward_context().attn_metadata = self._saved_metadata


def block_flops_share(spec: CompressionSpec | None, grid: tuple[int, int, int], num_blocks: int, num_other_tokens: int,
                      attn_keep: float = 0.2) -> dict[str, float]:
    """Relative DiT cost of the compressed model (1.0 = uncompressed), by FLOPs.

    Per block, per token: linear layers ~ ``12 d^2`` MACs (qkv+out+gate ~ 5 d^2, FFN SwiGLU 3 * d * 2.67 d), attention
    ~ ``2 * attended_keys * d``; VSA keeps ``attn_keep`` of the video keys. ``d`` cancels in the ratio when expressed in
    ``d`` units, so only the token counts matter; ``d = 5376`` is used for the attention/linear balance.
    """
    d = 5376.0
    video = grid[0] * grid[1] * grid[2]

    def cost(tokens_video: float) -> float:
        n = tokens_video + num_other_tokens
        linear = 13.4 * d * d * n
        attn = 2.0 * 2.0 * d * n * (attn_keep * tokens_video + num_other_tokens)
        return linear + attn

    full = cost(video)
    if spec is None:
        return {"relative_flops": 1.0, "speedup_flops": 1.0}
    reduced = cost(video / token_reduction(spec.mode, grid))
    inside = spec.end - spec.start
    total = (num_blocks - inside) * full + inside * reduced
    return {"relative_flops": total / (num_blocks * full), "speedup_flops": num_blocks * full / total}


__all__ = ["CompressionSpec", "TokenCompression", "block_flops_share", "compressed_grid", "grid_ops", "token_reduction"]
