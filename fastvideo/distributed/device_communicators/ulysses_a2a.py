# SPDX-License-Identifier: Apache-2.0
"""Fused NVLink all-to-all for Ulysses sequence parallelism.

Drop-in replacement for DistributedAutograd.AllToAll4D on a single-node
all-pairs NVLink mesh: same layout, byte-identical results, ~1.5x faster per
attention layer. Anything else falls back to the NCCL path.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fastvideo import envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# The kernel is template-specialized on the world size, so only these dispatch.
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)

# The kernel indexes operands with int32.
_INT32_MAX = 2**31 - 1

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# (scatter_dim, gather_dim) -> kernel mode.
#   0: [B, S_local, H, D]        -> [B, S_global, H_local, D]
#   1: [B, S_global, H_local, D] -> [B, S_local, H, D]
_MODE_FROM_DIMS = {(2, 1): 0, (1, 2): 1}


def is_enabled() -> bool:
    """Whether the fused path is opted in via FASTVIDEO_ULYSSES_A2A."""
    return envs.FASTVIDEO_ULYSSES_A2A == "auto"


class _FusedUlyssesA2A(torch.autograd.Function):
    """Differentiable fused all-to-all.

    The two directions are exact inverses, and Ulysses redistributes activations
    rather than reducing them, so backward is the opposite mode with no scaling.
    """

    @staticmethod
    def forward(ctx, helper: "UlyssesA2AHelper", x: torch.Tensor, mode: int) -> torch.Tensor:  # type: ignore[override]
        ctx.helper = helper
        ctx.mode = mode
        return helper.run_armed(x, mode)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Same numel and dtype as the forward output, so the communicator is
        # already sized for it; only contiguity needs restoring.
        grad_input = ctx.helper.run_armed(grad_output.contiguous(), 1 - ctx.mode)
        return None, grad_input, None


class UlyssesA2AHelper:
    """Owns the fused communicator for one sequence-parallel group.

    Construction is cheap and non-collective; the communicator is built on first
    use, once an operand shape is known.
    """

    def __init__(self, device_group: ProcessGroup, world_size: int, device: torch.device):
        self.device_group = device_group
        self.world_size = world_size
        self.device = device

        self._comm = None
        self._dtype: torch.dtype | None = None
        self._max_elems = 0
        self._disabled_reason: str | None = None

        if world_size not in SUPPORTED_WORLD_SIZES:
            self._disabled_reason = (f"world size {world_size} is not one of "
                                     f"{SUPPORTED_WORLD_SIZES}")

    # -- lifecycle -----------------------------------------------------------

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            logger.info("Ulysses fused all-to-all disabled: %s", reason)

    def _can_attempt(self) -> tuple[bool, str]:
        """Whether this rank could use the fused path, without allocating anything."""
        try:
            from fastvideo_kernel import comm_ops
            if not comm_ops.is_available():
                return False, "fastvideo-kernel was built without the Ulysses a2a kernel"
        except Exception as e:  # noqa: BLE001
            return False, f"backend unavailable ({type(e).__name__}: {e})"
        return True, ""

    def _agree(self, ok: bool) -> bool:
        """Reduce a local yes/no to a group-wide verdict: True only if all agree."""
        vote = torch.tensor([1 if ok else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.device_group)
        return bool(vote.item())

    def _build(self, dtype: torch.dtype, max_elems: int) -> bool:
        """Collectively build a communicator. Returns True if it is armed."""
        # The kernel opens with a barrier across every rank, so a rank that falls
        # back alone strands its peers until the NCCL watchdog fires. Hence the
        # vote, before anything is imported or allocated. There is deliberately
        # no second vote afterwards: teardown is itself collective while armed,
        # so recovering from a split state would be that same deadlock.
        ok, reason = self._can_attempt()
        if not self._agree(ok):
            self._disable(reason or "a peer rank cannot use the fused path")
            return False

        from .ulysses_comm import UlyssesCommunicator

        try:
            comm = UlyssesCommunicator(
                self.device_group,
                max_elems=max_elems,
                dtype=dtype,
                backend="auto",
                device=self.device,
            )
        except Exception as e:  # noqa: BLE001 - never break the caller
            self._disable(f"communicator construction failed ({type(e).__name__}: {e})")
            return False

        if comm.backend != "nvlink":
            # Its NCCL fallback is correct but not autograd-aware and no faster
            # than the path we already have, so hand the work back.
            self._disable(f"topology not eligible ({comm.fallback_reason})")
            try:
                comm.close()
            except Exception:  # noqa: BLE001
                logger.warning("Ulysses communicator close() failed after fallback", exc_info=True)
            return False

        self._comm = comm
        self._dtype = dtype
        self._max_elems = max_elems
        logger.info("Ulysses fused all-to-all armed: world_size=%d dtype=%s capacity=%d elems (%.0f MiB)",
                    self.world_size, dtype, max_elems, max_elems * dtype.itemsize / 2**20)
        return True

    def close(self) -> None:
        """Release the IPC staging buffer and peer mappings.

        Collective while armed, so every rank must reach it via
        GroupCoordinator.destroy().
        """
        if self._comm is None:
            return
        comm, self._comm = self._comm, None
        try:
            comm.close()
        except Exception:  # noqa: BLE001 - teardown must not mask a real error
            logger.warning("Ulysses communicator close() failed", exc_info=True)

    # -- collective ----------------------------------------------------------

    def run_armed(self, x: torch.Tensor, mode: int) -> torch.Tensor:
        """Run one collective on an already-armed communicator."""
        assert self._comm is not None, "run_armed called on an unarmed helper"
        return self._comm.scatter_heads(x) if mode == 0 else self._comm.gather_heads(x)

    def try_all_to_all_4D(self, x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor | None:
        """Fused collective, or None to let the caller use the NCCL path."""
        if self._disabled_reason is not None or not is_enabled():
            return None

        mode = _MODE_FROM_DIMS.get((scatter_dim, gather_dim))
        if mode is None:
            return None

        if x.dim() != 4 or x.dtype not in _SUPPORTED_DTYPES or not x.is_contiguous():
            return None
        if x.device != self.device:
            return None

        # Scatter splits the heads, gather splits the global sequence.
        if mode == 0 and x.shape[2] % self.world_size != 0:
            return None
        if mode == 1 and x.shape[1] % self.world_size != 0:
            return None

        numel = x.numel()
        if numel > _INT32_MAX:
            self._disable(f"operand of {numel} elements exceeds the kernel's int32 index range")
            return None

        # Spin-wait barriers and a host-side sync during init make the fused
        # kernel unsafe to capture.
        if torch.cuda.is_current_stream_capturing():
            return None

        # dtype and capacity are pinned at construction and the first operand is
        # not necessarily the largest, so grow rather than fall back.
        if self._comm is None:
            if not self._build(x.dtype, numel):
                return None
        elif x.dtype != self._dtype or numel > self._max_elems:
            logger.info("Ulysses communicator rebuild: dtype %s -> %s, capacity %d -> %d elems", self._dtype, x.dtype,
                        self._max_elems, max(self._max_elems, numel))
            self.close()
            if not self._build(x.dtype, max(self._max_elems, numel)):
                return None

        return _FusedUlyssesA2A.apply(self, x, mode)


def maybe_create_helper(device_group: ProcessGroup | None, world_size: int,
                        device: torch.device | None) -> UlyssesA2AHelper | None:
    """Create a helper if the fused path could apply to this group."""
    if not is_enabled():
        return None
    if world_size <= 1 or device_group is None or device is None or device.type != "cuda":
        return None
    if not dist.is_initialized():
        return None
    return UlyssesA2AHelper(device_group, world_size, device)
