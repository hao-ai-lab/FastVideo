# SPDX-License-Identifier: Apache-2.0
"""Fused NVLink all-to-all for Ulysses sequence parallelism.

Drop-in replacement for DistributedAutograd.AllToAll4D on a single-node
all-pairs NVLink mesh: same layout, byte-identical results, ~1.5x faster per
attention layer. Anything else falls back to the NCCL path.

The kernel stores straight into peers' memory through NCCL's device API
(ncclGetLsaPointer), so NCCL owns the window, the topology and the barrier --
there is no IPC handle exchange or topology probe here.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fastvideo import envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# The kernel is template-specialized on the world size, so only these dispatch.
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)

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
        # Same numel and dtype as the forward output, so the window is already
        # sized for it; only contiguity needs restoring.
        grad_input = ctx.helper.run_armed(grad_output.contiguous(), 1 - ctx.mode)
        return None, grad_input, None


class UlyssesA2AHelper:
    """Owns the fused all-to-all context for one sequence-parallel group.

    Construction is cheap and non-collective; the NCCL window is registered on
    first use, once an operand size is known.
    """

    def __init__(self, device_group: ProcessGroup, world_size: int, device: torch.device, pynccl_comm):
        self.device_group = device_group
        self.world_size = world_size
        self.device = device
        self.pynccl_comm = pynccl_comm

        self._handle: int | None = None
        self._nbytes = 0
        self._disabled_reason: str | None = None

        if world_size not in SUPPORTED_WORLD_SIZES:
            self._disabled_reason = (f"world size {world_size} is not one of "
                                     f"{SUPPORTED_WORLD_SIZES}")

    # -- lifecycle -----------------------------------------------------------

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            logger.info("Ulysses fused all-to-all disabled: %s", reason)

    def _comm_ptr(self) -> int:
        comm = self.pynccl_comm.comm
        return int(getattr(comm, "value", comm))

    def _can_attempt(self) -> tuple[bool, str]:
        """Whether this rank could use the fused path, without allocating anything."""
        try:
            from fastvideo_kernel import comm_ops
            if not comm_ops.is_available():
                return False, "fastvideo-kernel was built without the Ulysses a2a kernel"
            if not comm_ops.lsa_covers_group(self._comm_ptr(), self.world_size):
                return False, "the group is not a load-store-accessible (NVLink) mesh"
        except Exception as e:  # noqa: BLE001
            return False, f"backend unavailable ({type(e).__name__}: {e})"
        return True, ""

    def _agree(self, ok: bool) -> bool:
        """Reduce a local yes/no to a group-wide verdict: True only if all agree."""
        vote = torch.tensor([1 if ok else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.device_group)
        return bool(vote.item())

    def _build(self, nbytes: int) -> bool:
        """Collectively register the window. Returns True if it is armed."""
        # The kernel opens with a barrier across every rank, so a rank that falls
        # back alone strands its peers until the NCCL watchdog fires. Hence the
        # vote, before anything is registered. There is deliberately no second
        # vote afterwards: window teardown is itself collective, so recovering
        # from a split state would be that same deadlock.
        ok, reason = self._can_attempt()
        if not self._agree(ok):
            self._disable(reason or "a peer rank cannot use the fused path")
            return False

        from fastvideo_kernel import comm_ops

        try:
            self._handle = comm_ops.init(self._comm_ptr(), nbytes, self.pynccl_comm.rank, self.world_size)
        except Exception as e:  # noqa: BLE001 - never break the caller
            self._disable(f"window registration failed ({type(e).__name__}: {e})")
            return False

        self._nbytes = nbytes
        logger.info("Ulysses fused all-to-all armed: world_size=%d window=%.0f MiB", self.world_size, nbytes / 2**20)
        return True

    def close(self) -> None:
        """Deregister the window.

        Collective while armed, so every rank must reach it via
        GroupCoordinator.destroy().
        """
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            from fastvideo_kernel import comm_ops
            comm_ops.dispose(handle)
        except Exception:  # noqa: BLE001 - teardown must not mask a real error
            logger.warning("Ulysses window deregistration failed", exc_info=True)

    # -- collective ----------------------------------------------------------

    def run_armed(self, x: torch.Tensor, mode: int) -> torch.Tensor:
        """Run one collective on an already-armed context."""
        assert self._handle is not None, "run_armed called on an unarmed helper"
        from fastvideo_kernel import comm_ops

        w = self.world_size
        if mode == 0:
            B, S_local, H, D = x.shape
            out = torch.empty(B, S_local * w, H // w, D, dtype=x.dtype, device=x.device)
        else:
            B, S_global, H_local, D = x.shape
            S_local, H = S_global // w, H_local * w
            out = torch.empty(B, S_local, H, D, dtype=x.dtype, device=x.device)
        comm_ops.all_to_all(self._handle, x, out, B, S_local, H, D, mode)
        return out

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

        # Spin-wait barriers make the fused kernel unsafe to capture.
        if torch.cuda.is_current_stream_capturing():
            return None

        # The window is fixed at registration and the first operand is not
        # necessarily the largest, so grow rather than fall back.
        nbytes = x.numel() * x.element_size()
        if self._handle is None:
            if not self._build(nbytes):
                return None
        elif nbytes > self._nbytes:
            logger.info("Ulysses window grow: %d -> %d bytes", self._nbytes, nbytes)
            self.close()
            if not self._build(nbytes):
                return None

        return _FusedUlyssesA2A.apply(self, x, mode)


def maybe_create_helper(device_group: ProcessGroup | None, world_size: int, device: torch.device | None,
                        pynccl_comm) -> UlyssesA2AHelper | None:
    """Create a helper if the fused path could apply to this group."""
    if not is_enabled():
        return None
    if world_size <= 1 or device_group is None or device is None or device.type != "cuda":
        return None
    if not dist.is_initialized():
        return None
    # The kernel needs an ncclComm_t for the group; PyNcclCommunicator has one.
    if pynccl_comm is None or pynccl_comm.disabled:
        return None
    return UlyssesA2AHelper(device_group, world_size, device, pynccl_comm)
