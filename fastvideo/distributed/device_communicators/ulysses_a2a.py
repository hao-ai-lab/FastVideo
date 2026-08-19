# SPDX-License-Identifier: Apache-2.0
"""FlashInfer fused-transpose Ulysses all-to-all for sequence parallelism.

The sequence-parallel attention path performs two all-to-alls per attention
call: one that trades the sequence shard for a head shard before attention,
and its inverse afterwards (see ``fastvideo/attention/layer.py``). The default
implementation in :class:`DistributedAutograd.AllToAll4D` expresses that as
``permute -> dist.all_to_all_single -> permute``, because NCCL can only move
contiguous byte ranges and therefore needs the data pre-arranged into
per-destination blocks. Those permutes cost three full-tensor round trips
through HBM per collective and do no useful work.

FlashInfer's kernel folds the layout permutation into the cross-GPU write
addresses: each rank reads its local operand and pushes each destination block
straight into the peer's IPC-shared staging buffer over NVLink, so the same
bytes cross the wire with one pass over memory instead of four. Measured on
GB200 (sp_size=4, bf16): 1.50x on the head-scatter, 1.77x on the head-gather,
1.57x on the 3-scatter + 1-gather pattern an attention layer issues.

Layout equivalence with :class:`DistributedAutograd.AllToAll4D` is exact --
both compute ``y_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]``
-- so this is a drop-in replacement rather than a numerically different path,
and it is verified byte-for-byte in
``fastvideo/tests/distributed/test_ulysses_a2a_parity.py``.

The fused kernel only applies to a verified single-node all-pairs NVLink mesh
with a world size in ``(2, 4, 6, 8)``; everything else (multi-node, PCIe-only,
odd world sizes, oversized or oddly-typed operands, CUDA graph capture) falls
back to the existing NCCL path. Fallback is always silent and always correct:
a communication optimization must never be able to break a run.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fastvideo import envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# The kernel is template-specialized on the world size (RankData holds 8 peer
# pointers and NGPUS is a compile-time constant), so only these are dispatchable.
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)

# The kernel indexes operands with int32.
_INT32_MAX = 2**31 - 1

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# (scatter_dim, gather_dim) -> flashinfer mode.
#   mode 0 == scatter_heads: [B, S_local, H, D]        -> [B, S_global, H_local, D]
#   mode 1 == gather_heads:  [B, S_global, H_local, D] -> [B, S_local, H, D]
_MODE_FROM_DIMS = {(2, 1): 0, (1, 2): 1}


def is_enabled() -> bool:
    """Whether the fused path is opted in via ``FASTVIDEO_ULYSSES_A2A``."""
    return envs.FASTVIDEO_ULYSSES_A2A == "auto"


class _FusedUlyssesA2A(torch.autograd.Function):
    """Autograd wrapper around the fused collective.

    The two directions are exact inverses of each other, so the backward of a
    head-scatter is a head-gather and vice versa. No gradient scaling is
    involved: Ulysses redistributes activations, it does not reduce them, so
    every gradient element has exactly one source (unlike an all-reduce, whose
    backward must accumulate).
    """

    @staticmethod
    def forward(ctx, helper: "UlyssesA2AHelper", x: torch.Tensor, mode: int) -> torch.Tensor:  # type: ignore[override]
        ctx.helper = helper
        ctx.mode = mode
        return helper.run_armed(x, mode)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # The incoming gradient has the same element count and dtype as this
        # call's output, so the communicator is already sized for it; only
        # contiguity needs restoring (autograd may hand back a view).
        grad_input = ctx.helper.run_armed(grad_output.contiguous(), 1 - ctx.mode)
        return None, grad_input, None


class UlyssesA2AHelper:
    """Owns the FlashInfer communicator for one sequence-parallel group.

    Construction here is deliberately cheap and non-collective: the FlashInfer
    communicator is built on first use instead, because its constructor sizes a
    fixed staging buffer from ``max_elems`` and the process-group setup runs
    long before any activation shape is known. Building it lazily is safe
    because every rank runs the same module code and therefore reaches the same
    attention call with the same shapes.
    """

    def __init__(self, device_group: ProcessGroup, world_size: int, device: torch.device):
        self.device_group = device_group
        self.world_size = world_size
        self.device = device

        self._comm = None
        self._dtype: torch.dtype | None = None
        self._max_elems = 0
        # Set once the fused path is known to be unusable for this group, so
        # the topology probe and its collectives run at most once.
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
        """Whether this rank could use the fused path at all.

        Deliberately cheap and side-effect free: it allocates nothing and starts
        no collective, so it is safe to evaluate before the ranks have agreed on
        anything.
        """
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
        """Collectively build a communicator. Returns True if it is armed.

        The fused kernel opens with a barrier across every rank, so a rank that
        quietly falls back to NCCL while its peers proceed does not merely lose
        the optimization -- it strands them, and the job hangs until the NCCL
        watchdog fires. Deciding locally is therefore never correct here, even
        for a condition as mundane as an import failing on one node.

        So the ranks vote before anything is imported or allocated. Past that
        point every remaining failure is arbitrated by the backend's own
        group-wide protocol, which either arms all ranks or none. There is
        deliberately no second vote afterwards: teardown is itself collective
        while armed, so a recovery attempt from a split state would be the very
        deadlock it was trying to avoid.
        """
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
            # Not an error: FlashInfer's own NCCL fallback is correct, but it is
            # not autograd-aware and is no faster than the path we already have,
            # so hand the work back to DistributedAutograd.AllToAll4D.
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

    def precompile(self) -> None:
        """Report at warmup whether the fused path is usable.

        The kernel ships prebuilt in the fastvideo-kernel wheel, so unlike the
        JIT-based predecessor there is nothing to compile here. What remains is
        worth keeping: surfacing at startup, once, whether the kernel is present
        at all -- otherwise a wheel built without it looks identical to a slow
        run. Deliberately does not construct a communicator, since the warmup's
        dummy shapes and dtype are not the ones the model will use.
        """
        if self._disabled_reason is not None or not is_enabled():
            return
        ok, reason = self._can_attempt()
        if not ok:
            logger.info("Ulysses fused all-to-all unavailable at warmup: %s", reason)

    def close(self) -> None:
        """Release the IPC staging buffer and peer mappings.

        Collective while armed, so it must be reached by every rank -- which it
        is, via ``GroupCoordinator.destroy()``.
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
        """Fused collective, or ``None`` to let the caller use the NCCL path.

        Guards are ordered cheapest-first so the common reject costs a couple of
        comparisons. Every rejection is silent and local except the topology
        probe and the (re)build, which are collective -- and those are reached
        by all ranks together because the deciding inputs (world size, dtype,
        shape) are identical across ranks under SPMD execution.
        """
        if self._disabled_reason is not None or not is_enabled():
            return None

        mode = _MODE_FROM_DIMS.get((scatter_dim, gather_dim))
        if mode is None:
            return None

        if x.dim() != 4 or x.dtype not in _SUPPORTED_DTYPES or not x.is_contiguous():
            return None
        if x.device != self.device:
            return None

        # The head count must split across ranks for a scatter, and the global
        # sequence must split for a gather. FastVideo pads the sequence to a
        # multiple of sp_size before sharding, so the gather case holds by
        # construction; the scatter case depends on the model's head count.
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

        # The communicator pins both dtype and capacity at construction, and the
        # first operand seen is not necessarily representative. Rebuild rather
        # than fall back: a DiT uses one or two distinct shapes and a single
        # dtype, so this settles within the first step and is not a hot path.
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
    """Create a helper if the fused path could conceivably apply to this group."""
    if not is_enabled():
        return None
    if world_size <= 1 or device_group is None or device is None or device.type != "cuda":
        return None
    if not dist.is_initialized():
        return None
    return UlyssesA2AHelper(device_group, world_size, device)
