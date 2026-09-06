# SPDX-License-Identifier: Apache-2.0
"""Fused NVLink all-to-all for Ulysses sequence parallelism.

Drop-in replacement for DistributedAutograd.AllToAll4D when the group is a
load-store accessible NVLink mesh: same layout, byte-identical results, fewer
passes over local memory. Anything else falls back to the NCCL path.
"""

import socket
from array import array

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fastvideo import envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# The kernel is template-specialized on the world size, so only these dispatch.
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)

_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}

# Bound persistent registered memory per rank. Larger operands use NCCL instead
# of growing the window without limit.
MAX_WINDOW_BYTES = 1024**3
H3_TRAINING_PLANE_LIMIT_BYTES = 512 * 1024**2
_CONTRACT_SIZE = 12

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
    def forward(ctx, helper: "UlyssesA2AHelper", x: torch.Tensor, mode: int, chunked: bool,
                blocks: int) -> torch.Tensor:  # type: ignore[override]
        ctx.helper = helper
        ctx.mode = mode
        ctx.chunked = chunked
        ctx.blocks = blocks
        return helper.run_armed(x, mode, chunked, blocks)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Same numel and dtype as the forward output, so the window is already
        # sized for it; only contiguity needs restoring.
        # Reuse the forward plan, even if later calls chose another plan or
        # backward executes with grad tracking disabled.
        grad_input = ctx.helper.run_armed(grad_output.contiguous(), 1 - ctx.mode, ctx.chunked, ctx.blocks)
        return None, grad_input, None, None, None


class UlyssesA2AHelper:
    """Owns the fused all-to-all context for one sequence-parallel group.

    Group capability is agreed during construction; the NCCL window is
    registered on first use, once an operand size is known.
    """

    def __init__(self, cpu_group: ProcessGroup, device_group: ProcessGroup, world_size: int, device: torch.device,
                 pynccl_comm):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.world_size = world_size
        self.device = device
        self.pynccl_comm = pynccl_comm

        self._handle: int | None = None
        # Reuse storage, but exchange the current contract on every call. A
        # rank-local cache hit cannot establish what peers are doing now.
        self._local_contract = array("q", [0] * _CONTRACT_SIZE)
        self._local_tensor = torch.frombuffer(self._local_contract, dtype=torch.int64)
        self._gathered_tensor = torch.empty(world_size * _CONTRACT_SIZE, dtype=torch.int64, device="cpu")
        self._nbytes = 0
        self._disabled_reason: str | None = None
        self._h3_tuning_available: bool | None = None

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
        """Check local capability only; the caller exchanges every rank's result."""
        try:
            from fastvideo_kernel import comm_ops
            if not comm_ops.is_available():
                return False, "fastvideo-kernel was built without the Ulysses a2a kernel"
            elif not comm_ops.lsa_covers_group(self._comm_ptr(), self.world_size):
                return False, "the group is not a load-store-accessible (NVLink) mesh"
        except Exception as e:  # noqa: BLE001
            return False, f"backend unavailable ({type(e).__name__}: {e})"
        return True, ""

    def _agree(self, ok: bool) -> bool:
        """Reduce a local yes/no to a group-wide verdict: True only if all agree."""
        vote = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cpu")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.cpu_group)
        return bool(vote.item())

    def _allocate(self, nbytes: int) -> int:
        """Allocate locally; split out so allocation-failure tests can inject."""
        from fastvideo_kernel import comm_ops

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return comm_ops.allocate(nbytes, self.pynccl_comm.rank, self.world_size, device_index)

    def _register_window(self, handle: int) -> None:
        """Register the user window collectively."""
        from fastvideo_kernel import comm_ops

        comm_ops.register_window(handle, self._comm_ptr())

    def _create_dev_comm(self, handle: int) -> None:
        """Create the NCCL device communicator collectively."""
        from fastvideo_kernel import comm_ops

        comm_ops.create_dev_comm(handle)

    def _dispose(self, handle: int, *, synchronize: bool) -> None:
        from fastvideo_kernel import comm_ops

        if synchronize:
            # Kernel launches and copy-out are asynchronous. Do not deregister a
            # window that a prior call on this device is still accessing.
            torch.cuda.synchronize(self.device)
        comm_ops.dispose(handle)

    def _dispose_after_failure(self, handle: int | None) -> bool:
        """Best-effort group cleanup after a setup phase failed.

        Every rank votes after attempting cleanup, including a rank that never
        obtained a local allocation. This keeps the helper permanently disabled
        if teardown was not unanimous instead of re-entering with split state.
        """
        cleanup_ok = True
        if handle is not None:
            try:
                self._dispose(handle, synchronize=False)
            except Exception:  # noqa: BLE001 - converted to a group verdict below
                cleanup_ok = False
                logger.warning("Ulysses partial-context cleanup failed", exc_info=True)
        return self._agree(cleanup_ok)

    def _execution_plan(self, x: torch.Tensor, mode: int) -> tuple[bool, int]:
        """Use the measured GB200/SP4 launch only for H3's bf16 head geometry."""
        global_heads = x.shape[2] if mode == 0 else x.shape[2] * self.world_size
        if self.world_size != 4 or x.dtype != torch.bfloat16 or global_heads != 56 or x.shape[3] != 128:
            return False, 36
        if self._h3_tuning_available is None:
            from fastvideo_kernel import comm_ops

            props = torch.cuda.get_device_properties(self.device)
            self._h3_tuning_available = ("GB200" in props.name and props.major == 10 and props.minor == 0
                                         and props.multi_processor_count >= 144
                                         and getattr(comm_ops, "supports_tuned_launch", lambda: False)())
        return (True, 144) if self._h3_tuning_available else (False, 36)

    def _call_signature(self, x: torch.Tensor, scatter_dim: int, gather_dim: int) -> tuple[tuple[int, ...], str]:
        """Return a rank-comparable call contract and any local decline reason."""
        mode = _MODE_FROM_DIMS.get((scatter_dim, gather_dim))
        dtype_code = _DTYPE_CODES.get(x.dtype, 0)
        shape = tuple(int(dim) for dim in x.shape) if x.dim() == 4 else (0, 0, 0, 0)
        status = 1
        reason = ""

        if self._disabled_reason is not None:
            status, reason = -1, self._disabled_reason
        elif not is_enabled():
            status, reason = 0, "FASTVIDEO_ULYSSES_A2A is not auto"
        elif x.is_cuda and torch.cuda.is_current_stream_capturing():
            status, reason = 0, "the current CUDA stream is being captured"
        elif mode is None:
            status, reason = 0, "unsupported scatter/gather dimensions"
        elif x.dim() != 4:
            status, reason = 0, "input is not 4-D"
        elif dtype_code == 0:
            status, reason = 0, f"unsupported dtype {x.dtype}"
        elif not x.is_cuda or x.device != self.device:
            status, reason = 0, f"input device {x.device} does not match {self.device}"
        elif not x.is_contiguous():
            status, reason = 0, "input is not contiguous"
        elif mode == 0 and shape[2] % self.world_size != 0:
            status, reason = 0, "head count is not divisible by the group"
        elif mode == 1 and shape[1] % self.world_size != 0:
            status, reason = 0, "sequence length is not divisible by the group"

        nbytes = int(x.numel() * x.element_size())
        chunked, blocks = False, 36
        if status == 1 and nbytes:
            assert mode is not None
            chunked, blocks = self._execution_plan(x, mode)
        window_bytes = nbytes // shape[0] if chunked else nbytes
        if status == 1 and nbytes == 0:
            status, reason = 0, "input is empty"
        elif status == 1 and window_bytes > MAX_WINDOW_BYTES:
            status, reason = 0, f"operand window exceeds the {MAX_WINDOW_BYTES}-byte cap"
        elif (status == 1 and chunked and shape[0] > 1 and torch.is_grad_enabled() and x.requires_grad
              and window_bytes > H3_TRAINING_PLANE_LIMIT_BYTES):
            status, reason = 0, "large packed H3 training transfer uses NCCL"

        # status, armed, mode, dtype, B, S, H, D, window bytes, capacity,
        # chunked, blocks. Comparing the
        # whole vector prevents equal-size but differently-shaped ranks from
        # entering the fused kernel with incompatible address math. CUDA device
        # ordinals are deliberately absent: rank-local ordinals normally differ.
        signature = (status, int(self._handle is not None), -1 if mode is None else mode, dtype_code, *shape,
                     window_bytes, self._nbytes, int(chunked), blocks)
        return signature, reason

    def _agree_call(self, signature: tuple[int, ...]) -> tuple[bool, bool, bool]:
        """Agree on eligibility and the complete call signature across ranks.

        Returns ``(use_fused, permanently_unavailable, lifecycle_consistent)``. This control
        collective is intentionally eager-only; compiled regions use the NCCL
        implementation before reaching here.
        """
        # Host-side Gloo control keeps this agreement outside CUDA graph capture
        # and avoids inserting a second NCCL collective ahead of the data path.
        self._local_contract[:] = array("q", signature)
        dist.all_gather_into_tensor(self._gathered_tensor, self._local_tensor, group=self.cpu_group)
        values = self._gathered_tensor.tolist()
        contracts = [values[start:start + _CONTRACT_SIZE] for start in range(0, len(values), _CONTRACT_SIZE)]
        first = contracts[0]
        use_fused = first[0] == 1 and all(contract == first for contract in contracts)
        permanently_unavailable = any(contract[0] < 0 for contract in contracts)
        lifecycle_consistent = all(contract[1] == first[1] and contract[9] == first[9] for contract in contracts)
        return use_fused, permanently_unavailable, lifecycle_consistent

    def _build(self, nbytes: int) -> bool:
        """Collectively register the window. Returns True if it is armed."""
        handle: int | None = None
        allocation_reason = ""
        try:
            handle = self._allocate(nbytes)
        except Exception as e:  # noqa: BLE001 - converted to a group verdict below
            allocation_reason = f"window allocation failed ({type(e).__name__}: {e})"

        # Allocation is local, so vote before any rank enters registration.
        if not self._agree(handle is not None):
            cleanup_ok = self._dispose_after_failure(handle)
            reason = allocation_reason or "a peer rank could not allocate the window"
            if not cleanup_ok:
                reason += "; partial-context cleanup failed on a peer"
            self._disable(reason)
            return False

        assert handle is not None
        window_registered = False
        registration_reason = ""
        try:
            self._register_window(handle)
            window_registered = True
        except Exception as e:  # noqa: BLE001 - converted to a group verdict below
            registration_reason = f"window registration failed ({type(e).__name__}: {e})"

        if not self._agree(window_registered):
            cleanup_ok = self._dispose_after_failure(handle)
            reason = registration_reason or "a peer rank could not register the window"
            if not cleanup_ok:
                reason += "; partial-context cleanup failed on a peer"
            self._disable(reason)
            return False

        dev_comm_created = False
        creation_reason = ""
        try:
            self._create_dev_comm(handle)
            dev_comm_created = True
        except Exception as e:  # noqa: BLE001 - converted to a group verdict below
            creation_reason = f"device communicator creation failed ({type(e).__name__}: {e})"

        if not self._agree(dev_comm_created):
            cleanup_ok = self._dispose_after_failure(handle)
            reason = creation_reason or "a peer rank could not create the device communicator"
            if not cleanup_ok:
                reason += "; partial-context cleanup failed on a peer"
            self._disable(reason)
            return False

        self._handle = handle
        self._nbytes = nbytes
        logger.info("Ulysses fused all-to-all armed: world_size=%d window=%.0f MiB", self.world_size, nbytes / 2**20)
        return True

    def close(self) -> bool:
        """Collectively destroy the device communicator and its window.

        Returns whether all ranks completed teardown. An armed/unarmed split
        cannot safely enter NCCL window deregistration, so that exceptional
        state is leaked until process exit and permanently disabled instead of
        risking a distributed deadlock.
        """
        handle = self._handle
        all_armed = self._agree(handle is not None)
        all_unarmed = self._agree(handle is None)
        if all_unarmed:
            self._nbytes = 0
            return True
        if not all_armed:
            self._handle = None
            self._nbytes = 0
            self._disable("ranks disagreed on whether a fused window was armed during teardown")
            return False

        assert handle is not None
        synchronize_ok = True
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001 - converted to a group verdict below
            synchronize_ok = False
            logger.warning("Ulysses pre-teardown synchronization failed", exc_info=True)
        if not self._agree(synchronize_ok):
            self._disable("a peer rank could not synchronize before fused-window teardown")
            return False

        dispose_ok = True
        try:
            self._dispose(handle, synchronize=False)
        except Exception:  # noqa: BLE001 - teardown must not mask a real error
            dispose_ok = False
            logger.warning("Ulysses window deregistration failed", exc_info=True)

        group_ok = self._agree(dispose_ok)
        # The native disposer consumes the handle even when a cleanup call
        # reports an error, so never retry a potentially dangling pointer.
        self._handle = None
        self._nbytes = 0
        if not group_ok:
            self._disable("fused-window teardown failed on a peer rank")
        return group_ok

    # -- collective ----------------------------------------------------------

    def run_armed(self, x: torch.Tensor, mode: int, chunked: bool, blocks: int) -> torch.Tensor:
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
        if chunked:
            # Each copy completes on this stream before the next plane reuses
            # the registered window. The full result owns its storage; saved
            # activations never alias the reusable window.
            for plane in range(B):
                comm_ops.all_to_all(self._handle,
                                    x[plane:plane + 1],
                                    out[plane:plane + 1],
                                    1,
                                    S_local,
                                    H,
                                    D,
                                    mode,
                                    blocks=blocks)
        else:
            comm_ops.all_to_all(self._handle, x, out, B, S_local, H, D, mode)
        return out

    def try_all_to_all_4D(self, x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor | None:
        """Fused collective, or None to let the caller use the NCCL path."""
        if self._disabled_reason is not None:
            return None

        # Python lifecycle checks, votes, and pybind calls are not valid inside
        # a fullgraph region. The inherited NCCL path is compiler-visible, so
        # regional compile stays fullgraph by declining before any tensor read.
        if torch.compiler.is_compiling():
            return None

        signature, reason = self._call_signature(x, scatter_dim, gather_dim)
        use_fused, permanently_unavailable, lifecycle_consistent = self._agree_call(signature)
        if not use_fused:
            if not lifecycle_consistent:
                self.close()
                self._disable("ranks disagreed on the fused-window lifecycle")
            if permanently_unavailable:
                self._disable(reason or "a peer rank cannot use the fused path")
            return None

        mode = signature[2]
        nbytes = signature[8]
        if self._handle is None:
            if not self._build(nbytes):
                return None
        elif nbytes > self._nbytes:
            logger.info("Ulysses window grow: %d -> %d bytes", self._nbytes, nbytes)
            if not self.close():
                return None
            if not self._build(nbytes):
                return None

        return _FusedUlyssesA2A.apply(self, x, mode, bool(signature[10]), signature[11])


def maybe_create_helper(cpu_group: ProcessGroup | None, device_group: ProcessGroup | None, world_size: int,
                        device: torch.device | None, pynccl_comm) -> UlyssesA2AHelper | None:
    """Collectively create a helper only when every rank can use it."""
    if (world_size <= 1 or cpu_group is None or device_group is None or device is None or device.type != "cuda"):
        return None
    if not dist.is_initialized():
        return None

    helper = None
    reason = ""
    if not is_enabled():
        reason = "FASTVIDEO_ULYSSES_A2A is not auto"
    elif world_size not in SUPPORTED_WORLD_SIZES:
        reason = f"world size {world_size} is not one of {SUPPORTED_WORLD_SIZES}"
    elif pynccl_comm is None or pynccl_comm.disabled:
        reason = "the group has no usable PyNccl communicator"
    else:
        try:
            candidate = UlyssesA2AHelper(cpu_group, device_group, world_size, device, pynccl_comm)
            can_attempt, reason = candidate._can_attempt()
            if can_attempt:
                helper = candidate
        except Exception as e:  # noqa: BLE001 - converted to a group verdict below
            reason = f"helper construction failed ({type(e).__name__}: {e})"

    # Every rank reaches the same exchange, including configuration, constructor,
    # and backend failures. LSA covers addressability, not single-host locality.
    gathered: list[tuple[str, bool]] = [("", False)] * world_size
    dist.all_gather_object(gathered, (socket.gethostname(), helper is not None), group=cpu_group)
    hostnames = {hostname for hostname, _ in gathered}
    if len(hostnames) != 1:
        reason = f"ranks span multiple hosts: {sorted(hostnames)}"
    if len(hostnames) != 1 or not all(ok for _, ok in gathered):
        if dist.get_rank(cpu_group) == 0:
            logger.info("Ulysses fused all-to-all unavailable: %s", reason or "a peer rank declined")
        return None
    return helper
