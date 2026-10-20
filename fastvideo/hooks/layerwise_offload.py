from contextlib import contextmanager
from itertools import chain
from typing import Any
import torch
from torch import nn
from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
import fastvideo.envs as envs
from fastvideo.hooks.pinned_memory import PinnedTensorArena
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _tensor_placeholder(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create a rank-preserving empty placeholder on the specified device."""
    shape = (0, ) if tensor.ndim <= 0 else (0, ) * tensor.ndim
    return torch.empty(shape, device=device, dtype=tensor.dtype)


# Buffers at least this large also stream (e.g. packed NVFP4 weights registered as buffers);
# small ones (scales, caches) stay resident. Opt in with FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS=1.
_BUFFER_OFFLOAD_MIN_BYTES = 1 << 20


def _offload_tensors(module: nn.Module, names: dict[str, torch.Tensor] | None = None):
    """``(name, tensor)`` for every parameter and, when enabled, every large buffer.

    ``names`` restricts the walk to the tensors chosen at init: an offloaded buffer is a
    zero-element placeholder afterwards and would fail the size test.
    """
    if names is not None:
        for name, tensor in chain(module.named_parameters(), module.named_buffers()):
            if name in names:
                yield name, tensor
        return
    yield from module.named_parameters()
    if envs.FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS.get():
        for name, buf in module.named_buffers():
            if buf is not None and buf.numel() * buf.element_size() >= _BUFFER_OFFLOAD_MIN_BYTES:
                yield name, buf


class LayerwiseOffloadState:

    def __init__(
        self,
        async_copy_stream: torch.cuda.Stream,
        device: torch.device,
        next_state: "LayerwiseOffloadState | None" = None,
    ) -> None:
        self.async_copy_stream = async_copy_stream
        self.next_state = next_state
        self.gpu_named_parameters: dict[str, torch.Tensor] = {}
        self.cpu_named_parameters: dict[str, torch.Tensor] = {}
        self.module_ref: nn.Module = None  # type: ignore
        self.device: torch.device = device
        self.cpu_arena: PinnedTensorArena | None = None

    def _will_offload(self, name: str) -> bool:
        return True

    @torch.compiler.disable
    def on_init(self, module: nn.Module):
        self.module_ref = module
        self.clear_cpu_storage()
        self.cpu_arena = PinnedTensorArena(
            (name, param) for name, param in _offload_tensors(module) if self._will_offload(name))
        for name, param in _offload_tensors(self.module_ref):
            if self._will_offload(name):
                host = self.cpu_arena.empty_like(name, param)
                host.copy_(param.data.detach())
                self.cpu_named_parameters[name] = host
                param.data = _tensor_placeholder(param.data, self.device)

    def clear_cpu_storage(self) -> None:
        self.cpu_named_parameters.clear()
        if self.cpu_arena is not None:
            self.cpu_arena.close()
            self.cpu_arena = None

    @torch.compiler.disable
    def wait_and_replace_params(self):
        torch.cuda.current_stream().wait_stream(self.async_copy_stream)
        # now gpu_named_parameters are ready
        for name, param in _offload_tensors(self.module_ref, self.cpu_named_parameters):
            if not self._will_offload(name):
                continue
            if name not in self.gpu_named_parameters:
                # first load with blocking load
                self.gpu_named_parameters[name] = self.cpu_named_parameters[name].to(self.device)
            param.data = self.gpu_named_parameters[name]

    @torch.compiler.disable
    def prefetch_params(self):
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            for name, param in _offload_tensors(self.module_ref, self.cpu_named_parameters):
                if not self._will_offload(name):
                    continue
                assert name not in self.gpu_named_parameters
                gpu_param = self.cpu_named_parameters[name].to(self.device, non_blocking=True)
                gpu_param.record_stream(compute_stream)  # ensure tensor will not be freed until forward is completed
                self.gpu_named_parameters[name] = gpu_param

    @torch.compiler.disable
    def release_gpu_params(self):
        for name, param in _offload_tensors(self.module_ref, self.cpu_named_parameters):
            if self._will_offload(name):
                param.data = _tensor_placeholder(param.data, self.device)
                del self.gpu_named_parameters[name]
        assert len(self.gpu_named_parameters) == 0


class LayerwiseOffloadHook(ForwardHook):
    """A hook that enables layerwise CPU offloading during forward pass."""

    def __init__(self, state: LayerwiseOffloadState) -> None:
        self.state = state

    def on_attach(self, module: nn.Module):
        self.state.on_init(module)  # pyright: ignore

    def on_detach(self, module: nn.Module):
        self.state.async_copy_stream.synchronize()
        named_parameters = dict(_offload_tensors(module, self.state.cpu_named_parameters))
        for name, cpu_tensor in self.state.cpu_named_parameters.items():
            if name in named_parameters:
                gpu_tensor = self.state.gpu_named_parameters.get(name)
                named_parameters[name].data = gpu_tensor if gpu_tensor is not None else cpu_tensor.to(self.state.device)
            else:
                logger.warning("Parameter %s not found in module during detachment.", name)
        self.state.gpu_named_parameters.clear()
        self.state.clear_cpu_storage()
        self.state.next_state = None

    @classmethod
    def name(cls) -> str:
        return "LayerwiseOffloadHook"

    # These hook entry points only orchestrate host-side parameter
    # movement (stream waits, pinned H2D/D2H, param swapping) and must
    # always run eager. Without an explicit boundary, torch.compile
    # traces *into* the hook, hits the `@torch.compiler.disable`
    # `LayerwiseOffloadState` methods, and is forced to bail mid-trace
    # — an implicit graph break at the call site, once per layer every
    # step, which fragments the per-layer compiled region (and blocks
    # CUDA-graph capture, which cannot span a break). Marking the entry
    # points disabled makes the hook a clean opaque eager boundary so
    # torch.compile keeps one contiguous region around it. Completes
    # the pattern already applied to the State methods.
    @torch.compiler.disable
    def pre_forward(self, module: nn.Module, *args, **kwargs):
        self.state.wait_and_replace_params()  # pyright: ignore
        if self.state.next_state is not None:
            self.state.next_state.prefetch_params()  # pyright: ignore
        return args, kwargs

    @torch.compiler.disable
    def post_forward(self, module: torch.nn.Module, output: Any):
        self.state.release_gpu_params()  # pyright: ignore
        return output

    @contextmanager
    def mutate_params_scope(self):
        try:
            # load params to GPU and keep them there
            self.state.wait_and_replace_params()  # pyright: ignore
            yield
        finally:
            # instead of releasing, we should overwrite the original params since they have been modified
            self.state.gpu_named_parameters.clear()
            self.state.on_init(self.state.module_ref)  # pyright: ignore


def enable_layerwise_offload(model: nn.Module,
                             is_replace: bool = False,
                             *,
                             resident_blocks: int | None = None,
                             cyclic: bool = True):
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        logger.warning("CUDA is not available. Layerwise offloading is disabled.")
        return
    state_list = []
    async_stream = torch.cuda.Stream()
    # The first N entries skip offloading and stay wherever the model is placed (normally the
    # GPU), so a GPU with spare memory streams only the remainder over PCIe.
    if resident_blocks is not None:
        resident = max(0, resident_blocks)
    else:
        try:
            resident = max(0, envs.FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS.get())
        except ValueError as error:
            logger.warning("Ignoring malformed FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS: %s", error)
            resident = 0
    for name, submodule in model.named_children():
        if isinstance(submodule, nn.ModuleList):
            for idx, module_entry in enumerate(submodule):
                if idx < resident:
                    continue
                state = LayerwiseOffloadState(async_copy_stream=async_stream, device=device)
                state_list.append(state)
                hook_mgr = ModuleHookManager.get_from_or_default(module_entry)
                hook = LayerwiseOffloadHook(state)
                if is_replace:
                    existing_hook = hook_mgr.forward_hooks.get(hook.name())
                    if existing_hook is not None:
                        hook_mgr.replace_forward_hook(hook.name(), hook)
                    else:
                        raise AssertionError(f"Expect hook exists in {name} for replacement.")
                else:
                    hook_mgr.append_forward_hook(hook)
            break
    if len(state_list) == 0:
        if resident > 0:
            logger.info("FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS=%d keeps every block resident; nothing to offload",
                        resident)
            return
        raise ValueError("No nn.ModuleList found in the model for layerwise offloading.")

    # Repeated DiT steps prefetch the first block after the last. A once-per-request
    # encoder can skip that unused copy and release every layer after its forward.
    for i in range(len(state_list)):
        state_list[i].next_state = state_list[(i + 1) % len(state_list)] if cyclic or i + 1 < len(state_list) else None
