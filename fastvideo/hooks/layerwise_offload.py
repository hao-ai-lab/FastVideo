from typing import Any
import torch
from torch import nn
from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _tensor_placeholder(tensor: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """Create a rank-preserving empty placeholder on the specified device."""
    shape = (0, ) if tensor.ndim <= 0 else (0, ) * tensor.ndim
    return torch.empty(shape, device=device, dtype=tensor.dtype)


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

    def _will_offload(self, name: str) -> bool:
        return True

    @torch.compiler.disable
    def on_init(self, module: nn.Module):
        self.module_ref = module
        for name, param in self.module_ref.named_parameters():
            if self._will_offload(name):
                self.cpu_named_parameters[name] = (
                    param.data.detach().to("cpu").pin_memory())
                param.data = _tensor_placeholder(param.data, self.device)

    @torch.compiler.disable
    def wait_and_replace_params(self):
        torch.cuda.current_stream().wait_stream(self.async_copy_stream)
        # now gpu_named_parameters are ready
        for name, param in self.module_ref.named_parameters():
            if not self._will_offload(name):
                continue
            if name not in self.gpu_named_parameters:
                # first load with blocking load
                self.gpu_named_parameters[name] = self.cpu_named_parameters[
                    name].to(self.device)
            param.data = self.gpu_named_parameters[name]

    @torch.compiler.disable
    def prefetch_params(self):
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            for name, param in self.module_ref.named_parameters():
                if not self._will_offload(name):
                    continue
                assert name not in self.gpu_named_parameters
                gpu_param = self.cpu_named_parameters[name].to(
                    self.device, non_blocking=True)
                gpu_param.record_stream(
                    compute_stream
                )  # ensure tensor will not be freed until forward is completed
                self.gpu_named_parameters[name] = gpu_param

    @torch.compiler.disable
    def release_gpu_params(self):
        for name, param in self.module_ref.named_parameters():
            if self._will_offload(name):
                param.data = _tensor_placeholder(param.data, self.device)
                del self.gpu_named_parameters[name]
        assert len(self.gpu_named_parameters) == 0


class LayerwiseOffloadHook(ForwardHook):
    """A hook that enables layerwise CPU offloading during forward pass."""

    def __init__(self, state: LayerwiseOffloadState) -> None:
        self.state = state

    def on_attach(self, module: nn.Module):
        self.state.on_init(
            module)  # pyright: ignore[reportAttributeAccessIssue]

    def name(self) -> str:
        return "LayerwiseOffloadHook"

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        self.state.wait_and_replace_params()  # pyright: ignore
        if self.state.next_state is not None:
            self.state.next_state.prefetch_params()  # pyright: ignore
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any):
        self.state.release_gpu_params()  # pyright: ignore
        return output


def enable_layerwise_offload(model: nn.Module):
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        logger.warning(
            "CUDA is not available. Layerwise offloading is disabled.")
        return
    state_list = []
    async_stream = torch.cuda.Stream()
    for name, submodule in model.named_children():
        if isinstance(submodule, nn.ModuleList):
            for idx, module_entry in enumerate(submodule):
                state = LayerwiseOffloadState(async_copy_stream=async_stream,
                                              device=device)
                state_list.append(state)
                hook_mgr = ModuleHookManager.get_from_or_default(module_entry)
                hook = LayerwiseOffloadHook(state)
                hook_mgr.append_forward_hook(hook)
            break
    if len(state_list) == 0:
        raise ValueError(
            "No nn.ModuleList found in the model for layerwise offloading.")

    # circular linking of states
    for i in range(len(state_list)):
        state_list[i].next_state = state_list[(i + 1) % len(state_list)]
