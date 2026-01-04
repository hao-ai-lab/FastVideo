# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang's layerwise offload implementation:
# https://github.com/sgl-project/sglang/pull/15511
#
# This implementation provides a lightweight layerwise CPU offload manager
# with async H2D prefetch using a dedicated CUDA stream, following SGLang's design.

import re
from contextlib import contextmanager
from typing import Dict, Set, Optional, Tuple

import torch


class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.

    Offloads per-layer parameters/buffers from GPU to CPU, and supports async H2D
    prefetch using a dedicated CUDA stream.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        module_list_attr: str,
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        auto_initialize: bool = False,
    ) -> None:
        self.model = model
        self.module_list_attr = module_list_attr
        self.num_layers = int(num_layers)
        self.pin_cpu_memory = bool(pin_cpu_memory)

        self.enabled = bool(enabled and torch.cuda.is_available())
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if self.enabled else None
        )
        self.copy_stream = torch.cuda.Stream() if self.enabled else None

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(module_list_attr)}\.(\d+)(\.|$)"
        )

        self._cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cpu_dtypes: Dict[int, Dict[str, torch.dtype]] = {}

        self._gpu_layers: Dict[int, Set[str]] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}

        self._meta: Dict[str, Tuple[int, torch.dtype]] = {}

        if auto_initialize:
            self.initialize()

    def _match_layer_idx(self, name: str) -> Optional[int]:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    def _record_meta(self, name: str, t: torch.Tensor) -> None:
        if name not in self._meta:
            self._meta[name] = (int(t.ndim), t.dtype)

    def _make_placeholder(self, name: str) -> torch.Tensor:
        """Rank-preserving empty placeholder on GPU."""
        assert self.device is not None
        ndim, dtype = self._meta[name]
        shape = (0,) if ndim <= 0 else (0,) * ndim
        return torch.empty(shape, device=self.device, dtype=dtype)

    def _get_target(self, name: str) -> torch.Tensor:
        if name in self._named_parameters:
            return self._named_parameters[name]
        return self._named_buffers[name]

    def _offload_tensor(self, name: str, tensor: torch.Tensor, layer_idx: int) -> None:
        if layer_idx not in self._cpu_weights:
            self._cpu_weights[layer_idx] = {}
            self._cpu_dtypes[layer_idx] = {}

        self._record_meta(name, tensor)

        cpu_weight = tensor.detach().to("cpu")
        if self.pin_cpu_memory:
            cpu_weight = cpu_weight.pin_memory()

        self._cpu_weights[layer_idx][name] = cpu_weight
        self._cpu_dtypes[layer_idx][name] = tensor.dtype

        if self.device is not None:
            tensor.data = self._make_placeholder(name)

    @torch.compiler.disable
    def initialize(self) -> None:
        """Offload all matched layer tensors to CPU and prefetch layer 0 (sync)."""
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        for name, param in self._named_parameters.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            self._offload_tensor(name, param, layer_idx)

        for name, buf in self._named_buffers.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            self._offload_tensor(name, buf, layer_idx)

        self.prefetch_layer(0, non_blocking=False)
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """Prefetch a layer's tensors from CPU to GPU (async on copy_stream)."""
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if layer_idx not in self._cpu_weights:
            return

        self.copy_stream.wait_stream(torch.cuda.current_stream())

        param_names: Set[str] = set()
        with torch.cuda.stream(self.copy_stream):
            for name, cpu_weight in self._cpu_weights[layer_idx].items():
                target = self._get_target(name)

                gpu_weight = torch.empty(
                    cpu_weight.shape,
                    dtype=self._cpu_dtypes[layer_idx][name],
                    device=self.device,
                )
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)

                target.data = gpu_weight
                param_names.add(name)

        self._gpu_layers[layer_idx] = param_names

    @contextmanager
    def layer_scope(
        self,
        *,
        prefetch_layer_idx: Optional[int],
        release_layer_idx: Optional[int],
        non_blocking: bool = True,
    ):
        if self.enabled and release_layer_idx is not None:
            cur = release_layer_idx
            if (
                cur not in self._gpu_layers
                and cur in self._cpu_weights
                and self.device is not None
                and self.copy_stream is not None
            ):
                self.prefetch_layer(cur, non_blocking=False)
                torch.cuda.current_stream().wait_stream(self.copy_stream)

        if self.enabled and prefetch_layer_idx is not None:
            self.prefetch_layer(prefetch_layer_idx, non_blocking=non_blocking)

        try:
            yield
        finally:
            if self.enabled and self.copy_stream is not None:
                torch.cuda.current_stream().wait_stream(self.copy_stream)
            if self.enabled and release_layer_idx is not None:
                self.release_layer(release_layer_idx)


    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        """Release a layer's tensors back to placeholders (free VRAM)."""
        if not self.enabled or self.device is None:
            return

        if layer_idx < 0:
            return

        param_names = self._gpu_layers.pop(layer_idx, None)
        if not param_names:
            return

        for name in param_names:
            target = self._get_target(name)
            # Ensure meta exists even if something unexpected happened
            self._record_meta(name, target)
            target.data = self._make_placeholder(name)

    @torch.compiler.disable
    def release_all(self) -> None:
        """Release all currently-resident layers back to placeholders."""
        if not self.enabled or self.device is None:
            return

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers.keys()):
            param_names = self._gpu_layers.pop(layer_idx, None)
            if not param_names:
                continue
            for name in param_names:
                target = self._get_target(name)
                self._record_meta(name, target)
                target.data = self._make_placeholder(name)
