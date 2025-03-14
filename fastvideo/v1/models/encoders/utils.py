# SPDX-License-Identifier: Apache-2.0

import itertools
from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, Union, overload)

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig

from fastvideo.v1.logger import init_logger
from fastvideo.v1.loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        return ((out_name, data) for name, data in weights
                if (out_name := self._map_name(name)) is not None)


class AutoWeightsLoader:
    """
    Helper class to load weights into a :class:`torch.nn.Module`. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.

    Detailed weight loading information can be viewed by setting the
    environment variable ``VLLM_LOGGING_LEVEL=DEBUG``.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: Optional[List[str]] = None,
        ignore_unexpected_prefixes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []

    def _groupby_prefix(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Iterable[Tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts,
                                               key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (("" if len(parts) == 1 else parts[1], weights_data)
                 for parts, weights_data in group),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes)

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'")

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname,
                         param.shape)

            yield weight_qualname

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters "
                        "for module %s", module)
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(prefix,
                                             child_modules[child_prefix],
                                             child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                msg = (f"There is no module or parameter named '{prefix}' "
                       f"in {type(self.module).__name__}")
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> Set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor:
    ...


@overload
def flatten_bn(x: List[torch.Tensor]) -> List[torch.Tensor]:
    ...


@overload
def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: Literal[True],
) -> torch.Tensor:
    ...


@overload
def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    ...


def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, ...)```.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


class LayerFn(Protocol):

    def __call__(self, prefix: str) -> torch.nn.Module:
        ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes


def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    device = next(module.parameters()).device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = True

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty_strided(size=p.data.size(),
                                       stride=p.data.stride(),
                                       dtype=p.data.dtype,
                                       layout=p.data.layout,
                                       device='cpu',
                                       pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        p.data = cpu_data
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }
            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
            module.forward = forward
            return output

        module.forward = forward

    return module


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def extract_layer_index(layer_name: str) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: List[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (f"layer name {layer_name} should"
                                " only contain one integer")
    return int_vals[0]


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors

def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> torch.nn.ModuleList:
    """Make a list of layers with the given layer function.
    
    Args:
        num_hidden_layers: Number of layers to create.
        layer_fn: Function that creates a layer given a prefix.
        prefix: Prefix for layer names.
        
    Returns:
        A ModuleList containing all layers.
    """
    modules = torch.nn.ModuleList([
        maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
        for idx in range(num_hidden_layers)
    ])
    return modules


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"
