# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Hashable
from contextlib import nullcontext, contextmanager
from typing import Any
from collections.abc import Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.hooks.layerwise_offload import LayerwiseOffloadHook
from fastvideo.layers.lora.linear import (
    BaseLayerWithLoRA,
    get_lora_layer,
    replace_submodule,
)
from fastvideo.logger import init_logger
from fastvideo.models.loader.utils import get_param_names_mapping
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.utils import maybe_download_lora

logger = init_logger(__name__)


def _get_hook_ctx(module: nn.Module | None):
    if module is None:
        return nullcontext()
    hook_mgr = ModuleHookManager.get_from(module)
    if hook_mgr is not None:
        offload_hook = hook_mgr.forward_hooks.get(LayerwiseOffloadHook.name())
        if offload_hook is not None:
            return offload_hook.mutate_params_scope()  # type: ignore
    return nullcontext()


def _named_module_by_prefix(
    module: nn.Module, prefixes: list[str]
) -> list[tuple[str | None, list[tuple[str, nn.Module]]]]:
    none_list: list[tuple[str, nn.Module]] = []
    prefix_list: list[tuple[str, list[tuple[str, nn.Module]]]] = [
        (prefix, []) for prefix in prefixes
    ]
    for name, submodule in module.named_modules():
        for cur_prefix, cur_list in prefix_list:
            # we should exclude e.g. block.1 and block.12.attn
            if name.startswith(cur_prefix + "."):
                cur_list.append((name, submodule))
                break
        else:
            none_list.append((name, submodule))
    return prefix_list + [(None, none_list)]  # type: ignore


class LoRAModelLayers:

    def __init__(self, block_list: list[tuple[str, nn.Module]]) -> None:
        # block_name -> {layer_name -> layer}
        self.block_to_lora_layers: dict[str, dict[str, BaseLayerWithLoRA]] = {}
        # layer_name -> block_name
        self.lora_layers_to_block: dict[str, str | None] = {}
        self.other_lora_layers: dict[str, BaseLayerWithLoRA] = {}
        self.block_mapping = dict(block_list)

    def add_lora_layer(self, block_name: str | None, layer_name: str,
                       layer: BaseLayerWithLoRA):
        if block_name is None:
            self.other_lora_layers[layer_name] = layer
            self.lora_layers_to_block[layer_name] = None
        else:
            if block_name not in self.block_to_lora_layers:
                self.block_to_lora_layers[block_name] = {}
            self.block_to_lora_layers[block_name][layer_name] = layer
            self.lora_layers_to_block[layer_name] = block_name

    def all_lora_layers(
        self, ) -> Generator[tuple[str, BaseLayerWithLoRA], Any, None]:
        for block_layers in self.block_to_lora_layers.values():
            for name, layer in block_layers.items():
                yield name, layer
        for name, layer in self.other_lora_layers.items():
            yield name, layer

    def lora_layers_by_block(
        self,
    ) -> Generator[
            tuple[nn.Module | None, dict[str, BaseLayerWithLoRA]],
            Any,
            None,
    ]:
        for block_name, layers in self.block_to_lora_layers.items():
            yield self.block_mapping[block_name], layers
        yield None, self.other_lora_layers


class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports injecting LoRA adapters into the diffusion transformer.
    TODO: support training.
    """

    lora_adapters: dict[str, dict[str, torch.Tensor]] = defaultdict(
        dict
    )  # state dicts of loaded lora adapters (includes lora_A, lora_B, and lora_alpha)
    cur_adapter_name: str = ""
    cur_adapter_path: str = ""
    # model_name -> layers
    lora_layers: dict[str, LoRAModelLayers] = {}
    fastvideo_args: FastVideoArgs | TrainingArgs
    exclude_lora_layers: dict[str, list[str]] = {}
    device: torch.device = get_local_torch_device()
    lora_target_modules: list[str] | None = None
    lora_path: str | None = None
    lora_nickname: str = "default"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_initialized: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_local_torch_device()
        # build list of trainable transformers
        for transformer_name in self.trainable_transformer_names:
            if (transformer_name in self.modules
                    and self.modules[transformer_name] is not None):
                self.trainable_transformer_modules[transformer_name] = (
                    self.modules[transformer_name])
            # check for transformer_2 in case of Wan2.2 MoE or fake_score_transformer_2
            if transformer_name.endswith("_2"):
                raise ValueError(
                    f"trainable_transformer_name override in pipelines should not include _2 suffix: {transformer_name}"
                )

            secondary_transformer_name = transformer_name + "_2"
            if (secondary_transformer_name in self.modules
                    and self.modules[secondary_transformer_name] is not None):
                self.trainable_transformer_modules[
                    secondary_transformer_name] = self.modules[
                        secondary_transformer_name]

        logger.info(
            "trainable_transformer_modules: %s",
            self.trainable_transformer_modules.keys(),
        )

        for (
                transformer_name,
                transformer_module,
        ) in self.trainable_transformer_modules.items():
            self.exclude_lora_layers[transformer_name] = (
                transformer_module.config.arch_config.exclude_lora_layers)
        self.lora_target_modules = self.fastvideo_args.lora_target_modules
        self.lora_path = self.fastvideo_args.lora_path
        self.lora_nickname = self.fastvideo_args.lora_nickname
        self.training_mode = self.fastvideo_args.training_mode
        if self.training_mode and getattr(self.fastvideo_args, "lora_training",
                                          False):
            assert isinstance(self.fastvideo_args, TrainingArgs)
            if self.fastvideo_args.lora_alpha is None:
                self.fastvideo_args.lora_alpha = self.fastvideo_args.lora_rank
            self.lora_rank = self.fastvideo_args.lora_rank  # type: ignore
            self.lora_alpha = self.fastvideo_args.lora_alpha  # type: ignore
            logger.info(
                "Using LoRA training with rank %d and alpha %d",
                self.lora_rank,
                self.lora_alpha,
            )
            if self.lora_target_modules is None:
                self.lora_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out",
                    "to_qkv",
                    "to_gate_compress",
                ]
                logger.info(
                    "Using default lora_target_modules for all transformers: %s",
                    self.lora_target_modules,
                )
            else:
                logger.warning(
                    "Using custom lora_target_modules for all transformers, which may not be intended: %s",
                    self.lora_target_modules,
                )

            self.convert_to_lora_layers()
        # Inference
        elif not self.training_mode and self.lora_path is not None:
            self.convert_to_lora_layers()
            self.set_lora_adapter(
                self.lora_nickname,  # type: ignore
                self.lora_path,
            )  # type: ignore

    def is_target_layer(self, module_name: str) -> bool:
        if self.lora_target_modules is None:
            return True
        return any(target_name in module_name
                   for target_name in self.lora_target_modules)

    def set_trainable(self) -> None:

        def set_lora_grads(lora_layers: LoRAModelLayers,
                           device_mesh: DeviceMesh):
            for name, layer in lora_layers.all_lora_layers():
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)
                layer.base_layer.requires_grad_(False)
                layer.lora_A = nn.Parameter(
                    DTensor.from_local(layer.lora_A, device_mesh=device_mesh))
                layer.lora_B = nn.Parameter(
                    DTensor.from_local(layer.lora_B, device_mesh=device_mesh))

        is_lora_training = self.training_mode and getattr(
            self.fastvideo_args, "lora_training", False)
        if not is_lora_training:
            super().set_trainable()
            return

        device_mesh = init_device_mesh(
            "cuda",
            (dist.get_world_size(), 1),
            mesh_dim_names=["fake", "replicate"],
        )
        for (
                transformer_name,
                transformer_module,
        ) in self.trainable_transformer_modules.items():
            transformer_module.train()
            transformer_module.requires_grad_(False)
            if transformer_name in self.lora_layers:
                set_lora_grads(self.lora_layers[transformer_name], device_mesh)
            else:
                raise ValueError(
                    f"Transformer {transformer_name} should be trainable but not found in lora_layers"
                )

    def convert_to_lora_layers(self) -> None:
        """
        Unified method to convert the transformer to a LoRA transformer.
        """
        if self.lora_initialized:
            return
        self.lora_initialized = True
        for (
                transformer_name,
                transformer_module,
        ) in self.trainable_transformer_modules.items():
            converted_count = 0
            # init bookkeeping structures
            if transformer_name not in self.lora_layers:
                # get block list
                block_list = []
                for name, submodule in transformer_module.named_children():
                    if isinstance(submodule, nn.ModuleList):
                        block_list = [(f"{name}.{i}", m)
                                      for i, m in enumerate(submodule)]
                        break
                self.lora_layers[transformer_name] = LoRAModelLayers(block_list)
            logger.info("Converting %s to LoRA Transformer", transformer_name)
            # scan every module and convert to LoRA layer if applicable

            for block_name, block_modules in _named_module_by_prefix(
                    transformer_module,
                    list(self.lora_layers[transformer_name].block_mapping),
            ):
                if block_name is not None and (
                        not self.fastvideo_args.training_mode
                        and self.fastvideo_args.dit_layerwise_offload):
                    scope_ctx = _get_hook_ctx(
                        self.lora_layers[transformer_name].
                        block_mapping[block_name])
                else:
                    scope_ctx = nullcontext()
                with scope_ctx:
                    for name, layer in block_modules:
                        if not self.is_target_layer(name):
                            continue

                        excluded = False
                        for exclude_layer in self.exclude_lora_layers[
                                transformer_name]:
                            if exclude_layer in name:
                                excluded = True
                                break
                        if excluded:
                            continue

                        layer = get_lora_layer(
                            layer,
                            lora_rank=self.lora_rank,
                            lora_alpha=self.lora_alpha,
                            training_mode=self.training_mode,
                        )
                        if layer is not None:
                            block_name_split = name.split(".", 2)
                            if len(block_name_split) > 2:
                                block_name = (block_name_split[0] + "." +
                                              block_name_split[1])
                            else:
                                block_name = None
                            if (block_name
                                    not in self.lora_layers[transformer_name].
                                    block_mapping):
                                block_name = None
                            self.lora_layers[transformer_name].add_lora_layer(
                                block_name, name, layer)
                            replace_submodule(transformer_module, name, layer)
                            converted_count += 1
            logger.info("Converted %d layers to LoRA layers", converted_count)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None):  # type: ignore
        """
        Load a LoRA adapter into the pipeline and merge it into the transformer.
        Args:
            lora_nickname: The "nick name" of the adapter when referenced in the pipeline.
            lora_path: The path to the adapter, either a local path or a Hugging Face repo id.
        """

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        if not self.lora_initialized:
            self.convert_to_lora_layers()
        adapter_updated = False
        rank = dist.get_rank()
        if lora_path is not None and lora_path != self.cur_adapter_path:
            lora_local_path = maybe_download_lora(lora_path)
            lora_state_dict = load_file(lora_local_path)

            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].param_names_mapping)
            lora_param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].lora_param_names_mapping)

            # Extract alpha values and weights in a single pass
            to_merge_params: defaultdict[Hashable,
                                         dict[Any, Any]] = (defaultdict(dict))
            for name, weight in lora_state_dict.items():
                # Extract weights (lora_A, lora_B, and lora_alpha)
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")

                if "lora_alpha" in name:
                    # Store alpha with minimal mapping - same processing as lora_A/lora_B
                    # but store in lora_adapters with ".lora_alpha" suffix
                    layer_name = name.replace(".lora_alpha", "")
                    layer_name, _, _ = lora_param_names_mapping_fn(layer_name)
                    target_name, _, _ = param_names_mapping_fn(layer_name)
                    # Store alpha alongside weights with same target_name base
                    alpha_key = target_name + ".lora_alpha"
                    self.lora_adapters[lora_nickname][alpha_key] = (
                        weight.item()
                        if weight.numel() == 1 else float(weight.mean()))
                    continue

                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = (
                    param_names_mapping_fn(name))
                # for (in_dim, r) @ (r, out_dim), we only merge (r, out_dim * n) where n is the number of linear layers to fuse
                # see param mapping in HunyuanVideoArchConfig
                if merge_index is not None and "lora_B" in name:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        # cat at output dim according to the merge_index order
                        sorted_tensors = [
                            to_merge_params[target_name][i]
                            for i in range(num_params_to_merge)
                        ]
                        weight = torch.cat(sorted_tensors, dim=1)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in self.lora_adapters[lora_nickname]:
                    raise ValueError(
                        f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(
                    self.device)
            adapter_updated = True
            self.cur_adapter_path = lora_path
            logger.info("Rank %d: loaded LoRA adapter %s", rank, lora_path)

        if not adapter_updated and self.cur_adapter_name == lora_nickname:
            return
        self.cur_adapter_name = lora_nickname

        # Merge the new adapter
        adapted_count = 0
        for (
                transformer_name,
                transformer_lora_layers,
        ) in self.lora_layers.items():
            for (
                    module,
                    layers,
            ) in transformer_lora_layers.lora_layers_by_block():
                with _get_hook_ctx(module):
                    for name, layer in layers.items():
                        lora_A_name = name + ".lora_A"
                        lora_B_name = name + ".lora_B"
                        lora_alpha_name = name + ".lora_alpha"
                        if (lora_A_name in self.lora_adapters[lora_nickname]
                                and lora_B_name
                                in self.lora_adapters[lora_nickname]):
                            # Get alpha value for this layer (defaults to None if not present)
                            lora_A = self.lora_adapters[lora_nickname][
                                lora_A_name]
                            lora_B = self.lora_adapters[lora_nickname][
                                lora_B_name]
                            # Simple lookup - alpha stored with same naming scheme as lora_A/lora_B
                            alpha = (self.lora_adapters[lora_nickname].get(
                                lora_alpha_name) if adapter_updated else None)
                            try:
                                layer.set_lora_weights(
                                    lora_A,
                                    lora_B,
                                    lora_alpha=alpha,
                                    training_mode=self.fastvideo_args.
                                    training_mode,
                                    lora_path=lora_path,
                                )
                            except Exception as e:
                                logger.error(
                                    "Error setting LoRA weights for layer %s: %s",
                                    name,
                                    str(e),
                                )
                                raise e
                            adapted_count += 1
                        else:
                            if rank == 0:
                                logger.warning(
                                    "LoRA adapter %s does not contain the weights for layer %s. LoRA will not be applied to it.",
                                    lora_path,
                                    name,
                                )
                            layer.disable_lora = True
        logger.info(
            "Rank %d: LoRA adapter %s applied to %d layers",
            rank,
            lora_path,
            adapted_count,
        )

    def merge_lora_weights(self) -> None:
        for (
                transformer_name,
                transformer_lora_layers,
        ) in self.lora_layers.items():
            for (
                    module,
                    layers,
            ) in transformer_lora_layers.lora_layers_by_block():
                with _get_hook_ctx(module):
                    for name, layer in layers.items():
                        layer.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        for (
                transformer_name,
                transformer_lora_layers,
        ) in self.lora_layers.items():
            for (
                    module,
                    layers,
            ) in transformer_lora_layers.lora_layers_by_block():
                with _get_hook_ctx(module):
                    for name, layer in layers.items():
                        layer.unmerge_lora_weights()

    @contextmanager
    def disable_adapter(self):
        """
        Context manager to temporarily disable LoRA adapters.
        This is used for computing reference model outputs in GRPO training.
        Aligned with flow_grpo's PeftModel.disable_adapter().
        """
        # Store original disable_lora states
        original_states = {}
        for (
                transformer_name,
                transformer_lora_layers,
        ) in self.lora_layers.items():
            for layer_name, layer in transformer_lora_layers.all_lora_layers():
                original_states[(transformer_name, layer_name)] = layer.disable_lora
                layer.disable_lora = True
        
        try:
            yield
        finally:
            # Restore original disable_lora states
            for (
                    transformer_name,
                    transformer_lora_layers,
            ) in self.lora_layers.items():
                for layer_name, layer in transformer_lora_layers.all_lora_layers():
                    layer.disable_lora = original_states[(transformer_name, layer_name)]
