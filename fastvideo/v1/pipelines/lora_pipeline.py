# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, List, Optional

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.layers.lora.linear import (BaseLayerWithLoRA, get_lora_layer,
                                             replace_submodule)
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.utils import get_param_names_mapping
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.utils import maybe_download_lora

logger = init_logger(__name__)


class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports injecting LoRA adapters into the diffusion transformer.
    TODO: support training.
    """
    lora_adapters: Dict[str, Dict[str, torch.Tensor]] = defaultdict(
        dict)  # state dicts of loaded lora adapters
    cur_adapter_name: str = ""
    lora_layers: Dict[str, BaseLayerWithLoRA] = {}
    fastvideo_args: FastVideoArgs
    exclude_lora_layers: List[str] = []
    device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude_lora_layers = self.modules[
            "transformer"].config.arch_config.exclude_lora_layers

        self.convert_to_lora_layers()
        if self.fastvideo_args.pipeline_config.lora_path is not None:
            self.set_lora_adapter(
                self.fastvideo_args.pipeline_config.
                lora_nickname,  # type: ignore
                self.fastvideo_args.pipeline_config.lora_path)

    def is_target_layer(self, module_name: str) -> bool:
        if self.fastvideo_args.pipeline_config.lora_target_names is None:
            return True
        return any(target_name in module_name for target_name in
                   self.fastvideo_args.pipeline_config.lora_target_names)

    def convert_to_lora_layers(self) -> None:
        """
        Converts the transformer to a LoRA transformer.
        """

        for name, layer in self.modules["transformer"].named_modules():
            if not self.is_target_layer(name):
                continue

            excluded = False
            for exclude_layer in self.exclude_lora_layers:
                if exclude_layer in name:
                    excluded = True
                    break
            if excluded:
                continue

            layer = get_lora_layer(layer)
            if layer is not None:
                self.lora_layers[name] = layer
                replace_submodule(self.modules["transformer"], name, layer)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: Optional[str] = None):  # type: ignore
        """
        Loads a LoRA adapter into the pipeline and applies it to the transformer.
        Args:
            lora_nickname: The "nick name" of the adapter when referenced in the pipeline.
            lora_path: The path to the adapter, either a local path or a Hugging Face repo id.
        """

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        adapter_updated = False
        rank = dist.get_rank()
        if lora_path is not None:
            lora_local_path = maybe_download_lora(lora_path)
            lora_state_dict = load_file(lora_local_path)
            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"]._param_names_mapping)
            lora_param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"]._lora_param_names_mapping)

            to_merge_params: DefaultDict[Hashable,
                                         Dict[Any, Any]] = defaultdict(dict)
            for name, weight in lora_state_dict.items():
                name = ".".join(
                    name.split(".")
                    [1:-1])  # remove the transformer prefix and .weight suffix
                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                    name)
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
                self.lora_adapters[lora_nickname][target_name] = weight.to(
                    self.device)
            adapter_updated = True
            logger.info("Rank %d: loaded LoRA adapter %s", rank, lora_path)

        if not adapter_updated and lora_nickname == self.cur_adapter_name:
            return

        # Merge the new adapter
        adapted_count = 0
        for name, layer in self.lora_layers.items():
            lora_A_name = name + ".lora_A"
            lora_B_name = name + ".lora_B"
            if lora_A_name in self.lora_adapters[lora_nickname]\
                and lora_B_name in self.lora_adapters[lora_nickname]:
                if layer.merged:
                    layer.unmerge_lora_weights()
                layer.set_lora_weights(
                    self.lora_adapters[lora_nickname][lora_A_name],
                    self.lora_adapters[lora_nickname][lora_B_name],
                    training_mode=self.fastvideo_args.training_mode)
                adapted_count += 1
            else:
                if rank == 0:
                    logger.warning(
                        "LoRA adapter %s does not contain the weights for layer %s. LoRA will not be applied to it.",
                        lora_path, name)
                layer.disable_lora = True
        logger.info("Rank %d: LoRA adapter %s applied to %d layers", rank,
                    lora_path, adapted_count)
        self.cur_adapter_name = lora_nickname
