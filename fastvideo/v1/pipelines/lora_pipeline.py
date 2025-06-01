import logging
from collections import defaultdict
from typing import Dict, Optional

import safetensors
import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.layers.lora.linear import (BaseLayerWithLoRA, get_lora_layer,
                                             replace_submodule)
from fastvideo.v1.models.loader.utils import get_param_names_mapping
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.utils import maybe_download_lora

logger = logging.getLogger(__name__)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convert_to_lora_layers()
        if self.fastvideo_args.lora_path is not None:
            self.set_lora_adapter(
                self.fastvideo_args.lora_nickname,  # type: ignore
                self.fastvideo_args.lora_path)

    def is_target_layer(self, module_name: str) -> bool:
        if self.fastvideo_args.lora_target_names is None:
            return True
        return module_name.split(
            ".")[-1] in self.fastvideo_args.lora_target_names

    def convert_to_lora_layers(self) -> None:
        """
        Converts the transformer to a LoRA transformer.
        """
        for name, layer in self.modules["transformer"].named_modules():
            if not self.is_target_layer(name):
                continue
            layer = get_lora_layer(layer)
            if layer is not None:
                self.lora_layers[name] = layer
                replace_submodule(self.modules["transformer"], name, layer)

    def set_lora_adapter(self,
                         adapter_nickname: str,
                         adapter_path: Optional[str] = None):  # type: ignore
        """
        Loads a LoRA adapter into the pipeline and applies it to the transformer.
        Args:
            adapter_nickname: The "nick name" of the adapter when referenced in the pipeline.
            adapter_path: The path to the adapter, either a local path or a Hugging Face repo id.
        """

        if adapter_nickname not in self.lora_adapters and adapter_path is None:
            raise ValueError(
                f"Adapter {adapter_nickname} not found in the pipeline. Please provide adapter_path to load it."
            )
        adapter_updated = False
        if adapter_path is not None:
            logger.info("Loading LoRA adapter {}", adapter_path)
            lora_local_path = maybe_download_lora(adapter_path)
            lora_state_dict = safetensors.torch.load_file(lora_local_path)
            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"]._param_names_mapping)
            for name, weight in lora_state_dict.items():
                target_name = param_names_mapping_fn(name)
                self.lora_adapters[adapter_nickname][target_name] = weight
                adapter_updated = True

        if not adapter_updated and adapter_nickname == self.cur_adapter_name:
            return

        # Merge the new adapter
        for name, layer in self.lora_layers.items():
            lora_A_name = name + ".lora_A"
            lora_B_name = name + ".lora_B"
            if lora_A_name in self.lora_adapters[adapter_nickname]\
                and lora_B_name in self.lora_adapters[adapter_nickname]:
                if layer.merged:
                    layer.unmerge_lora_weights()
                layer.set_lora_weights(
                    self.lora_adapters[adapter_nickname][lora_A_name],
                    self.lora_adapters[adapter_nickname][lora_B_name],
                    training_mode=self.fastvideo_args.training_mode)
                layer.merge_lora_weights()
            else:
                logger.warning(
                    "LoRA adapter {} does not contain the weights for layer {}. LoRA will not be applied to it.",
                    adapter_path, name)
                layer.disable_lora = True
        logger.info("LoRA adapter {} applied", adapter_path)
        self.cur_adapter_name = adapter_nickname
