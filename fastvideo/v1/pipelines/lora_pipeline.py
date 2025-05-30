from typing import Dict
import torch
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.models.loader.lora_load import load_lora_adapter
from fastvideo.v1.layers.lora.linear import BaseLayerWithLoRA, get_lora_layer

class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports LoRA adapters. Currently only supports inference.
    """
    lora_adapters: Dict[str, Dict[str, torch.Tensor]] = {} # state dicts of loaded lora adapters
    cur_adapter_name: str = ""
    lora_layers: Dict[str, BaseLayerWithLoRA] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convert_to_lora_layers()
        if self.fastvideo_args.lora_path is not None:
            self.set_lora_adapter(self.fastvideo_args.lora_nick_name, self.fastvideo_args.lora_path)

    def convert_to_lora_layers(self):
        """
        Converts the transformer to a LoRA transformer.
        """
        for name, layer in self.modules["transformer"].named_modules():
            if get_lora_layer(layer) is not None:

    def set_lora_adapter(self, adapter_nick_name: str, adapter_path: str = None):
        """
        Loads a LoRA adapter into the pipeline and applies it to the transformer.
        Args:
            adapter_nick_name: The "nick name" of the adapter when referenced in the pipeline.
            adapter_path: The path to the adapter, either a local path or a Hugging Face repo id.
        """
        if adapter_nick_name not in self.lora_adapters and adapter_path is None:
            raise ValueError(f"Adapter {adapter_nick_name} not found in the pipeline. Please provide adapter_path to load it.")
        if adapter_path is not None:
            self.lora_adapters[adapter_nick_name] = load_lora_adapter(self.modules["transformer"], adapter_path)

        # Merge the new adapter
        
    
