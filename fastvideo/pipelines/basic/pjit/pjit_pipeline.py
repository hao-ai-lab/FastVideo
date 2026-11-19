# SPDX-License-Identifier: Apache-2.0
"""Load Linum's original pixel DiT and its separately published Qwen3.5 text tower."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.attention.selector import record_resolved_attention_backend
from fastvideo.configs.models.dits.pjit import PJiTArchConfig, PJiTConfig
from fastvideo.configs.pipelines.pjit import PJiTPipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.pjit import PyramidJiT
from fastvideo.models.loader.component_loader import TextEncoderLoader, TokenizerLoader
from fastvideo.pipelines.basic.pjit.stages import PJiTDenoisingStage, PJiTTextEncodingStage
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lazy_module import LazyModule
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


def _load_transformer(path: Path, config: PJiTConfig, device: torch.device) -> PyramidJiT:
    # Allocate only the checkpoint's master weights. Casting those weights to bf16 before
    # the AdaLN and RMSNorm operations changes the model even under bf16 matmul autocast.
    with torch.device("meta"):
        model = PyramidJiT(config=config, hf_config={})
    state = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            raise ValueError(f"P-JiT checkpoint parameter {name} must be fp32, got {param.dtype}.")
    return model.to(device).eval().requires_grad_(False)


def _load_qwen(path: str, args: FastVideoArgs):
    config = deepcopy(args.pipeline_config.text_encoder_configs[0])
    with (Path(path) / "config.json").open() as handle:
        hf = json.load(handle)
    # The Qwen release bundles a vision encoder and language-model head. All DiT conditioning
    # comes from the text tower, whose geometry is under text_config in the raw checkpoint.
    config.update_model_arch({**hf["text_config"], "architectures": ["Qwen3_5TextModel"]})
    record_resolved_attention_backend(config)
    return TextEncoderLoader().load_model(path,
                                          config,
                                          get_local_torch_device(),
                                          args,
                                          dtype="bf16",
                                          use_text_encoder_override=True)


class PJiTPipeline(ComposedPipelineBase):
    model_path: str
    pipeline_config_cls = PJiTPipelineConfig
    sampling_params_cls = SamplingParam
    _required_config_modules = ["text_encoder", "tokenizer", "transformer"]
    _lazy_module_names = ("text_encoder", "transformer")

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str]:
        return ["config.json", "model.safetensors"]

    def load_modules(self, fastvideo_args: FastVideoArgs, loaded_modules: dict | None = None) -> dict[str, Any]:
        args = fastvideo_args
        if args.num_gpus != 1 or args.sp_size != 1 or args.tp_size != 1 or args.use_fsdp_inference:
            raise ValueError("P-JiT supports single-device inference with sp_size=1, tp_size=1 and FSDP disabled.")
        if args.lora_path is not None:
            raise ValueError("P-JiT runs the released base transformer and does not support a LoRA adapter.")
        if args.pipeline_config.dit_config.quant_config is not None:
            raise ValueError("P-JiT requires fp32 master weights and does not support transformer quantization.")
        if args.dit_layerwise_offload:
            logger.warning("P-JiT uses whole-component DiT offload for this checkpoint.")
            args.dit_layerwise_offload = False
            args.dit_cpu_offload = True
        config = args.pipeline_config
        if config.dit_precision != "fp32" or tuple(config.text_encoder_precisions) != ("bf16", ):
            raise ValueError("P-JiT requires fp32 transformer weights and a bf16 Qwen text encoder.")
        if config.noise_sm_count is not None and config.noise_sm_count < 1:
            raise ValueError("noise_sm_count must be a positive SM count or None.")
        self.model_path = maybe_download_model(self.model_path,
                                               revision=args.revision,
                                               allow_patterns=self.get_hf_download_allow_patterns())
        root = Path(self.model_path)
        with (root / "config.json").open() as handle:
            config.dit_config.arch_config = PJiTArchConfig(**json.load(handle)["architecture"])
        record_resolved_attention_backend(config.dit_config)
        modules = dict(loaded_modules or {})
        lazy = self._lazy_module_load_enabled(args)

        def add_component(name: str, load: Callable[[], Any]) -> None:
            if name not in modules:
                modules[name] = LazyModule(name, load) if lazy and name in self._lazy_module_names else load()

        if "text_encoder" not in modules or "tokenizer" not in modules:
            qwen = maybe_download_model(config.qwen_model_path,
                                        revision=config.qwen_revision,
                                        allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt"])
            add_component("tokenizer", partial(TokenizerLoader().load, qwen, args))
            add_component("text_encoder", partial(_load_qwen, qwen, args))
        device = torch.device("cpu") if args.dit_cpu_offload and not lazy else get_local_torch_device()
        add_component("transformer", partial(_load_transformer, root, config.dit_config, device))
        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage("text_encoding_stage",
                       PJiTTextEncodingStage(self.get_module("text_encoder"), self.get_module("tokenizer")))
        self.add_stage("denoising_stage", PJiTDenoisingStage(self.get_module("transformer")))


EntryClass = PJiTPipeline
