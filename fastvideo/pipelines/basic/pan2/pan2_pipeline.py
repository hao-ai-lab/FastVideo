# SPDX-License-Identifier: Apache-2.0
"""PAN2 video generation: text-to-video, and image-to-video from a first frame."""

import contextlib
import importlib
import sys

import torch

from fastvideo import envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.pan2.guardrails import (
    PAN2Guardrails,
    PAN2PromptGuardrailStage,
    PAN2VideoGuardrailStage,
)
from fastvideo.pipelines.basic.pan2.stages import (
    PAN2ConditioningStage,
    PAN2DenoisingStage,
    PAN2InputValidationStage,
    PAN2TextEncodingStage,
    PAN2TimestepPreparationStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import DecodingStage, LatentPreparationStage

logger = init_logger(__name__)

# The modules that read FASTVIDEO_FA4 when they are imported.
_FA4_READERS = ("fastvideo.attention.utils.flash_attn_default", "fastvideo.attention.utils.flash_attn_no_pad")


def flash_attention_4_by_default() -> bool:
    """Whether PAN2 turns FlashAttention-4 on, which on Blackwell GPUs is about 2.3x faster than the SDPA fallback.

    Only when FASTVIDEO_FA4 is unset, the GPU is sm_100 or newer, FA4 imports, and no attention module has read the
    variable yet.
    """
    if envs.FASTVIDEO_FA4.is_set() or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import fastvideo.attention.utils.flash_attn_cute  # noqa: F401
    except ImportError:
        logger.info("PAN2 would use FlashAttention-4 on this GPU, but flash_attn.cute is not installed.")
        return False
    if any(name in sys.modules for name in _FA4_READERS):
        logger.warning("PAN2 defaults to FlashAttention-4 on this GPU, but attention was set up before the pipeline; "
                       "set FASTVIDEO_FA4=1 to use it.")
        return False
    return True


class PAN2Pipeline(ComposedPipelineBase):
    """One stage chain serves both modes: a request with ``image_path`` (or a PIL ``pil_image``) runs image-to-video
    from that first frame, and one without runs text-to-video."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def __init__(self, *args, **kwargs) -> None:
        if not flash_attention_4_by_default():
            super().__init__(*args, **kwargs)
            return
        # FASTVIDEO_FA4 is set only while the pipeline loads: the attention modules read it when they are imported
        # and keep that choice, so other code in the process does not see the variable. Those modules are shared, so a
        # later pipeline in the same process still runs the attention they chose.
        logger.info("PAN2 uses FlashAttention-4 on this GPU; set FASTVIDEO_FA4=0 to turn it off.")
        with envs.FASTVIDEO_FA4.override(True):
            for name in _FA4_READERS:
                # flash_attn_no_pad also needs the flash_attn FA2 package.
                with contextlib.suppress(ImportError):
                    importlib.import_module(name)
            super().__init__(*args, **kwargs)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        pipeline_config = getattr(fastvideo_args, "pipeline_config", None)
        # The guardrails are on unless the config turns them off.
        guardrails = None
        if getattr(pipeline_config, "enable_guardrails", True):
            offload = bool(getattr(pipeline_config, "offload_guardrail_models", False))
            guardrails = PAN2Guardrails(offload_to_cpu=offload)

        self.add_stage(stage_name="input_validation_stage", stage=PAN2InputValidationStage())
        if guardrails is not None:
            self.add_stage(stage_name="prompt_guardrail_stage", stage=PAN2PromptGuardrailStage(guardrails))
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=PAN2TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=PAN2TimestepPreparationStage(scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer"),
                                                    latent_dtype=torch.float32))
        self.add_stage(stage_name="conditioning_stage", stage=PAN2ConditioningStage(vae=self.get_module("vae")))
        self.add_stage(stage_name="denoising_stage",
                       stage=PAN2DenoisingStage(transformer=self.get_module("transformer"),
                                                scheduler=self.get_module("scheduler"),
                                                pipeline=self))
        # The performance switches of PAN2PipelineConfig: the DiT's fused kernels and the VAE's decode kernel.
        transformer = self.get_module("transformer")
        if hasattr(transformer, "set_fused_kernels"):
            transformer.set_fused_kernels(bool(getattr(pipeline_config, "dit_fused_kernels", False)))
        vae = self.get_module("vae")
        vae_config = getattr(vae, "config", None)
        if hasattr(vae_config, "decode_kernel"):
            vae_config.decode_kernel = bool(getattr(pipeline_config, "vae_decode_kernel", False))
            vae_config.decode_kernel_parallel = bool(getattr(pipeline_config, "vae_decode_kernel_parallel", False))
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=vae, pipeline=self))
        if guardrails is not None:
            self.add_stage(stage_name="video_guardrail_stage", stage=PAN2VideoGuardrailStage(guardrails))


class PAN2ModularPipeline(PAN2Pipeline):
    """The pipeline class PAN2 checkpoints declare in their ``modular_model_index.json``."""


EntryClass = [PAN2Pipeline, PAN2ModularPipeline]
