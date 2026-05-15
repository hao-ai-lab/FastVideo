"""Vendored GLM-ASR architecture for transformers 4.57.

Upstream lives in transformers ≥5.0 (``transformers.models.glmasr``)
and the HF repo ``zai-org/GLM-ASR-Nano-2512`` ships no remote modeling
code. This package vendors just enough of the modeling/processing/
config code to make ``AutoModel.from_pretrained(...)`` work against
fastvideo's pinned transformers. Apache-2.0 (see ``LICENSE``).

Call :func:`register_with_auto` once before ``AutoModel.from_pretrained``.
"""

from __future__ import annotations

from .configuration_glmasr import GlmAsrConfig, GlmAsrEncoderConfig
from .modeling_glmasr import GlmAsrEncoder, GlmAsrForConditionalGeneration
from .processing_glmasr import GlmAsrProcessor

__all__ = [
    "GlmAsrConfig",
    "GlmAsrEncoderConfig",
    "GlmAsrEncoder",
    "GlmAsrForConditionalGeneration",
    "GlmAsrProcessor",
    "register_with_auto",
]


def register_with_auto() -> None:
    """Register the vendored classes with transformers' Auto* tables. Idempotent."""
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    CONFIG_MAPPING.register("glmasr_encoder", GlmAsrEncoderConfig, exist_ok=True)
    CONFIG_MAPPING.register("glmasr", GlmAsrConfig, exist_ok=True)
    AutoConfig.register("glmasr_encoder", GlmAsrEncoderConfig, exist_ok=True)
    AutoConfig.register("glmasr", GlmAsrConfig, exist_ok=True)
    # GlmAsrForConditionalGeneration.__init__ instantiates the audio tower
    # via AutoModel.from_config(config.audio_config), so both must register.
    AutoModel.register(GlmAsrConfig, GlmAsrForConditionalGeneration, exist_ok=True)
    AutoModel.register(GlmAsrEncoderConfig, GlmAsrEncoder, exist_ok=True)
    AutoProcessor.register(GlmAsrConfig, GlmAsrProcessor, exist_ok=True)
