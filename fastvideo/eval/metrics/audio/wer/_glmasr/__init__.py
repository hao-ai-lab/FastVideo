"""Vendored GLM-ASR architecture for transformers 4.57.

Upstream lives in transformers >=5.0 under ``transformers.models.glmasr``.
4.57 doesn't ship the architecture, the HF repo
``zai-org/GLM-ASR-Nano-2512`` doesn't include remote modeling code, so
``trust_remote_code`` cannot rescue the load. This package vendors just
enough of upstream's modeling/processing/config code to make
``AutoModel.from_pretrained(...)`` work against fastvideo's pinned
transformers — without the much wider blast radius of bumping
transformers itself.

The directory name is leading-underscore so the eval auto-discovery
walker (in ``fastvideo/eval/metrics/__init__.py``) skips it; this
package is loaded only when
``fastvideo.eval.metrics.audio.wer.metric._setup_glm_asr`` runs.

Public API:
- :func:`register_with_auto` — idempotent registration of the vendored
  config / model / processor classes against transformers' Auto* tables
  so the standard ``AutoModel.from_pretrained`` flow finds them.
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
    """Register the vendored classes with transformers' Auto* tables.

    Idempotent — safe to call multiple times. After this returns,
    ``AutoConfig.from_pretrained``, ``AutoModel.from_pretrained``, and
    ``AutoProcessor.from_pretrained`` all recognize ``model_type=glmasr``
    (and the ``glmasr_encoder`` audio-tower sub-config).
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    CONFIG_MAPPING.register("glmasr_encoder", GlmAsrEncoderConfig, exist_ok=True)
    CONFIG_MAPPING.register("glmasr", GlmAsrConfig, exist_ok=True)
    AutoConfig.register("glmasr_encoder", GlmAsrEncoderConfig, exist_ok=True)
    AutoConfig.register("glmasr", GlmAsrConfig, exist_ok=True)
    # AutoModel needs both: top-level model + the audio-tower sub-model
    # that GlmAsrForConditionalGeneration.__init__ instantiates via
    # AutoModel.from_config(config.audio_config).
    AutoModel.register(GlmAsrConfig, GlmAsrForConditionalGeneration, exist_ok=True)
    AutoModel.register(GlmAsrEncoderConfig, GlmAsrEncoder, exist_ok=True)
    AutoProcessor.register(GlmAsrConfig, GlmAsrProcessor, exist_ok=True)
