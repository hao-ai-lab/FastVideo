"""GLM-ASR config classes, ported to transformers 4.57's classic
``PretrainedConfig.__init__(**kwargs)`` idiom.

Upstream (transformers 5.x, ``models/glmasr/configuration_glmasr.py``)
defines these as ``@strict``-decorated dataclasses inheriting from the
new ``PreTrainedConfig`` base. That base class doesn't exist in
4.57.3, so we cannot just rename — the dataclass-vs-classic-init
contracts are different. The defaults and field semantics here track
upstream exactly; only the constructor shape changed.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class GlmAsrEncoderConfig(PretrainedConfig):
    """Audio-tower config for GLM-ASR (Whisper-style mel encoder)."""

    model_type = "glmasr_encoder"

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 20,
        num_key_value_heads: int | None = None,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 1500,
        initializer_range: float = 0.02,
        rope_parameters: dict | None = None,
        attention_dropout: float = 0.0,
        num_mel_bins: int = 128,
        partial_rotary_factor: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.num_mel_bins = num_mel_bins
        self.partial_rotary_factor = partial_rotary_factor
        super().__init__(**kwargs)


class GlmAsrConfig(PretrainedConfig):
    """Composite config: audio encoder + Llama-style text decoder."""

    model_type = "glmasr"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 59264,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "eos_token_id": [59246, 59253, 59255],
        "rope_parameters": {
            "rope_theta": 10000.0,
            "rope_type": "default"
        },
    }

    def __init__(
        self,
        audio_config: dict | PretrainedConfig | None = None,
        text_config: dict | PretrainedConfig | None = None,
        audio_token_id: int = 59260,
        projector_hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        if isinstance(audio_config, dict):
            audio_config = dict(audio_config)
            audio_config.setdefault("model_type", "glmasr_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["glmasr_encoder"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.setdefault("model_type", "llama")
            merged = {**self._default_text_config_kwargs, **text_config}
            text_config = CONFIG_MAPPING[merged["model_type"]](**merged)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.projector_hidden_act = projector_hidden_act
        super().__init__(**kwargs)


__all__ = ["GlmAsrEncoderConfig", "GlmAsrConfig"]
