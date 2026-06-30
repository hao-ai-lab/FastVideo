# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.base import ArchConfig, ModelConfig
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class DiTArchConfig(ArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    # When True, the denoising stage casts text/prompt embeddings to the DiT's
    # working dtype before the diffusion loop. Flux2 requires this (BFL casts ctx
    # to bf16 before denoising); models with fp32 text encoders (Wan, Hunyuan15,
    # SD3.5) leave it False to preserve full-precision embeddings.
    cast_prompt_embeds_to_dit_dtype: bool = False
    _supported_attention_backends: tuple[AttentionBackendEnum,
                                         ...] = (AttentionBackendEnum.SAGE_ATTN, AttentionBackendEnum.FLASH_ATTN,
                                                 AttentionBackendEnum.TORCH_SDPA,
                                                 AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                                                 AttentionBackendEnum.VMOBA_ATTN, AttentionBackendEnum.SAGE_ATTN_THREE,
                                                 AttentionBackendEnum.ATTN_QAT_INFER,
                                                 AttentionBackendEnum.ATTN_QAT_TRAIN, AttentionBackendEnum.SLA_ATTN,
                                                 AttentionBackendEnum.SAGE_SLA_ATTN)

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    in_channels: int = 0
    out_channels: int = 0
    exclude_lora_layers: list[str] = field(default_factory=list)
    boundary_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class DiTConfig(ModelConfig):
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)

    # FastVideoDiT-specific parameters
    prefix: str = ""
    quant_config: QuantizationConfig | None = None
    # Some model families are only numerically correct with a specific attention
    # backend (e.g. FastWan is sparse-distilled with VSA). When set, the model
    # fails loudly if the selected FASTVIDEO_ATTENTION_BACKEND does not match it.
    # None = no requirement (the common case).
    required_attention_backend: AttentionBackendEnum | None = None
    # A checkpoint can support the model's ordinary attention implementation
    # while still being incompatible with a particular backend-specific block
    # layout. Keep those checkpoint constraints separate from the architecture's
    # general _supported_attention_backends list.
    incompatible_attention_backends: tuple[AttentionBackendEnum, ...] = ()

    @staticmethod
    def _parse_attention_backend(backend: AttentionBackendEnum | str, field_name: str) -> AttentionBackendEnum:
        if isinstance(backend, AttentionBackendEnum):
            return backend
        if isinstance(backend, str):
            try:
                return AttentionBackendEnum[backend]
            except KeyError as exc:
                valid_backends = ", ".join(AttentionBackendEnum.__members__)
                raise ValueError(
                    f"Unknown attention backend {backend!r} in {field_name}; expected one of: {valid_backends}"
                ) from exc
        raise TypeError(f"{field_name} must contain AttentionBackendEnum values or names, got {type(backend).__name__}")

    def update_model_config(self, source_model_dict: dict[str, Any]) -> None:
        source_model_dict = source_model_dict.copy()
        required_backend = source_model_dict.get("required_attention_backend")
        if isinstance(required_backend, str):
            source_model_dict["required_attention_backend"] = self._parse_attention_backend(
                required_backend, "required_attention_backend")

        incompatible_backends = source_model_dict.get("incompatible_attention_backends")
        if incompatible_backends is not None:
            if not isinstance(incompatible_backends, list | tuple):
                raise TypeError("incompatible_attention_backends must be a list or tuple")
            source_model_dict["incompatible_attention_backends"] = tuple(
                self._parse_attention_backend(backend, "incompatible_attention_backends")
                for backend in incompatible_backends)

        super().update_model_config(source_model_dict)

    def validate_attention_backend_compatibility(
        self,
        selected_backend: AttentionBackendEnum | None,
        *,
        model_name: str = "This model",
    ) -> None:
        if selected_backend in self.incompatible_attention_backends:
            assert selected_backend is not None
            raise ValueError(f"{model_name} is incompatible with the {selected_backend.name} attention backend. "
                             "Select a compatible attention backend before loading the model.")

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "dit-config") -> Any:
        """Add CLI arguments for DiTConfig fields"""
        parser.add_argument(
            f"--{prefix}.prefix",
            type=str,
            dest=f"{prefix.replace('-', '_')}.prefix",
            default=DiTConfig.prefix,
            help="Prefix for the DiT model",
        )

        parser.add_argument(
            f"--{prefix}.quant-config",
            type=str,
            dest=f"{prefix.replace('-', '_')}.quant_config",
            default=None,
            help="Quantization configuration for the DiT model",
        )

        return parser
