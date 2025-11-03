from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, VAEConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def umt5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess UMT5/T5 encoder outputs to fixed length 512 embeddings.

    TODO: Not sure if using same procedure as t5 postprocess_text function is correct.
    """
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in prompt_embeds
    ], dim=0)
    return prompt_embeds_tensor


@dataclass
class LongCatT2V480PConfig(PipelineConfig):
    """Configuration for LongCat pipeline (480p) aligned to LongCat-Video modules.

    Components expected by loaders:
      - tokenizer: AutoTokenizer
      - text_encoder: UMT5EncoderModel
      - transformer: LongCatVideoTransformer3DModel
      - vae: AutoencoderKLWan (Wan VAE, 4x8 compression)
      - scheduler: FlowMatchEulerDiscreteScheduler
    """

    # DiT placeholder (unused by LongCat transformer loader; kept for base compatibility)
    dit_config: DiTConfig = field(default_factory=DiTConfig)

    # VAE config: Wan VAE with encoder+decoder enabled
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Precision defaults
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    # Text encoding (UMT5 uses T5-like config; postprocess to fixed 512)
    text_encoder_configs: tuple[T5Config, ...] = field(default_factory=lambda: (T5Config(),))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (umt5_postprocess_text,)
    )

    # LongCat-specific runtime toggles (consumed by pipeline/stages)
    enable_kv_cache: bool = True
    offload_kv_cache: bool = False
    enable_bsa: bool = False
    use_distill: bool = False
    enhance_hf: bool = False
    t_thresh: float | None = None  # refine stage default controlled by sampling args

    # LongCat 不使用 Wan 的 flow_shift
    flow_shift: float | None = None
    dmd_denoising_steps: list[int] | None = None

    def __post_init__(self):
        # LongCat 推理需要 VAE 编解码器均可用
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
    