from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, VAEConfig
from fastvideo.configs.models.dits import LongCatVideoConfig  # Native model config
from fastvideo.configs.models.dits.base import DiTArchConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class LongCatDiTArchConfig(DiTArchConfig):
    """Extended DiTArchConfig with LongCat-specific fields.
    
    NOTE: This is for Phase 1 wrapper compatibility. For native model (Phase 2),
    use LongCatVideoConfig from fastvideo.configs.models.dits.longcat instead.
    """
    # LongCat-specific architecture parameters
    adaln_tembed_dim: int = 512
    caption_channels: int = 4096
    depth: int = 48
    enable_bsa: bool = False
    enable_flashattn3: bool = False
    enable_flashattn2: bool = True
    enable_xformers: bool = False
    frequency_embedding_size: int = 256
    in_channels: int = 16
    mlp_ratio: int = 4
    num_heads: int = 32
    out_channels: int = 16
    text_tokens_zero_pad: bool = True
    patch_size: list[int] = field(default_factory=lambda: [1, 2, 2])
    cp_split_hw: list[int] | None = None
    bsa_params: dict | None = None


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
      - transformer: LongCatVideoTransformer3DModel (Phase 1 wrapper)
                  OR LongCatTransformer3DModel (Phase 2 native)
      - vae: AutoencoderKLWan (Wan VAE, 4x8 compression)
      - scheduler: FlowMatchEulerDiscreteScheduler
    """

    # DiT config with LongCat-specific arch_config
    # NOTE: For Phase 1 wrapper, uses LongCatDiTArchConfig
    # For Phase 2 native model, can use LongCatVideoConfig directly
    dit_config: DiTConfig = field(default_factory=lambda: DiTConfig(arch_config=LongCatDiTArchConfig()))

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

    # LongCat doesnot need flow_shift
    flow_shift: float | None = None
    dmd_denoising_steps: list[int] | None = None

    def __post_init__(self):
        # LongCat inference requires vae encoder and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class LongCatT2V720PConfig(LongCatT2V480PConfig):
    """Configuration for LongCat pipeline (720p) with BSA enabled by default.
    
    720p resolution requires BSA for efficient inference. BSA parameters
    should be configured in the transformer config.json with chunk_3d_shape=[3,5,4]
    to match the latent dimensions after VAE compression and patch embedding:
    - Input: 720×1280×93
    - VAE (4x8): 90×160×93  
    - Patch [1,2,2]: 45×80×93
    - chunk [3,5,4]: 93%3=0, 45%5=0, 80%4=0 ✅
    """
    
    # Enable BSA by default for 720p
    enable_bsa: bool = True
    