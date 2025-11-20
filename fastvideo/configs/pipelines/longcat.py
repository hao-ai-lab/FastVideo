# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
import html

import ftfy
import regex as re
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


def longcat_preprocess_text(prompt: str) -> str:
    """Clean and preprocess text like original LongCat implementation.
    
    This function applies the same text cleaning pipeline as the original
    LongCat-Video implementation to ensure identical tokenization results.
    
    Steps:
    1. basic_clean: Fix unicode issues and unescape HTML entities
    2. whitespace_clean: Normalize whitespace to single spaces
    
    Args:
        prompt: Raw input text prompt
        
    Returns:
        Cleaned and normalized text prompt
    """
    # basic_clean: fix unicode and HTML entities
    text = ftfy.fix_text(prompt)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    
    # whitespace_clean: normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text


def umt5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """
    Postprocess UMT5/T5 encoder outputs to fixed length 512 embeddings.
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
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (longcat_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (umt5_postprocess_text,)
    )

    # LongCat-specific runtime toggles (consumed by pipeline/stages)
    enable_kv_cache: bool = True
    offload_kv_cache: bool = False
    enable_bsa: bool = False
    use_distill: bool = False
    enhance_hf: bool = False
    # BSA runtime overrides (preferred over bsa_params if provided via CLI)
    bsa_sparsity: float | None = None
    bsa_cdf_threshold: float | None = None
    bsa_chunk_q: list[int] | None = None
    bsa_chunk_k: list[int] | None = None
    t_thresh: float | None = None  # refine stage default controlled by sampling args

    # LongCat doesnot need flow_shift
    flow_shift: float | None = None
    dmd_denoising_steps: list[int] | None = None

    def __post_init__(self):
        # LongCat inference requires vae encoder and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class LongCatT2V704PConfig(LongCatT2V480PConfig):
    """Configuration for LongCat pipeline (704p) with BSA enabled by default.
    
    Uses the same resolution and BSA parameters as original LongCat refinement stage.
    BSA parameters configured in transformer config.json with chunk_3d_shape=[4,4,4]:
    - Input: 704×1280×96
    - VAE (8x): 88×160×96  
    - Patch [1,2,2]: 44×80×96
    - chunk [4,4,4]: 96%4=0, 44%4=0, 80%4=0 ✅
    
    This configuration matches the original LongCat refinement stage parameters.
    """
    
    # Enable BSA by default for 704p
    enable_bsa: bool = True
