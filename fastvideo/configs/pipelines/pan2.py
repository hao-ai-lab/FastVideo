# SPDX-License-Identifier: Apache-2.0
"""PAN2 pipeline configuration: text-to-video, and image-to-video from a first frame."""

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import PAN2VideoConfig
from fastvideo.configs.models.encoders import Qwen3_5TextConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.models.wan.vae_config import WanVAEConfig

PAN2_SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."


@dataclass
class PAN2PipelineConfig(PipelineConfig):
    """One configuration serves text-to-video and, given a first frame, image-to-video."""

    dit_config: DiTConfig = field(default_factory=PAN2VideoConfig)
    # The Wan2.2 48-channel VAE; its architecture and latent statistics come from vae/config.json at load time.
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Qwen3_5TextConfig(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    # Decode the video with fastvideo-kernel's Wan2.2 decoder (B200 / GB200 / B300; a 704 x 1248 video, batch 1): bf16
    # with fp32 accumulation, about 7x faster than the fp32 decode on a B300, at PSNR 61-63 dB and SSIM 0.998 against
    # it. Other devices and sizes, and False, keep the fp32 decode.
    vae_decode_kernel: bool = True
    # With vae_decode_kernel, split the decode over the sequence-parallel GPUs (2, 4 or 8) rather than decoding the
    # whole video on each of them. On B300s a 20 s video decodes in 3.0 s on one GPU, 1.55 s on 2 (the same video, bit
    # for bit), 0.79 s on 4 and 0.47 s on 8 (rounded differently, at the same PSNR against the fp32 decode).
    vae_decode_kernel_parallel: bool = True
    # Run the DiT's fused Triton kernels (Q/K RMSNorm + RoPE, LayerNorm + modulation): 21% less denoising time on 8
    # B300s, as far from the eager path's samples as bf16 rounding differences carry them.
    dit_fused_kernels: bool = True
    # Unset: the checkpoint's scheduler config decides, which is shift 9 for IFM/PAN2-TEST, the reference sampler's
    # fixed value. A fixed default here would override the released shift with whatever a test checkpoint declares.
    flow_shift: float | None = None
    # Image-to-video: the first latent frame (the conditioning image) takes the positive prediction instead of the
    # guided one, as the reference sampler does by default (``--i2v_skip_first_latent_cfg``).
    i2v_skip_first_latent_cfg: bool = True
    # PAN2 samples on fp32 latents; keep the Euler update out of autocast.
    scheduler_step_in_fp32: bool = True

    # Prompt encoding: PAN2 conditions on the tokens after the chat template's user header, taken from the text
    # encoder's hidden states with its last two layers skipped (``hidden_states[-3]``).
    text_system_prompt: str = PAN2_SYSTEM_PROMPT
    text_encoder_skip_layers: int = 2
    text_max_sequence_length: int = 1536

    def __post_init__(self) -> None:
        # The encoder is needed for the image-to-video first frame.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.text_encoder_configs[0].arch_config.output_hidden_states = True
