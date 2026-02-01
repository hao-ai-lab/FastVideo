import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import PIL
import torch
from diffusers.image_processor import VaeImageProcessor

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import FluxConfig
from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
    TextEncoderConfig,
)
from fastvideo.configs.models.encoders.base import TextEncoderArchConfig
from fastvideo.configs.models.encoders import Qwen2_5_VLConfig as Qwen3TextConfig
from fastvideo.configs.models.encoders.qwen2_5 import _is_transformer_layer
from fastvideo.configs.models.vaes.fluxvae import Flux2VAEConfig, FluxVAEConfig
from fastvideo.configs.pipelines.base import (
    ImagePipelineConfig,
    ModelTaskType,
    preprocess_text,
    shard_rotary_emb_for_sp,
)
from fastvideo.configs.pipelines.hunyuan import (
    clip_postprocess_text,
    clip_preprocess_text,
)
from fastvideo.configs.pipelines.qwen_image import _pack_latents
from fastvideo.distributed import get_local_torch_device


def t5_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs: torch.Tensor | None = None
) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class FluxPipelineConfig(ImagePipelineConfig):
    """Configuration for the FLUX pipeline."""

    embedded_cfg_scale: float = 3.5
    flow_shift: float | None = 3.0
    timestep_input_scale: float | None = 1.0 / 1000.0
    embedded_cfg_scale_multiplier: float = 1.0
    force_dynamic_shifting: bool = True
    use_flux_ode_schedule: bool = True
    flux_base_shift: float = 0.5
    flux_max_shift: float = 1.15
    flux_shift: bool = True

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=FluxConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)

    enable_autocast: bool = False

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), CLIPTextConfig())
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, clip_preprocess_text),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (t5_postprocess_text, clip_postprocess_text)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            None,
            dict(
                max_length=77,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
            ),
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def _flux_time_shift(self, mu: float, sigma: float,
                         t: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.tensor(mu, device=t.device, dtype=t.dtype)) / (
            torch.exp(torch.tensor(mu, device=t.device, dtype=t.dtype)) +
            (1 / t - 1)**sigma)

    def _flux_get_lin_fn(self, x1: float = 256, y1: float = 0.5,
                         x2: float = 4096,
                         y2: float = 1.15) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def get_flux_timesteps(self,
                           num_steps: int,
                           image_seq_len: int,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        timesteps = torch.linspace(1,
                                   0,
                                   num_steps + 1,
                                   device=device,
                                   dtype=dtype)
        if self.flux_shift:
            mu = self._flux_get_lin_fn(y1=self.flux_base_shift,
                                       y2=self.flux_max_shift)(image_seq_len)
            timesteps = self._flux_time_shift(mu, 1.0, timesteps)
        return timesteps

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def _prepare_latent_image_ids(self, original_height, original_width, device):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = int(original_height) // (vae_scale_factor * 2)
        width = int(original_width) // (vae_scale_factor * 2)
        latent_image_ids = torch.zeros(height, width, 3, device=device)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)
        img_ids = self._prepare_latent_image_ids(
            original_height=height,
            original_width=width,
            device=device,
        )

        # NOTE(mick): prepare it here, to avoid unnecessary computations
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def post_denoising_loop(self, latents, batch):
        return self.unpack_latents_for_decoding(latents)

    def pack_latents_for_denoising(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            if latents.shape[2] != 1:
                return latents
            latents_4d = latents[:, :, 0]
        elif latents.ndim == 4:
            latents_4d = latents
        else:
            return latents

        in_channels = self.dit_config.arch_config.in_channels
        vae_latent_channels = self.vae_config.arch_config.latent_channels
        if latents_4d.shape[1] == in_channels:
            return latents_4d.unsqueeze(2)
        if latents_4d.shape[1] != vae_latent_channels:
            return latents

        packed = _patchify_latents(latents_4d)
        return packed.unsqueeze(2)

    def unpack_latents_for_decoding(self,
                                    latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            if latents.shape[2] != 1:
                return latents
            latents_4d = latents[:, :, 0]
            add_time = True
        elif latents.ndim == 4:
            latents_4d = latents
            add_time = False
        else:
            return latents

        in_channels = self.dit_config.arch_config.in_channels
        vae_latent_channels = self.vae_config.arch_config.latent_channels
        if latents_4d.shape[1] == vae_latent_channels:
            return latents_4d.unsqueeze(2) if add_time else latents_4d
        if latents_4d.shape[1] != in_channels:
            return latents

        unpacked = _unpatchify_latents(latents_4d)
        return unpacked.unsqueeze(2) if add_time else unpacked

    def preprocess_decoding(self, latents, vae=None):
        latents = self.unpack_latents_for_decoding(latents)
        if latents.ndim != 5:
            return latents, None
        batch_size, channels, num_frames, height, width = latents.shape
        if num_frames == 1:
            return latents[:, :, 0], {"unsqueeze_time": True}
        flat = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        return flat, {"batch_size": batch_size, "num_frames": num_frames}

    def postprocess_decoding(self, images, ctx, vae=None):
        if not ctx:
            return images
        if ctx.get("unsqueeze_time"):
            return images.unsqueeze(2)
        batch_size = ctx.get("batch_size")
        num_frames = ctx.get("num_frames")
        if batch_size is None or num_frames is None:
            return images
        images = images.reshape(batch_size, num_frames, *images.shape[1:])
        return images.permute(0, 2, 1, 3, 4)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
            "pooled_projections": batch.clip_embedding_pos,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.negative_prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
            "pooled_projections": batch.clip_embedding_neg,
        }


def _prepare_latent_ids(
    latents: torch.Tensor,  # (B, C, H, W)
):
    r"""
    Generates 4D position coordinates (T, H, W, L) for latent tensors.

    Args:
        latents (torch.Tensor):
            Latent tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor:
            Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
            H=[0..H-1], W=[0..W-1], L=0
    """

    batch_size, _, height, width = latents.shape

    t = torch.arange(1)  # [0] - time dimension
    h = torch.arange(height)
    w = torch.arange(width)
    layer = torch.arange(1)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t, h, w, layer)

    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids


def _unpack_latents_with_ids(
    x: torch.Tensor, x_ids: torch.Tensor
) -> list[torch.Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    x_ids = x_ids.to(device=x.device)
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(
        batch_size, num_channels_latents * 4, height // 2, width // 2
    )
    return latents


def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
    )
    return latents


def _prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: Optional[torch.Tensor] = None,
):
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        layer = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, layer)
        out_ids.append(coords)

    return torch.stack(out_ids)


def _prepare_image_ids(
    image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
    scale: int = 10,
):
    if not isinstance(image_latents, list):
        raise ValueError(
            f"Expected `image_latents` to be a list, got {type(image_latents)}."
        )

    # create time offset for each reference image
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape

        x_ids = torch.cartesian_prod(
            t, torch.arange(height), torch.arange(width), torch.arange(1)
        )
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    image_latent_ids = image_latent_ids.unsqueeze(0)

    return image_latent_ids


def flux2_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    hidden_states_layers: list[int] = [10, 20, 30]

    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


def flux2_klein_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    hidden_states_layers: list[int] = [9, 18, 27]

    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


@dataclass
class Flux2MistralTextArchConfig(TextEncoderArchConfig):
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer]
    )

    def __post_init__(self):
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class Flux2MistralTextConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=Flux2MistralTextArchConfig
    )


def format_text_input(prompts: List[str], system_message: str = None):
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


def flux_2_preprocess_text(prompt: str):
    system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."
    return format_text_input([prompt], system_message=system_message)


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def flux2_pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    return latents


@dataclass
class Flux2PipelineConfig(FluxPipelineConfig):
    embedded_cfg_scale: float = 4.0

    task_type: ModelTaskType = ModelTaskType.TI2I

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Flux2MistralTextConfig(),)
    )
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (flux_2_preprocess_text,),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (flux2_postprocess_text,)
    )
    vae_config: VAEConfig = field(default_factory=Flux2VAEConfig)

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        # flatten to 1-d list
        prompts = [p for prompt in prompts for p in prompt]
        inputs = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            # 2048 from official github repo, 512 from diffusers
            max_length=512,
        )

        return inputs

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels
        shape = (batch_size, num_channels_latents, height // 2, width // 2)
        return shape

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def calculate_condition_image_size(
        self, image, width, height
    ) -> Optional[tuple[int, int]]:
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        multiple_of = vae_scale_factor * 2

        target_area: int = 1024 * 1024
        if width is not None and height is not None:
            new_width, new_height = width, height
            if width * height > target_area:
                scale = math.sqrt(target_area / (width * height))
                new_width = int(width * scale)
                new_height = int(height * scale)

            # Flux requires multiples of (VAE scale 8 * Patch size 2)
            new_width = (new_width // multiple_of) * multiple_of
            new_height = (new_height // multiple_of) * multiple_of

            if new_width != width or new_height != height:
                return new_width, new_height

        return None

    def preprocess_condition_image(
        self, image, target_width, target_height, vae_image_processor: VaeImageProcessor
    ):
        img = image.resize((target_width, target_height), PIL.Image.Resampling.LANCZOS)
        image_width, image_height = img.size
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        multiple_of = vae_scale_factor * 2
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of
        img = vae_image_processor.preprocess(
            img, height=image_height, width=image_width, resize_mode="crop"
        )
        return img, (image_width, image_height)

    def postprocess_image_latent(self, latent_condition, batch):
        batch_size = batch.batch_size
        # get image_latent_ids right after scale & shift
        image_latent_ids = _prepare_image_ids([latent_condition])
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(get_local_torch_device())
        batch.condition_image_latent_ids = image_latent_ids

        # latent: (1, 128, 32, 32)
        packed = self.maybe_pack_latents(
            latent_condition, None, batch
        )  # (1, 1024, 128)
        packed = packed.squeeze(0)  # (1024, 128) - remove batch dim

        # Concatenate all reference tokens along sequence dimension
        image_latents = packed.unsqueeze(0)  # (1, N*1024, 128)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        return image_latents

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        txt_ids = _prepare_text_ids(prompt_embeds).to(device=device)

        img_ids = batch.latent_ids
        if batch.image_latent is not None:
            image_latent_ids = batch.condition_image_latent_ids
            img_ids = torch.cat([img_ids, image_latent_ids], dim=1).to(device=device)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # NOTE(mick): prepare it here, to avoid unnecessary computations
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def maybe_pack_latents(self, latents, batch_size, batch):
        return flux2_pack_latents(latents)

    def maybe_prepare_latent_ids(self, latents):
        return _prepare_latent_ids(latents)

    def postprocess_vae_encode(self, image_latents, vae):
        # patchify
        image_latents = _patchify_latents(image_latents)
        return image_latents

    def _check_vae_has_bn(self, vae):
        """Check if VAE has bn attribute (cached check to avoid repeated hasattr calls)."""
        if not hasattr(self, "_vae_has_bn_cache"):
            self._vae_has_bn_cache = hasattr(vae, "bn") and vae.bn is not None
        return self._vae_has_bn_cache

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Preprocess latents before decoding.

        Dynamically adapts based on VAE type:
        - Standard Flux2 VAE (has bn): needs unpatchify (128 channels -> 32 channels)
        - Distilled VAE (no bn): keeps patchified latents (128 channels)
        """
        if vae is not None and self._check_vae_has_bn(vae):
            return _unpatchify_latents(latents)
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """Get scale and shift for decoding.

        Dynamically adapts based on VAE type:
        - Standard Flux2 VAE (has bn): uses BatchNorm statistics
        - Distilled VAE (no bn): uses scaling_factor from config
        """
        vae_arch_config = self.vae_config.arch_config

        if self._check_vae_has_bn(vae):
            # Standard Flux2 VAE: use BatchNorm statistics
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
            latents_bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae_arch_config.batch_norm_eps
            ).to(device, dtype)
            return 1 / latents_bn_std, latents_bn_mean

        # Distilled VAE or unknown: use scaling_factor
        scaling_factor = (
            getattr(vae.config, "scaling_factor", None)
            if hasattr(vae, "config")
            else getattr(vae, "scaling_factor", None)
        ) or getattr(vae_arch_config, "scaling_factor", 0.13025)

        scale = torch.tensor(scaling_factor, device=device, dtype=dtype).view(
            1, 1, 1, 1
        )
        return 1 / scale, None

    def post_denoising_loop(self, latents, batch):
        latent_ids = batch.latent_ids
        latents = _unpack_latents_with_ids(latents, latent_ids)

        return latents

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1) :]
        return noise


@dataclass
class Flux2KleinPipelineConfig(Flux2PipelineConfig):
    # Klein is distilled, so no guidance embeddings
    should_use_guidance: bool = False

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3TextConfig(),)
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text,),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (flux2_klein_postprocess_text,)
    )

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        if prompts and isinstance(prompts[0], list):
            prompts = [p for prompt in prompts for p in prompt]

        def _apply_chat_template(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        texts = [_apply_chat_template(prompt) for prompt in prompts]

        tok_kwargs = dict(tok_kwargs or {})
        max_length = tok_kwargs.pop("max_length", 512)
        padding = tok_kwargs.pop("padding", "max_length")
        truncation = tok_kwargs.pop("truncation", True)
        return_tensors = tok_kwargs.pop("return_tensors", "pt")

        return tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **tok_kwargs,
        )


@dataclass
class FluxT2IConfig(FluxPipelineConfig):
    """Alias for registry compatibility."""
    pass
