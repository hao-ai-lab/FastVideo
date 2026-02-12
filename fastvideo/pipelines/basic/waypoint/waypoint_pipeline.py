# SPDX-License-Identifier: Apache-2.0
"""Waypoint-1-Small interactive world model pipeline.

This pipeline supports streaming generation of video frames conditioned on
text prompts and real-time controller inputs (mouse, keyboard, scroll).
"""

from dataclasses import dataclass, field

import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.utils import PRECISION_TO_TYPE
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import ComposedPipelineBase, ForwardBatch

logger = init_logger(__name__)


@dataclass
class CtrlInput:
    """Controller input for Waypoint world model.
    
    Attributes:
        button: Set of pressed button IDs (0-255). Uses Owl-Control keycodes.
        mouse: Tuple of (x, y) mouse velocity as floats.
        scroll: Scroll wheel value (-1, 0, or 1).
    """
    button: set[int] = field(default_factory=set)
    mouse: tuple[float, float] = (0.0, 0.0)
    scroll: float = 0.0

    def to_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
        n_buttons: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to tensor format for model input.
        
        Returns:
            mouse: [1, 1, 2] tensor
            button: [1, 1, n_buttons] one-hot tensor
            scroll: [1, 1, 1] tensor with sign of scroll
        """
        mouse = torch.tensor([[list(self.mouse)]], device=device, dtype=dtype)

        button = torch.zeros(1, 1, n_buttons, device=device, dtype=dtype)
        for b in self.button:
            if 0 <= b < n_buttons:
                button[0, 0, b] = 1.0

        scroll = torch.tensor(
            [[[float(self.scroll > 0) - float(self.scroll < 0)]]],
            device=device,
            dtype=dtype)

        return mouse, button, scroll


@dataclass
class StreamingContext:
    """Context for streaming generation."""
    batch: ForwardBatch
    fastvideo_args: FastVideoArgs
    frame_index: int = 0
    kv_cache: list | None = None  # Per-layer list of {"k", "v", "end"}
    prompt_emb: torch.Tensor | None = None
    prompt_pad_mask: torch.Tensor | None = None


class WaypointPipeline(ComposedPipelineBase):
    """Waypoint interactive world model pipeline.
    
    This pipeline generates video frames autoregressively, conditioned on:
    - Text prompts (encoded via UMT5)
    - Controller inputs (mouse, keyboard, scroll)
    
    Usage:
        1. Call ``streaming_reset()`` with initial prompt to set up
        2. Call ``streaming_step()`` repeatedly with control inputs
        3. Call ``streaming_clear()`` when done
    """

    _required_config_modules = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_ctx: StreamingContext | None = None
        self._vae_cache = None

    def create_pipeline_stages(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Create pipeline stages.
        
        Note: Waypoint uses a custom streaming interface rather than
        the standard stage-based pipeline due to its interactive nature.
        """
        # Ensure consistent dtype across all model parameters.
        # The safetensors checkpoint may store some biases in float32
        # while the model was initialised in bfloat16; a single
        # ``.to()`` call normalises everything.
        transformer = self.get_module("transformer")
        dit_dtype = next(transformer.parameters()).dtype
        transformer.to(dtype=dit_dtype)

        vae = self.get_module("vae", None)
        if vae is not None:
            pipeline_config = fastvideo_args.pipeline_config
            vae_precision = getattr(pipeline_config, "vae_precision", "fp32")
            vae_dtype = PRECISION_TO_TYPE.get(vae_precision, torch.float32)
            vae.to(dtype=vae_dtype)

        logger.info("WaypointPipeline initialized for interactive generation")

    def _create_waypoint_kv_cache(self, batch: ForwardBatch,
                                  fastvideo_args: FastVideoArgs) -> list | None:
        """Create per-layer KV cache for autoregressive cross-frame attention."""
        transformer = self.get_module("transformer", None)
        if transformer is None:
            return None
        pipeline_config = fastvideo_args.pipeline_config
        dit_config = pipeline_config.dit_config
        arch = getattr(dit_config, "arch_config", dit_config)
        n_layers = getattr(arch, "n_layers", getattr(arch, "num_layers", 22))
        n_kv_heads = getattr(arch, "n_kv_heads", 20)
        head_dim = getattr(arch, "attention_head_dim",
                           arch.d_model // arch.n_heads)
        tokens_per_frame = getattr(arch, "tokens_per_frame", 256)
        max_frames = getattr(pipeline_config, "max_kv_cache_frames", 64)
        cache_size = max_frames * tokens_per_frame
        device = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype
        B = 1
        kv_cache = []
        for _ in range(n_layers):
            kv_cache.append({
                "k":
                torch.zeros(
                    B,
                    n_kv_heads,
                    cache_size,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "v":
                torch.zeros(
                    B,
                    n_kv_heads,
                    cache_size,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "end":
                0,
            })
        return kv_cache

    @torch.no_grad()
    def streaming_reset(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Reset streaming state and encode the prompt.
        
        Args:
            batch: Forward batch containing the prompt
            fastvideo_args: FastVideo arguments
        """
        if not self.post_init_called:
            self.post_init()

        # Encode prompt using text encoder
        text_encoder = self.get_module("text_encoder", None)
        tokenizer = self.get_module("tokenizer", None)

        prompt_emb = None
        prompt_pad_mask = None

        if text_encoder is not None and tokenizer is not None:
            max_length = getattr(getattr(text_encoder, "config", None),
                                 "text_len", 512)
            text_inputs = tokenizer(
                batch.prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            enc_device = next(text_encoder.parameters()).device
            input_ids = text_inputs.input_ids.to(enc_device)
            attention_mask = text_inputs.attention_mask.to(enc_device)

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            prompt_emb = outputs.last_hidden_state
            # Zero out padding so cross-attention does not attend to pad tokens (match HF)
            prompt_emb = prompt_emb * attention_mask.unsqueeze(-1).to(
                prompt_emb.dtype)
            prompt_pad_mask = attention_mask.eq(0)

        # Build KV cache for autoregressive cross-frame attention (HF parity)
        kv_cache = self._create_waypoint_kv_cache(batch, fastvideo_args)

        self._streaming_ctx = StreamingContext(
            batch=batch,
            fastvideo_args=fastvideo_args,
            frame_index=0,
            kv_cache=kv_cache,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )

        self._vae_cache = None
        logger.info(
            "Waypoint streaming reset complete (KV cache: %s)",
            "enabled" if kv_cache else "disabled",
        )

    @torch.no_grad()
    def streaming_step(
        self,
        keyboard_action: torch.Tensor,
        mouse_action: torch.Tensor,
    ) -> ForwardBatch:
        """Generate next frame(s) given control input.
        
        Args:
            keyboard_action: Button conditioning, expected shape
                [T, 256] or [B, T, 256].
            mouse_action: Mouse velocity conditioning, expected shape
                [T, 2] or [B, T, 2].
            
        Returns:
            ForwardBatch with decoded frames in batch.output
        """
        assert self._streaming_ctx is not None, \
            "Call streaming_reset() first"

        ctx = self._streaming_ctx
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        pipeline_config = ctx.fastvideo_args.pipeline_config
        dit_config = pipeline_config.dit_config
        arch = getattr(dit_config, "arch_config", dit_config)
        patch = getattr(arch, "patch", (1, 1))
        ph, pw = (patch, patch) if isinstance(patch, int) else (patch[0],
                                                                patch[1])
        latent_h = arch.height * ph
        latent_w = arch.width * pw

        device = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype

        # Normalize action tensor shapes to [B, T, ...]
        if keyboard_action.dim() == 2:
            keyboard_action = keyboard_action.unsqueeze(0)
        if mouse_action.dim() == 2:
            mouse_action = mouse_action.unsqueeze(0)
        assert keyboard_action.dim() == 3, \
            "keyboard_action must be [B,T,256] or [T,256]"
        assert mouse_action.dim() == 3, \
            "mouse_action must be [B,T,2] or [T,2]"

        # Ensure same time length
        t = min(keyboard_action.shape[1], mouse_action.shape[1])
        keyboard_action = keyboard_action[:, :t].to(device=device, dtype=dtype)
        mouse_action = mouse_action[:, :t].to(device=device, dtype=dtype)

        scroll_action = torch.zeros(keyboard_action.shape[0],
                                    t,
                                    1,
                                    device=device,
                                    dtype=dtype)

        button = keyboard_action
        mouse = mouse_action
        scroll = scroll_action

        # Scheduler sigma schedule
        sigmas = torch.tensor(
            pipeline_config.scheduler_sigmas,
            device=device,
            dtype=dtype,
        )

        generated_frames: list[torch.Tensor] = []

        # Multi-frame debug: log first N frames to trace white/yellow drift (set 0 to disable)
        DEBUG_MULTIFRAME_MAX = 5
        # Log last N frames to trace end-of-video blur (set 0 to disable)
        DEBUG_LAST_N = 5
        _log_last_frames = DEBUG_LAST_N > 0 and t > DEBUG_LAST_N

        # HF Waypoint expects latents [B, 1, 16, 32, 32] (32x32 -> 16x16 tokens/frame)
        if latent_h != 32 or latent_w != 32:
            logger.warning(
                "Waypoint expects 32x32 latents (tokens_per_frame=256). "
                "Got %dx%d; RoPE/attention may be wrong.",
                latent_h,
                latent_w,
            )

        for _ in range(t):
            # Noise: [B, 1, C, H, W] with H,W = latent grid (height*ph, width*pw)
            latent_shape = (
                keyboard_action.shape[0],
                1,
                arch.channels,
                latent_h,
                latent_w,
            )
            x = torch.randn(latent_shape, device=device, dtype=dtype)

            frame_ts = torch.full(
                (keyboard_action.shape[0], 1),
                ctx.frame_index,
                device=device,
                dtype=torch.long,
            )
            ctrl_step = (min(ctx.frame_index, mouse.shape[1] -
                             1) if mouse.shape[1] > 0 else 0)

            _is_last_window = (_log_last_frames
                               and ctx.frame_index >= t - DEBUG_LAST_N)
            if ctx.frame_index < DEBUG_MULTIFRAME_MAX:
                m_slice = mouse[:, ctrl_step:ctrl_step + 1]
                b_slice = button[:, ctrl_step:ctrl_step + 1]
                logger.info(
                    "DEBUG frame %d: ctrl_step=%d frame_ts=%s "
                    "mouse mean=%.4f button sum=%.2f (cond slice)",
                    ctx.frame_index,
                    ctrl_step,
                    frame_ts[0, 0].item(),
                    m_slice.float().mean().item(),
                    b_slice.float().sum().item(),
                )
            elif _is_last_window:
                logger.info(
                    "DEBUG (last) frame %d/%d: ctrl_step=%d frame_ts=%s",
                    ctx.frame_index,
                    t,
                    ctrl_step,
                    frame_ts[0, 0].item(),
                )

            # Denoise through sigma schedule
            if ctx.frame_index == 0:
                logger.info(
                    "DEBUG sigmas=%s  x_init: mean=%.4f std=%.4f",
                    [round(s.item(), 4) for s in sigmas],
                    x.float().mean().item(),
                    x.float().std().item(),
                )
            for i in range(len(sigmas) - 1):
                sigma_curr = sigmas[i]
                sigma_next = sigmas[i + 1]
                sigma = torch.full(
                    (x.shape[0], 1),
                    sigma_curr.item(),
                    device=device,
                    dtype=dtype,
                )

                attn_metadata = SDPAMetadata(current_timestep=i, attn_mask=None)

                # Ensure prompt tensors are on same device as transformer (needed with CPU offload)
                prompt_emb = ctx.prompt_emb.to(
                    device=device,
                    dtype=dtype) if ctx.prompt_emb is not None else None
                prompt_pad_mask = ctx.prompt_pad_mask.to(
                    device=device) if ctx.prompt_pad_mask is not None else None
                with set_forward_context(
                        current_timestep=i,
                        attn_metadata=attn_metadata,
                        forward_batch=None,
                ):
                    v_pred = transformer(
                        x=x,
                        sigma=sigma,
                        frame_timestamp=frame_ts,
                        prompt_emb=prompt_emb,
                        prompt_pad_mask=prompt_pad_mask,
                        mouse=mouse[:, ctrl_step:ctrl_step + 1],
                        button=button[:, ctrl_step:ctrl_step + 1],
                        scroll=scroll[:, ctrl_step:ctrl_step + 1],
                        kv_cache=ctx.kv_cache,
                    )

                if ctx.frame_index == 0:
                    xf = x.float()
                    vf = v_pred.float()
                    dsig = (sigma_next - sigma_curr).item()
                    logger.info(
                        "DEBUG step %d: sigma=%.4f->%.4f  "
                        "v_pred mean=%.4f std=%.4f min=%.4f max=%.4f  "
                        "x mean=%.4f std=%.4f  dsig=%.4f",
                        i,
                        sigma_curr.item(),
                        sigma_next.item(),
                        vf.mean().item(),
                        vf.std().item(),
                        vf.min().item(),
                        vf.max().item(),
                        xf.mean().item(),
                        xf.std().item(),
                        dsig,
                    )

                x = x + (sigma_next - sigma_curr) * v_pred

            if ctx.frame_index < DEBUG_MULTIFRAME_MAX:
                xf = x.float()
                logger.info(
                    "DEBUG frame %d denoised x: mean=%.4f std=%.4f min=%.4f max=%.4f",
                    ctx.frame_index,
                    xf.mean().item(),
                    xf.std().item(),
                    xf.min().item(),
                    xf.max().item(),
                )
            elif _is_last_window:
                xf = x.float()
                logger.info(
                    "DEBUG (last) frame %d denoised x: mean=%.4f std=%.4f "
                    "min=%.4f max=%.4f",
                    ctx.frame_index,
                    xf.mean().item(),
                    xf.std().item(),
                    xf.min().item(),
                    xf.max().item(),
                )

            # Cache pass: run forward with sigma=0 to update KV cache for next frame
            if ctx.kv_cache is not None:
                sigma_zero = torch.zeros(x.shape[0],
                                         1,
                                         device=device,
                                         dtype=dtype)
                with set_forward_context(
                        current_timestep=0,
                        attn_metadata=SDPAMetadata(current_timestep=0,
                                                   attn_mask=None),
                        forward_batch=None,
                ):
                    transformer(
                        x=x,
                        sigma=sigma_zero,
                        frame_timestamp=frame_ts,
                        prompt_emb=prompt_emb,
                        prompt_pad_mask=prompt_pad_mask,
                        mouse=mouse[:, ctrl_step:ctrl_step + 1],
                        button=button[:, ctrl_step:ctrl_step + 1],
                        scroll=scroll[:, ctrl_step:ctrl_step + 1],
                        kv_cache=ctx.kv_cache,
                        update_cache=True,
                    )
                    if ctx.frame_index < DEBUG_MULTIFRAME_MAX or _is_last_window:
                        logger.info(
                            "DEBUG %sframe %d: KV cache updated (sigma=0 pass done)",
                            "(last) " if _is_last_window else "",
                            ctx.frame_index,
                        )

            # Decode latent to frame (WorldEngineVAE / OWL VAE).
            # CRITICAL: Do NOT resize/interpolate latents before decode. The VAE is a
            # spatial upscaler: it expects the transformer's native grid (e.g. 32x32
            # for Waypoint-1-Small, or 48x32 for full Waypoint) and outputs full-res
            # pixels (e.g. 384x256). Resizing latents would force low-res blurry output.
            if vae is not None:
                latent_in = x[:, 0]  # [B, C, H, W] from DiT
                expected_spatial = (latent_h, latent_w)
                assert latent_in.shape[-2:] == expected_spatial, (
                    "Latent spatial size must match DiT output; got "
                    f"{tuple(latent_in.shape[-2:])}, expected {expected_spatial}. "
                    "Do not resize latents before decode.")
                # WorldEngineVAE/OWL VAE: use VAE dtype (bf16/fp16), NOT float32.
                vae_dtype = next(vae.parameters()).dtype
                latent_in = latent_in.to(dtype=vae_dtype)
                vae_config = getattr(vae, "config", None)
                scaling_factor = (getattr(vae_config, "scaling_factor", None)
                                  or getattr(vae, "scaling_factor", 1.0))
                if scaling_factor is not None and abs(
                        float(scaling_factor) - 1.0) > 1e-5:
                    latent_in = latent_in / float(scaling_factor)
                shift = getattr(vae_config, "shift_factor", None)
                if shift is not None:
                    if isinstance(shift, torch.Tensor):
                        shift = shift.to(latent_in.device, latent_in.dtype)
                    latent_in = latent_in + shift
                if ctx.frame_index == 0:
                    logger.info(
                        "DEBUG VAE class: %s  type(vae)=%s",
                        type(vae).__name__,
                        type(vae).__mro__,
                    )
                if ctx.frame_index < DEBUG_MULTIFRAME_MAX:
                    lf = latent_in.float()
                    logger.info(
                        "DEBUG frame %d VAE input: mean=%.4f std=%.4f min=%.4f max=%.4f",
                        ctx.frame_index,
                        lf.mean().item(),
                        lf.std().item(),
                        lf.min().item(),
                        lf.max().item(),
                    )
                elif _is_last_window:
                    lf = latent_in.float()
                    logger.info(
                        "DEBUG (last) frame %d VAE input: mean=%.4f std=%.4f "
                        "min=%.4f max=%.4f",
                        ctx.frame_index,
                        lf.mean().item(),
                        lf.std().item(),
                        lf.min().item(),
                        lf.max().item(),
                    )
                decoded = vae.decode(latent_in)
                if ctx.frame_index == 0:
                    logger.info(
                        "DEBUG VAE raw output: type=%s "
                        "has_sample=%s",
                        type(decoded).__name__,
                        hasattr(decoded, "sample"),
                    )
                _d = decoded.sample if hasattr(decoded, "sample") else decoded
                if isinstance(_d, torch.Tensor
                              ) and ctx.frame_index < DEBUG_MULTIFRAME_MAX:
                    _df = _d.float()
                    logger.info(
                        "DEBUG frame %d decoded: mean=%.2f std=%.2f min=%.1f max=%.1f",
                        ctx.frame_index,
                        _df.mean().item(),
                        _df.std().item(),
                        _df.min().item(),
                        _df.max().item(),
                    )
                    if _d.dim() >= 3 and _d.shape[-1] == 3:
                        for c in range(3):
                            ch = _df[..., c]
                            logger.info(
                                "DEBUG frame %d ch%d (RGB): mean=%.2f min=%.1f max=%.1f",
                                ctx.frame_index,
                                c,
                                ch.mean().item(),
                                ch.min().item(),
                                ch.max().item(),
                            )
                elif isinstance(_d, torch.Tensor) and _is_last_window:
                    _df = _d.float()
                    logger.info(
                        "DEBUG (last) frame %d decoded: mean=%.2f std=%.2f "
                        "min=%.1f max=%.1f",
                        ctx.frame_index,
                        _df.mean().item(),
                        _df.std().item(),
                        _df.min().item(),
                        _df.max().item(),
                    )

                frame = (decoded.sample
                         if hasattr(decoded, "sample") else decoded)
                # Normalize to [B, C, H, W]
                if frame.dim() == 3:
                    frame = frame.unsqueeze(0)
                if frame.shape[-1] == 3:
                    frame = frame.permute(0, 3, 1, 2)
                # Verify VAE output is full-res (e.g. 384x256), not latent res (32x32)
                if ctx.frame_index == 0:
                    out_h, out_w = frame.shape[-2], frame.shape[-1]
                    if (out_h, out_w) == expected_spatial:
                        logger.warning(
                            "Waypoint VAE output is latent resolution %dx%d; "
                            "expected full-res (e.g. 384x256). Check VAE is spatial upscaler.",
                            out_h,
                            out_w,
                        )
                    else:
                        logger.info(
                            "Waypoint VAE decode: latent %s -> pixel %dx%d",
                            expected_spatial,
                            out_h,
                            out_w,
                        )
                # Clamp only; do not rescale. Output in [0, 1] for streaming_generator.
                if frame.dtype in (torch.uint8, torch.int8, torch.int16,
                                   torch.int32, torch.int64):
                    frame = frame.float() / 255.0
                else:
                    frame = frame.float().clamp(0.0, 1.0)

                if ctx.frame_index < DEBUG_MULTIFRAME_MAX:
                    logger.info(
                        "DEBUG frame %d final: mean=%.4f min=%.4f max=%.4f "
                        "(drift to 1.0 = white, high mean = washout)",
                        ctx.frame_index,
                        frame.mean().item(),
                        frame.min().item(),
                        frame.max().item(),
                    )
                elif _is_last_window:
                    logger.info(
                        "DEBUG (last) frame %d final: mean=%.4f min=%.4f max=%.4f "
                        "(blur/washout check)",
                        ctx.frame_index,
                        frame.mean().item(),
                        frame.min().item(),
                        frame.max().item(),
                    )
                generated_frames.append(frame)

            ctx.frame_index += 1

        if generated_frames:
            # Stack to [B, C, T, H, W]
            ctx.batch.output = torch.stack(generated_frames, dim=2)

        return ctx.batch

    def streaming_clear(self) -> None:
        """Clear streaming state."""
        self._streaming_ctx = None
        self._vae_cache = None
        logger.info("Waypoint streaming cleared")


EntryClass = [WaypointPipeline]
