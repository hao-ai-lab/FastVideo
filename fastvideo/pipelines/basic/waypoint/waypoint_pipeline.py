# SPDX-License-Identifier: Apache-2.0
"""Waypoint-1-Small interactive world model pipeline.

This pipeline supports streaming generation of video frames conditioned on
text prompts and real-time controller inputs (mouse, keyboard, scroll).
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.fastvideo_args import FastVideoArgs
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
    button: Set[int] = field(default_factory=set)
    mouse: Tuple[float, float] = (0.0, 0.0)
    scroll: float = 0.0
    
    def to_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
        n_buttons: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            device=device, dtype=dtype)
        
        return mouse, button, scroll


@dataclass
class StreamingContext:
    """Context for streaming generation."""
    batch: ForwardBatch
    fastvideo_args: FastVideoArgs
    frame_index: int = 0
    kv_cache: Optional[dict] = None
    prompt_emb: Optional[torch.Tensor] = None
    prompt_pad_mask: Optional[torch.Tensor] = None


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
        "transformer", "vae", "text_encoder", "tokenizer",
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_ctx: Optional[StreamingContext] = None
        self._vae_cache = None
    
    def create_pipeline_stages(
        self, fastvideo_args: FastVideoArgs,
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
            vae.to(dtype=dit_dtype)

        logger.info(
            "WaypointPipeline initialized for interactive generation")
    
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
            max_length = getattr(
                getattr(text_encoder, "config", None),
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
            prompt_pad_mask = attention_mask.bool()
        
        self._streaming_ctx = StreamingContext(
            batch=batch,
            fastvideo_args=fastvideo_args,
            frame_index=0,
            kv_cache=None,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        
        self._vae_cache = None
        logger.info("Waypoint streaming reset complete")
    
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
        keyboard_action = keyboard_action[:, :t].to(
            device=device, dtype=dtype)
        mouse_action = mouse_action[:, :t].to(
            device=device, dtype=dtype)

        scroll_action = torch.zeros(
            keyboard_action.shape[0], t, 1,
            device=device, dtype=dtype)

        button = keyboard_action
        mouse = mouse_action
        scroll = scroll_action
        
        # Scheduler sigma schedule
        sigmas = torch.tensor(
            pipeline_config.scheduler_sigmas,
            device=device, dtype=dtype,
        )
        
        generated_frames: list[torch.Tensor] = []

        for _ in range(t):
            # Noise: [B, 1, C, H, W] (single frame)
            latent_shape = (
                keyboard_action.shape[0], 1, dit_config.channels,
                dit_config.height, dit_config.width,
            )
            x = torch.randn(latent_shape, device=device, dtype=dtype)
            
            frame_ts = torch.full(
                (keyboard_action.shape[0], 1),
                ctx.frame_index, device=device, dtype=torch.long,
            )
            
            # Denoise through sigma schedule
            for i in range(len(sigmas) - 1):
                sigma_curr = sigmas[i]
                sigma_next = sigmas[i + 1]
                sigma = sigma_curr.view(1, 1)
                
                attn_metadata = SDPAMetadata(
                    current_timestep=i, attn_mask=None)
                
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=attn_metadata,
                    forward_batch=None,
                ):
                    v_pred = transformer(
                        x=x,
                        sigma=sigma,
                        frame_timestamp=frame_ts,
                        prompt_emb=ctx.prompt_emb.to(dtype=dtype),
                        prompt_pad_mask=ctx.prompt_pad_mask,
                        mouse=mouse[:, :1],
                        button=button[:, :1],
                        scroll=scroll[:, :1],
                        kv_cache=ctx.kv_cache,
                    )
                
                x = x + (sigma_next - sigma_curr) * v_pred
            
            # Decode latent to frame
            if vae is not None:
                frame = vae.decode(x[:, 0])
                # Ensure frame is [B, C, H, W]
                if frame.dim() == 3:
                    # [H, W, C] -> [1, C, H, W]
                    frame = frame.permute(2, 0, 1).unsqueeze(0)
                elif frame.dim() == 4 and frame.shape[-1] <= 4:
                    # [B, H, W, C] -> [B, C, H, W]
                    frame = frame.permute(0, 3, 1, 2)
                generated_frames.append(frame)
            
            ctx.frame_index += 1
        
        if generated_frames:
            # Stack to [B, C, T, H, W]
            ctx.batch.output = torch.stack(
                generated_frames, dim=2)
        
        return ctx.batch
    
    def streaming_clear(self) -> None:
        """Clear streaming state."""
        self._streaming_ctx = None
        self._vae_cache = None
        logger.info("Waypoint streaming cleared")


EntryClass = [WaypointPipeline]
