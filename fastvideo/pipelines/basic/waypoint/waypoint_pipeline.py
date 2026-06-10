# SPDX-License-Identifier: Apache-2.0
"""Waypoint-1-Small interactive world model pipeline.

This pipeline supports streaming generation of video frames conditioned on
text prompts and real-time controller inputs (mouse, keyboard, scroll).
"""

from dataclasses import dataclass, field

import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.stages.waypoint_stages import WaypointTextEncodingStage
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

        scroll = torch.tensor([[[float(self.scroll > 0) - float(self.scroll < 0)]]], device=device, dtype=dtype)

        return mouse, button, scroll


@dataclass
class StreamingContext:
    """Context for streaming generation."""
    batch: ForwardBatch
    fastvideo_args: FastVideoArgs
    frame_index: int = 0
    kv_cache: list | None = None  # Per-layer ring caches (see _create_waypoint_kv_cache)
    prompt_emb: torch.Tensor | None = None
    prompt_pad_mask: torch.Tensor | None = None
    ref_latent_std: float | None = None
    disable_latent_norm: bool = False
    bf16_denoise: bool = False


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
        # Bulk DiT dtype (bf16/fp16) after load; denoise_step_emb may be fp32 alone.
        self._waypoint_dit_dtype: torch.dtype | None = None

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
        self._waypoint_dit_dtype = dit_dtype

        # Official Overworld keeps denoise_step_emb (NoiseConditioner) in
        # fp32 via NoCastModule + _keep_in_fp32_modules.  The MLP weights
        # inside the conditioner must stay fp32 so the sigma-to-embedding
        # mapping retains full precision; bf16 weights cause conditioning
        # error that compounds across autoregressive frames (latent drift).
        if hasattr(transformer, "denoise_step_emb"):
            transformer.denoise_step_emb.to(dtype=torch.float32)
            logger.info("Upcast denoise_step_emb to fp32 (matches official "
                        "Overworld NoCastModule)")

        vae = self.get_module("vae", None)
        if vae is not None:
            pipeline_config = fastvideo_args.pipeline_config
            vae_precision = getattr(pipeline_config, "vae_precision", "fp32")
            vae_dtype = PRECISION_TO_TYPE.get(vae_precision, torch.float32)
            vae.to(dtype=vae_dtype)

        logger.info("WaypointPipeline initialized for interactive generation")

    def _waypoint_compute_dtype(self, transformer: torch.nn.Module) -> torch.dtype:
        """DiT activations dtype (bf16/…); not denoise_step_emb’s fp32 params."""
        if self._waypoint_dit_dtype is not None:
            return self._waypoint_dit_dtype
        return next(transformer.parameters()).dtype

    def _cache_pass(
        self,
        transformer: torch.nn.Module,
        x: torch.Tensor,
        frame_ts: torch.Tensor,
        prompt_emb: torch.Tensor | None,
        prompt_pad_mask: torch.Tensor | None,
        mouse: torch.Tensor,
        button: torch.Tensor,
        scroll: torch.Tensor,
        kv_cache: list | None,
    ) -> None:
        """Run sigma=0 forward to persist a frame in the KV cache.

        Mirrors official StaticKVCache cache pass: unfreeze, write K/V, refreeze.
        """
        if kv_cache is None:
            return
        kv_cache[0]["frozen_ref"][0] = False
        sigma_zero = x.new_zeros((x.shape[0], 1))
        with set_forward_context(
                current_timestep=0,
                attn_metadata=SDPAMetadata(current_timestep=0, attn_mask=None),
                forward_batch=None,
        ):
            transformer(
                x=x,
                sigma=sigma_zero,
                frame_timestamp=frame_ts,
                prompt_emb=prompt_emb,
                prompt_pad_mask=prompt_pad_mask,
                mouse=mouse,
                button=button,
                scroll=scroll,
                kv_cache=kv_cache,
                update_cache=True,
            )
        kv_cache[0]["frozen_ref"][0] = True

    @torch.no_grad()
    def _seed_image_frame(
        self,
        image,
        ctx: "StreamingContext",
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Encode an init image and seed it as frame 0 in the KV cache.

        Matches official WorldEnginePrepareLatentsStep: VAE-encode the image to a
        [1,1,C,h,w] latent, run a sigma=0 cache pass at frame_timestamp=0 with zero
        control, then advance frame_index to 1 so denoising starts at the next frame.
        """
        vae = self.get_module("vae", None)
        transformer = self.get_module("transformer")
        if vae is None or ctx.kv_cache is None:
            return
        device = next(transformer.parameters()).device
        dtype = self._waypoint_compute_dtype(transformer)

        img = self._to_uint8_hwc(image)
        latent = vae.encode(img)  # [1, C, h, w]
        if latent.dim() == 4:
            latent = latent.unsqueeze(1)  # [1, 1, C, h, w]
        latent = latent.to(device=device, dtype=dtype)

        frame_ts = torch.zeros(1, 1, device=device, dtype=torch.long)
        prompt_emb = ctx.prompt_emb.to(device=device, dtype=dtype) if ctx.prompt_emb is not None else None
        prompt_pad_mask = ctx.prompt_pad_mask.to(device=device) if ctx.prompt_pad_mask is not None else None
        mouse = torch.zeros(1, 1, 2, device=device, dtype=dtype)
        button = torch.zeros(1, 1, 256, device=device, dtype=dtype)
        scroll = torch.zeros(1, 1, 1, device=device, dtype=dtype)

        self._cache_pass(transformer, latent, frame_ts, prompt_emb, prompt_pad_mask, mouse, button, scroll,
                         ctx.kv_cache)
        ctx.frame_index = 1
        logger.info("Waypoint seeded init image as frame 0 (kv_cache warm)")

    @staticmethod
    def _to_uint8_hwc(image) -> torch.Tensor:
        """Convert PIL/tensor (any of [H,W,3], [C,H,W], batched) to uint8 [H,W,3]."""
        import PIL.Image
        if isinstance(image, PIL.Image.Image):
            import numpy as np
            arr = np.asarray(image.convert("RGB"))
            return torch.from_numpy(arr).to(torch.uint8)
        t = image
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        t = t.detach().cpu()
        while t.dim() > 3:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
            t = t.permute(1, 2, 0)
        if t.shape[-1] == 1:
            t = t.repeat(1, 1, 3)
        t = t[..., :3]
        if t.dtype != torch.uint8:
            t = t.float()
            if t.max() <= 1.5:
                t = t * 255.0
            t = t.round().clamp(0, 255).to(torch.uint8)
        return t.contiguous()

    def _create_waypoint_kv_cache(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> list | None:
        """Create per-layer ring KV cache matching official StaticKVCache.

        Local layers use a ``local_window`` ring; global layers (every
        ``global_attn_period`` from ``global_attn_offset``) use a wider
        ``global_window`` ring with ``global_pinned_dilation``. Each layer ring
        holds ``L`` history tokens plus a ``tokens_per_frame`` tail for the
        current frame.
        """
        transformer = self.get_module("transformer", None)
        if transformer is None:
            return None
        pipeline_config = fastvideo_args.pipeline_config
        dit_config = pipeline_config.dit_config
        arch = getattr(dit_config, "arch_config", dit_config)
        n_layers = getattr(arch, "n_layers", getattr(arch, "num_layers", 22))
        n_kv_heads = getattr(arch, "n_kv_heads", 20)
        head_dim = getattr(arch, "attention_head_dim", arch.d_model // arch.n_heads)
        tpf = getattr(arch, "tokens_per_frame", 256)
        local_window = getattr(arch, "local_window", 16)
        global_window = getattr(arch, "global_window", 128)
        period = getattr(arch, "global_attn_period", 4)
        offset = getattr(arch, "global_attn_offset", 0) % period
        global_pinned_dilation = getattr(arch, "global_pinned_dilation", 8)
        device = next(transformer.parameters()).device
        dtype = self._waypoint_compute_dtype(transformer)
        B = 1
        frozen_ref = [True]
        kv_cache = []
        for layer_idx in range(n_layers):
            is_global = (layer_idx - offset) % period == 0
            window = global_window if is_global else local_window
            pinned_dilation = global_pinned_dilation if is_global else 1
            L = window * tpf
            capacity = L + tpf
            num_buckets = (L // tpf) // pinned_dilation
            written = torch.zeros(capacity, dtype=torch.bool, device=device)
            written[L:] = True
            frame_offsets = torch.arange(tpf, dtype=torch.long, device=device)
            kv_cache.append({
                "k": torch.zeros(B, n_kv_heads, capacity, head_dim, device=device, dtype=dtype),
                "v": torch.zeros(B, n_kv_heads, capacity, head_dim, device=device, dtype=dtype),
                "L": L,
                "tpf": tpf,
                "pinned_dilation": pinned_dilation,
                "num_buckets": num_buckets,
                "written": written,
                "frame_offsets": frame_offsets,
                "current_idx": frame_offsets + L,
                "frozen_ref": frozen_ref,
            })
        return kv_cache

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run one-shot generation for ``VideoGenerator`` / ``execute_forward``.

        ``ComposedPipelineBase.forward`` iterates ``self.stages``, but Waypoint
        leaves stages empty (interactive API uses ``streaming_*`` only). Bridge
        by calling the same reset/step/clear path as streaming.
        """
        if not self.post_init_called:
            self.post_init()

        keyboard = batch.keyboard_cond
        mouse = batch.mouse_cond
        if keyboard is None or mouse is None:
            transformer = self.get_module("transformer")
            device = next(transformer.parameters()).device
            dtype = self._waypoint_compute_dtype(transformer)
            nf = batch.num_frames
            nf = (max(nf) if nf else 1) if isinstance(nf, list) else int(nf)
            n_buttons = getattr(fastvideo_args.pipeline_config, "n_buttons", 256)
            # Zero only the missing tensor so a caller that supplies just one of
            # keyboard_cond / mouse_cond does not have the provided one discarded.
            if keyboard is None:
                logger.warning("Waypoint forward: missing keyboard_cond; using zeros for "
                               "%d frames", nf)
                keyboard = torch.zeros(1, nf, n_buttons, device=device, dtype=dtype)
            if mouse is None:
                logger.warning("Waypoint forward: missing mouse_cond; using zeros for "
                               "%d frames", nf)
                mouse = torch.zeros(1, nf, 2, device=device, dtype=dtype)

        try:
            self.streaming_reset(batch, fastvideo_args)
            return self.streaming_step(keyboard, mouse)
        finally:
            self.streaming_clear()

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

        # Encode prompt using WaypointTextEncodingStage (reuse stages)
        text_encoder = self.get_module("text_encoder", None)
        tokenizer = self.get_module("tokenizer", None)
        prompt_emb = None
        prompt_pad_mask = None
        if text_encoder is not None and tokenizer is not None:
            text_stage = WaypointTextEncodingStage(text_encoder, tokenizer)
            batch = text_stage(batch, fastvideo_args)
            extra = getattr(batch, "extra", None) or {}
            prompt_emb = extra.get("waypoint_prompt_emb")
            prompt_pad_mask = extra.get("waypoint_prompt_pad_mask")

        # Build KV cache for autoregressive cross-frame attention (StaticKVCache
        # semantics: frozen during denoise, unfrozen only for sigma=0 cache pass).
        kv_cache = self._create_waypoint_kv_cache(batch, fastvideo_args)

        pipeline_config = fastvideo_args.pipeline_config
        ctx = StreamingContext(
            batch=batch,
            fastvideo_args=fastvideo_args,
            frame_index=0,
            kv_cache=kv_cache,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
            disable_latent_norm=getattr(pipeline_config, "disable_latent_norm", False),
            bf16_denoise=getattr(pipeline_config, "bf16_denoise", False),
        )
        self._streaming_ctx = ctx

        # Init-image conditioning: encode the image and seed it as frame 0 so the
        # rollout is conditioned on the given first frame (matches official).
        image = (getattr(batch, "image", None) or getattr(batch, "pil_image", None))
        if image is None:
            image_path = getattr(batch, "image_path", None)
            if image_path:
                import PIL.Image
                if str(image_path).startswith(("http://", "https://")):
                    import io
                    import urllib.request
                    with urllib.request.urlopen(image_path) as resp:
                        image = PIL.Image.open(io.BytesIO(resp.read())).convert("RGB")
                else:
                    image = PIL.Image.open(image_path).convert("RGB")
        if image is not None:
            self._seed_image_frame(image, ctx, fastvideo_args)

        logger.info(
            "Waypoint streaming reset complete (KV cache: %s)",
            "enabled (StaticKVCache semantics)" if kv_cache else "disabled",
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
        ph, pw = (patch, patch) if isinstance(patch, int) else (patch[0], patch[1])
        latent_h = arch.height * ph
        latent_w = arch.width * pw

        device = next(transformer.parameters()).device
        dtype = self._waypoint_compute_dtype(transformer)

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

        scroll_action = torch.zeros(keyboard_action.shape[0], t, 1, device=device, dtype=dtype)

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

        # HF Waypoint expects latents [B, 1, 16, 32, 32] (32x32 -> 16x16 tokens/frame)
        if latent_h != 32 or latent_w != 32:
            logger.warning(
                "Waypoint expects 32x32 latents (tokens_per_frame=256). "
                "Got %dx%d; RoPE/attention may be wrong.",
                latent_h,
                latent_w,
            )

        # Prompt tensors are constant across the rollout; move them to the
        # transformer device/dtype once (needed under CPU offload) instead of
        # re-copying every frame.
        prompt_emb = ctx.prompt_emb.to(device=device, dtype=dtype) if ctx.prompt_emb is not None else None
        prompt_pad_mask = ctx.prompt_pad_mask.to(device=device) if ctx.prompt_pad_mask is not None else None

        for local_i in range(t):
            # Noise: [B, 1, C, H, W] with H,W = latent grid (height*ph, width*pw)
            # Use explicit generator per frame (official: seed + frame_idx) for repro.
            latent_shape = (
                keyboard_action.shape[0],
                1,
                arch.channels,
                latent_h,
                latent_w,
            )
            seed = getattr(ctx.batch, "seed", None) or getattr(ctx.fastvideo_args, "seed", None) or 0
            g = torch.Generator(device=device).manual_seed(int(seed) + ctx.frame_index)
            x = torch.randn(latent_shape, device=device, dtype=dtype, generator=g)

            frame_ts = torch.full(
                (keyboard_action.shape[0], 1),
                ctx.frame_index,
                device=device,
                dtype=torch.long,
            )
            # Index controls by the local step within THIS call's action tensor,
            # not the global frame_index: with image seeding (frame_index starts
            # at 1) or a second streaming_step (frame_index already large), the
            # global index would drop/repeat actions. Equivalent to frame_index
            # on the single-shot-from-0 path.
            ctrl_step = (min(local_i, mouse.shape[1] - 1) if mouse.shape[1] > 0 else 0)

            # StaticKVCache semantics: cache is read-only during denoise.
            if ctx.kv_cache is not None:
                ctx.kv_cache[0]["frozen_ref"][0] = True

            # Rectified-flow ODE accumulation. Official accumulates in bf16; the
            # fp32 path (default) reduces drift across long autoregressive rollouts.
            # Toggle with pipeline_config.bf16_denoise to match official exactly.
            acc_dtype = dtype if ctx.bf16_denoise else torch.float32
            x_acc = x.to(acc_dtype)
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

                # Accumulate in acc_dtype (fp32 default / bf16 if matching official)
                x_acc = x_acc + (sigma_next - sigma_curr).to(acc_dtype) * v_pred.to(acc_dtype)
                x = x_acc.to(dtype)

            # Smooth latent normalization (non-official stabilization hack): the
            # autoregressive loop causes latent std to grow over hundreds of frames
            # (~0.5 → ~1.5). Smoothly pull every frame's scale toward frame-0's std
            # (strength=0.8). The official pipeline does NOT do this; it flattens
            # output (std ~16 vs ~52). Disable with pipeline_config.disable_latent_norm.
            if not ctx.disable_latent_norm:
                xf = x.float()
                cur_std = max(xf.std().item(), 1e-6)
                if ctx.ref_latent_std is None:
                    ctx.ref_latent_std = cur_std
                else:
                    target_std = ctx.ref_latent_std
                    strength = 0.8
                    blended_std = (strength * target_std + (1.0 - strength) * cur_std)
                    scale = blended_std / cur_std
                    if abs(scale - 1.0) > 1e-4:
                        x = (xf * scale).to(dtype)

            # Cache pass: run forward with sigma=0 to update KV cache for next frame.
            self._cache_pass(
                transformer,
                x,
                frame_ts,
                prompt_emb,
                prompt_pad_mask,
                mouse[:, ctrl_step:ctrl_step + 1],
                button[:, ctrl_step:ctrl_step + 1],
                scroll[:, ctrl_step:ctrl_step + 1],
                ctx.kv_cache,
            )

            # Decode latent to frame. The VAE is a spatial upscaler expecting the
            # transformer's native grid (e.g. 32x32); do not resize latents before
            # decode or output is low-res.
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
                scaling_factor = (getattr(vae_config, "scaling_factor", None) or getattr(vae, "scaling_factor", 1.0))
                if scaling_factor is not None and abs(float(scaling_factor) - 1.0) > 1e-5:
                    latent_in = latent_in / float(scaling_factor)
                shift = getattr(vae_config, "shift_factor", None)
                if shift is not None:
                    if isinstance(shift, torch.Tensor):
                        shift = shift.to(latent_in.device, latent_in.dtype)
                    latent_in = latent_in + shift
                decoded = vae.decode(latent_in)

                frame = (decoded.sample if hasattr(decoded, "sample") else decoded)
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
                # WorldEngineVAE decodes to roughly [-1, 1]. clamp(0, 1) alone maps
                # all negatives to 0 (crushed blacks); streaming latents can drift
                # so later frames look progressively darker.
                if frame.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    frame = frame.float() / 255.0
                else:
                    frame = frame.float()
                    fmin = frame.amin().item()
                    fmax = frame.amax().item()
                    if fmax > 1.5:
                        frame = frame / 255.0
                    elif fmin < -0.02:
                        frame = (frame + 1.0) * 0.5
                    frame = frame.clamp(0.0, 1.0)

                generated_frames.append(frame)

            ctx.frame_index += 1

        if generated_frames:
            # Stack to [B, C, T, H, W]
            ctx.batch.output = torch.stack(generated_frames, dim=2)

        return ctx.batch

    def streaming_clear(self) -> None:
        """Clear streaming state."""
        self._streaming_ctx = None
        logger.info("Waypoint streaming cleared")


EntryClass = [WaypointPipeline]
