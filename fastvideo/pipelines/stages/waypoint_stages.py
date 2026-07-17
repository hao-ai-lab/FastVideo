# SPDX-License-Identifier: Apache-2.0
"""Waypoint-1-Small pipeline stages."""

from dataclasses import dataclass
import io
import urllib.request

import numpy as np
from PIL import Image
import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.waypoint_transformer import WaypointKVCache
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


class WaypointTextEncodingStage(PipelineStage):
    """Encode the prompt with UMT5."""

    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.prompt is None:
            return batch

        text_encoder = self.text_encoder
        tokenizer = self.tokenizer
        max_length = getattr(getattr(text_encoder, "config", None), "text_len", 512)
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
        prompt_emb = prompt_emb * attention_mask.unsqueeze(-1).to(device=prompt_emb.device, dtype=prompt_emb.dtype)
        prompt_pad_mask = attention_mask.eq(0)

        if batch.extra is None:
            batch.extra = {}
        batch.extra["waypoint_prompt_emb"] = prompt_emb
        batch.extra["waypoint_prompt_pad_mask"] = prompt_pad_mask

        return batch


@dataclass
class WaypointStreamingContext:
    batch: ForwardBatch
    fastvideo_args: FastVideoArgs
    kv_cache: WaypointKVCache
    frame_index: int = 0
    prompt_emb: torch.Tensor | None = None
    prompt_pad_mask: torch.Tensor | None = None


class WaypointDenoisingStage(PipelineStage):
    """Generate Waypoint latents autoregressively."""

    def __init__(self, transformer, vae, dtype: torch.dtype):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.dtype = dtype
        self._device = next(transformer.parameters()).device
        self._streaming_ctx: WaypointStreamingContext | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _cache_pass(
        self,
        x: torch.Tensor,
        frame_timestamp: torch.Tensor,
        mouse: torch.Tensor,
        button: torch.Tensor,
        scroll: torch.Tensor,
    ) -> None:
        ctx = self._require_context()
        ctx.kv_cache.set_frozen(False)
        with set_forward_context(
                current_timestep=0,
                attn_metadata=SDPAMetadata(current_timestep=0, attn_mask=None),
                forward_batch=None,
        ):
            self.transformer(
                x=x,
                sigma=x.new_zeros((x.shape[0], 1)),
                frame_timestamp=frame_timestamp,
                prompt_emb=ctx.prompt_emb,
                prompt_pad_mask=ctx.prompt_pad_mask,
                mouse=mouse,
                button=button,
                scroll=scroll,
                kv_cache=ctx.kv_cache,
            )
        ctx.kv_cache.set_frozen(True)

    @staticmethod
    def _image_tensor(image, height: int, width: int) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            return torch.from_numpy(np.asarray(image).copy()).to(torch.uint8)

        tensor = torch.as_tensor(image).detach().cpu()
        while tensor.dim() > 3:
            tensor = tensor[0]
        if tensor.shape[0] in (1, 3) and tensor.shape[-1] not in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)
        tensor = tensor[..., :3]
        if tensor.dtype != torch.uint8:
            tensor = tensor.float()
            if tensor.max() <= 1.5:
                tensor = tensor * 255
            tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        if tuple(tensor.shape[:2]) != (height, width):
            tensor = torch.nn.functional.interpolate(
                tensor.permute(2, 0, 1).unsqueeze(0).float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        return tensor.contiguous()

    @staticmethod
    def _load_image(batch: ForwardBatch):
        image = getattr(batch, "image", None) or batch.pil_image
        if image is not None:
            return image
        image_path = batch.image_path
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None
        if image_path is None:
            return None
        if str(image_path).startswith(("http://", "https://")):
            with urllib.request.urlopen(image_path) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(image_path).convert("RGB")

    def _seed_image(self, image) -> None:
        ctx = self._require_context()
        vae_config = getattr(self.vae.config, "arch_config", self.vae.config)
        height, width = getattr(vae_config, "sample_size", (360, 640))
        latent = self.vae.encode(self._image_tensor(image, height, width))
        if latent.dim() == 4:
            latent = latent.unsqueeze(1)
        latent = latent.to(device=self.device, dtype=self.dtype)
        frame_timestamp = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        mouse = torch.zeros(1, 1, 2, device=self.device, dtype=self.dtype)
        button = torch.zeros(
            1,
            1,
            self.transformer.config.n_buttons,
            device=self.device,
            dtype=self.dtype,
        )
        scroll = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
        self._cache_pass(latent, frame_timestamp, mouse, button, scroll)
        ctx.frame_index = 1

    def _require_context(self) -> WaypointStreamingContext:
        if self._streaming_ctx is None:
            raise RuntimeError("Call streaming_reset() before streaming_step()")
        return self._streaming_ctx

    def streaming_reset(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        extra = batch.extra or {}
        prompt_emb = extra.get("waypoint_prompt_emb")
        prompt_pad_mask = extra.get("waypoint_prompt_pad_mask")
        if prompt_emb is not None:
            prompt_emb = prompt_emb.to(device=self.device, dtype=self.dtype)
        if prompt_pad_mask is not None:
            prompt_pad_mask = prompt_pad_mask.to(self.device)

        cache = WaypointKVCache(
            self.transformer.config,
            batch_size=1,
            dtype=self.dtype,
        ).to(self.device)
        self._streaming_ctx = WaypointStreamingContext(
            batch=batch,
            fastvideo_args=fastvideo_args,
            kv_cache=cache,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        image = self._load_image(batch)
        if image is not None:
            self._seed_image(image)

    @staticmethod
    def _controls(
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
        scroll: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if keyboard.dim() == 2:
            keyboard = keyboard.unsqueeze(0)
        if mouse.dim() == 2:
            mouse = mouse.unsqueeze(0)
        if keyboard.dim() != 3 or mouse.dim() != 3:
            raise ValueError("keyboard and mouse controls must have shape [B,T,C]")
        length = min(keyboard.shape[1], mouse.shape[1])
        if scroll is None:
            scroll = torch.zeros(keyboard.shape[0], length, 1)
        elif scroll.dim() == 1:
            scroll = scroll.view(1, -1, 1)
        elif scroll.dim() == 2:
            scroll = scroll.unsqueeze(-1)
        if scroll.dim() != 3 or scroll.shape[-1] != 1:
            raise ValueError("scroll controls must have shape [B,T,1], [B,T], or [T]")
        length = min(length, scroll.shape[1])
        return (
            keyboard[:, :length].to(device=device, dtype=dtype),
            mouse[:, :length].to(device=device, dtype=dtype),
            scroll[:, :length].to(device=device, dtype=dtype).sign(),
        )

    @torch.no_grad()
    def streaming_step(
        self,
        keyboard: torch.Tensor,
        mouse: torch.Tensor,
        scroll: torch.Tensor | None = None,
    ) -> ForwardBatch:
        ctx = self._require_context()
        keyboard, mouse, scroll = self._controls(keyboard, mouse, scroll, self.device, self.dtype)
        config = ctx.fastvideo_args.pipeline_config
        arch = getattr(config.dit_config, "arch_config", config.dit_config)
        patch_h, patch_w = arch.patch
        latent_shape = (
            keyboard.shape[0],
            1,
            arch.channels,
            arch.height * patch_h,
            arch.width * patch_w,
        )
        sigmas = torch.tensor(config.scheduler_sigmas, device=self.device, dtype=self.dtype)
        latents = []

        for control_index in range(keyboard.shape[1]):
            seed = int(ctx.batch.seed or 0) + ctx.frame_index
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latent = torch.randn(
                latent_shape,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            frame_timestamp = torch.full(
                (keyboard.shape[0], 1),
                ctx.frame_index,
                device=self.device,
                dtype=torch.long,
            )
            frame_mouse = mouse[:, control_index:control_index + 1]
            frame_button = keyboard[:, control_index:control_index + 1]
            frame_scroll = scroll[:, control_index:control_index + 1]
            ctx.kv_cache.set_frozen(True)

            for step, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=False)):
                with set_forward_context(
                        current_timestep=step,
                        attn_metadata=SDPAMetadata(current_timestep=step, attn_mask=None),
                        forward_batch=None,
                ):
                    velocity = self.transformer(
                        x=latent,
                        sigma=sigma.expand(latent.shape[0], 1),
                        frame_timestamp=frame_timestamp,
                        prompt_emb=ctx.prompt_emb,
                        prompt_pad_mask=ctx.prompt_pad_mask,
                        mouse=frame_mouse,
                        button=frame_button,
                        scroll=frame_scroll,
                        kv_cache=ctx.kv_cache,
                    )
                latent = latent + (next_sigma - sigma) * velocity

            self._cache_pass(
                latent,
                frame_timestamp,
                frame_mouse,
                frame_button,
                frame_scroll,
            )
            latents.append(latent[:, 0])
            ctx.frame_index += 1

        ctx.batch.latents = torch.stack(latents, dim=2)
        return ctx.batch

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        frames = max(batch.num_frames) if isinstance(batch.num_frames, list) else batch.num_frames
        keyboard = batch.keyboard_cond
        mouse = batch.mouse_cond
        if keyboard is None:
            keyboard = torch.zeros(1, frames, self.transformer.config.n_buttons)
        if mouse is None:
            mouse = torch.zeros(1, frames, 2)
        self.streaming_reset(batch, fastvideo_args)
        try:
            return self.streaming_step(keyboard, mouse, batch.scroll_cond)
        finally:
            self.streaming_clear()

    def streaming_clear(self) -> None:
        self._streaming_ctx = None


class WaypointDecodingStage(PipelineStage):
    """Decode Waypoint latents to RGB frames."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.latents is None:
            return batch
        dtype = next(self.vae.parameters()).dtype
        frames = []
        for index in range(batch.latents.shape[2]):
            frame = self.vae.decode(batch.latents[:, :, index].to(dtype=dtype))
            if frame.dim() == 3:
                frame = frame.permute(2, 0, 1).unsqueeze(0)
            frames.append(frame.float().div(255))
        batch.output = torch.stack(frames, dim=2)
        return batch
