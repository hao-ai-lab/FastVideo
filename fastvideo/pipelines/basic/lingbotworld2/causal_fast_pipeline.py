# SPDX-License-Identifier: Apache-2.0
"""LingBot World 2 causal-fast image-to-video pipeline."""

import math
import os

from einops import rearrange
import numpy as np
import torch
import torchvision.transforms.functional as TF

from fastvideo.distributed import get_local_torch_device
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.lingbotworld2.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    InputValidationStage,
    TextEncodingStage,
)
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class LingBotWorld2TextEncodingStage(TextEncodingStage):
    """Keep LingBot World 2 T5 attention masks so DiT context matches the source pipeline."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Initialize mask storage before running the shared text-encoding stage."""
        if batch.prompt_attention_mask is None:
            batch.prompt_attention_mask = []
        return super().forward(batch, fastvideo_args)


class LingBotWorld2CausalFastGenerationStage(PipelineStage):
    """Prepare LingBot World 2 conditions and run the released causal-fast sampling loop."""

    def __init__(self, transformer, scheduler, vae) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self._cross_attn_initialized = False

    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert LingBot World 2 flow prediction to x0 using the scheduler sigma."""
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    def _initialize_self_kv_cache(
        self,
        batch_size: int,
        kv_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[dict]:
        """Allocate per-block self-attention KV cache tensors."""
        head_dim = self.transformer.dim // self.transformer.num_heads
        num_heads = self.transformer.num_heads // get_sp_world_size()
        shape = [batch_size, kv_size, num_heads, head_dim]
        return [{
            "k": torch.zeros(shape, dtype=dtype, device=device),
            "v": torch.zeros(shape, dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.transformer.num_layers)]

    def _initialize_crossattn_cache(
        self,
        batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[dict]:
        """Allocate per-block text cross-attention KV cache tensors."""
        head_dim = self.transformer.dim // self.transformer.num_heads
        shape = [batch_size, max_sequence_length, self.transformer.num_heads, head_dim]
        return [{
            "k": torch.zeros(shape, dtype=dtype, device=device),
            "v": torch.zeros(shape, dtype=dtype, device=device),
            "is_init": torch.tensor([0], dtype=torch.bool, device=device),
        } for _ in range(self.transformer.num_layers)]

    @staticmethod
    def _prompt_context(batch: ForwardBatch, device: torch.device) -> list[torch.Tensor]:
        """Slice padded text encoder states back to LingBot World 2's unpadded context list."""
        assert batch.prompt_embeds
        context_tensor = batch.prompt_embeds[0].to(device)
        if batch.prompt_attention_mask:
            mask = batch.prompt_attention_mask[0].to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            return [u[:v] for u, v in zip(context_tensor, seq_lens, strict=True)]
        return [u for u in context_tensor]

    def _prepare_image_tensor(self, batch: ForwardBatch, device: torch.device) -> torch.Tensor:
        """Return the source-style normalized image tensor `[C,H,W]`."""
        image = batch.pil_image
        if image is None:
            raise ValueError("LingBot World 2 causal-fast requires `image_path` or `pil_image`.")
        if isinstance(image, torch.Tensor):
            if image.ndim == 5:
                return image[0, :, 0].to(device)
            if image.ndim == 4:
                return image[0].to(device)
            return image.to(device)
        return TF.to_tensor(image).sub_(0.5).div_(0.5).to(device)

    def _prepare_camera(
        self,
        action_path: str,
        c2ws: np.ndarray,
        h: int,
        w: int,
        lat_f: int,
        lat_h: int,
        lat_w: int,
        chunk_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the LingBot World 2 camera Plucker tensor for latent chunks."""
        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(
            Ks,
            height_org=480,
            width_org=832,
            height_resize=h,
            width_resize=w,
            height_final=h,
            width_final=w,
        )
        Ks = Ks[0]
        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer.to(device), Ks.to(device), h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        return rearrange(
            c2ws_plucker_emb,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        ).to(device=device, dtype=dtype)

    def _encode_condition_video(
        self,
        img: torch.Tensor,
        h: int,
        w: int,
        frames: int,
        mask: torch.Tensor,
        fastvideo_args: FastVideoArgs,
    ) -> torch.Tensor:
        """Encode the first-frame conditioning video and prepend mask channels."""
        device = get_local_torch_device()
        self.vae = self.vae.to(device)
        video_condition = torch.concat(
            [
                torch.nn.functional.interpolate(
                    img[None].cpu(),
                    size=(h, w),
                    mode="bicubic",
                ).transpose(0, 1),
                torch.zeros(3, frames - 1, h, w),
            ],
            dim=1,
        ).to(device)
        vae_dtype = torch.float32
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=False):
            encoder_output = self.vae.encode(video_condition.unsqueeze(0).to(torch.float32))
        latent_condition = encoder_output.mean
        if not bool(getattr(self.vae, "handles_latent_denorm", False)):
            if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
                latent_condition -= self.vae.shift_factor.to(latent_condition.device, latent_condition.dtype)
            latent_condition = latent_condition * self.vae.scaling_factor.to(latent_condition.device,
                                                                             latent_condition.dtype)
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return torch.concat([mask, latent_condition[0]], dim=0)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Execute LingBot World 2 causal-fast generation and store final latents on the batch."""
        device = get_local_torch_device()
        cfg = fastvideo_args.pipeline_config.dit_config.arch_config
        chunk_size = int(cfg.chunk_size)
        max_sequence_length = int(batch.max_sequence_length or cfg.text_len)
        action_path = batch.action_path
        if action_path is None:
            raise ValueError("LingBot World 2 causal-fast requires `action_path`.")

        c2ws = np.load(os.path.join(action_path, "poses.npy"))
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = ((int(batch.num_frames) - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        img = self._prepare_image_tensor(batch, device)
        h0, w0 = img.shape[1:]
        aspect_ratio = h0 / w0
        lat_h = round(np.sqrt(cfg.max_area * aspect_ratio) // 8 // cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(np.sqrt(cfg.max_area / aspect_ratio) // 8 // cfg.patch_size[2] * cfg.patch_size[2])
        h = lat_h * 8
        w = lat_w * 8
        lat_f = (frame_num - 1) // 4 + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        frames = (lat_f - 1) * 4 + 1
        batch.height = h
        batch.width = w
        batch.num_frames = frames

        seed = int(batch.seed if batch.seed is not None else 42)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=device)

        mask = torch.ones(1, frames, lat_h, lat_w, device=device)
        mask[:, 1:] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        self.scheduler.set_timesteps(cfg.num_train_timesteps, shift=cfg.sample_shift)
        timesteps = self.scheduler.timesteps[list(cfg.timesteps_index)].to(device)
        context = self._prompt_context(batch, device)
        c2ws_plucker_emb = self._prepare_camera(
            action_path,
            c2ws,
            h,
            w,
            lat_f,
            lat_h,
            lat_w,
            chunk_size,
            torch.bfloat16,
            device,
        )
        y = self._encode_condition_video(img, h, w, frames, mask, fastvideo_args).to(device=device,
                                                                                     dtype=torch.bfloat16)

        transformer_dtype = torch.bfloat16
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * cfg.local_attn_size if cfg.local_attn_size > -1 else frame_seqlen * lat_f
        self_kv_cache = self._initialize_self_kv_cache(1, kv_size, transformer_dtype, device)
        cross_kv_cache = self._initialize_crossattn_cache(1, max_sequence_length, transformer_dtype, device)

        self.transformer = self.transformer.to(device)
        self._cross_attn_initialized = False
        pred_latent_chunks = []
        latents_chunk = noise.split(chunk_size, dim=1)
        condition_chunk = y.split(chunk_size, dim=1)
        c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
        max_seq_len = int(math.ceil(chunk_size * lat_h * lat_w // 4))

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            for chunk_id, current_latent in enumerate(latents_chunk):
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                dit_cond_dict = {"c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0)}
                kwargs = {
                    "context": [context[0]],
                    "seq_len": max_seq_len,
                    "y": [current_condition],
                    "dit_cond_dict": dit_cond_dict,
                    "kv_cache": self_kv_cache,
                    "crossattn_cache": cross_kv_cache,
                    "current_start": chunk_id * chunk_size * frame_seqlen,
                    "max_attention_size": kv_size,
                    "frame_seqlen": frame_seqlen,
                }
                x0 = current_latent
                for timestep_idx, timestep_value in enumerate(timesteps):
                    timestep = torch.stack([timestep_value]).to(device)
                    noise_pred = self.transformer(
                        x=[current_latent.to(device)],
                        t=timestep,
                        cross_attn_first_call=not self._cross_attn_initialized,
                        **kwargs,
                    )[0]
                    self._cross_attn_initialized = True
                    x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, timestep_value)
                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1].reshape(1)
                        current_latent = self.scheduler.add_noise(
                            x0,
                            torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype),
                            next_timestep,
                        )
                pred_latent_chunks.append(x0)
                context_timestep = torch.stack([timesteps[-1] * 0.0]).to(device)
                self.transformer(x=[x0], t=context_timestep, cross_attn_first_call=False, **kwargs)

        batch.latents = torch.cat(pred_latent_chunks, dim=1).unsqueeze(0)
        return batch


class LingBotWorld2CausalFastPipeline(LoRAPipeline, ComposedPipelineBase):
    """FastVideo pipeline for LingBot World 2 14B causal-fast I2V generation."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        if "scheduler" not in self.modules:
            self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=1.0, use_dynamic_shifting=False)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up the LingBot World 2 causal-fast pipeline stages."""
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=LingBotWorld2TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(
            stage_name="lingbotworld2_causal_fast_generation_stage",
            stage=LingBotWorld2CausalFastGenerationStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
            ),
        )
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = LingBotWorld2CausalFastPipeline
