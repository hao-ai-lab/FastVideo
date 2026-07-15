# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any

import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.attention import LocalAttention
from fastvideo.attention.backends.nabla import NablaAttentionMetadataBuilder
from fastvideo.attention.selector import backend_name_to_enum
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader, VAELoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vision_utils import normalize, numpy_to_pt, pil_to_numpy, resize
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.encoding import EncodingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class Kandinsky5LatentPreparationStage(PipelineStage):

    def __init__(self, scheduler, transformer) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("height and width must be provided for Kandinsky5.")
        height = int(batch.height)
        width = int(batch.width)
        num_frames = int(batch.num_frames)

        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size

        if num_frames % temporal_ratio != 1:
            num_frames = num_frames // temporal_ratio * temporal_ratio + 1
            batch.num_frames = num_frames

        required_divisor_h = spatial_ratio * patch_size[1]
        required_divisor_w = spatial_ratio * patch_size[2]
        if height % required_divisor_h != 0 or width % required_divisor_w != 0:
            raise ValueError(f"Kandinsky5 height must be divisible by {required_divisor_h} and width by "
                             f"{required_divisor_w}; "
                             f"got height={height}, width={width}.")

        # NABLA sparse attention (Pro checkpoints) reshapes the post-patch grid
        # into 8x8 blocks; validate here instead of crashing mid-denoise after
        # all the encoding work is done.
        arch_cfg = getattr(self.transformer, "config", None) or fastvideo_args.pipeline_config.dit_config.arch_config
        if getattr(arch_cfg, "attention_type", "regular") == "nabla":
            nabla_divisor_h = required_divisor_h * 8
            nabla_divisor_w = required_divisor_w * 8
            if height % nabla_divisor_h != 0 or width % nabla_divisor_w != 0:
                raise ValueError(f"Kandinsky5 NABLA checkpoints require height divisible by {nabla_divisor_h} and "
                                 f"width divisible by {nabla_divisor_w}; "
                                 f"got height={height}, width={width}.")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        batch_size *= batch.num_videos_per_prompt

        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        device = get_local_torch_device()
        num_latent_frames = (num_frames - 1) // temporal_ratio + 1
        num_channels = getattr(
            self.transformer,
            "in_visual_dim",
            fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim,
        )
        shape = (
            batch_size,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
            num_channels,
        )

        if isinstance(batch.generator, list) and len(batch.generator) != batch_size:
            raise ValueError(f"generator list length {len(batch.generator)} does not match batch size {batch_size}.")

        visual_cond = getattr(self.transformer, "visual_cond", False)

        if batch.latents is None:
            latents = randn_tensor(shape, generator=batch.generator, device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            valid_shapes = [shape]
            if visual_cond:
                valid_shapes.append((*shape[:-1], num_channels * 2 + 1))
            if tuple(batch.latents.shape) not in valid_shapes:
                raise ValueError(f"Provided latents shape {list(batch.latents.shape)} does not match expected "
                                 f"Kandinsky5 latent shape(s): {[list(s) for s in valid_shapes]}.")
            latents = batch.latents.to(device=device, dtype=dtype)

        if visual_cond and latents.shape[-1] == num_channels:
            cond = torch.zeros_like(latents)
            cond_mask = torch.zeros(
                (*latents.shape[:-1], 1),
                device=latents.device,
                dtype=latents.dtype,
            )
            latents = torch.cat([latents, cond, cond_mask], dim=-1)

        # I2V image conditioning is placed by Kandinsky5ImageEncodingStage,
        # which runs AFTER this stage so the initial noise is the generator's
        # first draw (matching the official kandinskylab/kandinsky-5 order).
        batch.latents = latents
        batch.raw_latent_shape = (
            batch_size,
            num_channels,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
        )
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky5DenoisingStage(PipelineStage):

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @staticmethod
    def _scale_factor(height: int, width: int) -> tuple[float, float, float]:
        if 480 <= height <= 854 and 480 <= width <= 854:
            return (1.0, 2.0, 2.0)
        return (1.0, 3.16, 3.16)

    @staticmethod
    def _text_rope_pos(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        seq_len = int(mask.sum(1).max().item())
        return torch.arange(seq_len, device=device)

    # Backends whose selection is all-or-nothing: fastvideo.platforms.cuda
    # raises ImportError immediately if the kernel isn't available, rather
    # than the generic "requested backend unsupported by this layer, falling
    # back to automatic selection" path _cached_get_attn_backend takes for
    # most other backend names (see fastvideo/attention/selector.py). Both
    # values are also always in Kandinsky5ArchConfig's
    # _supported_attention_backends, so that generic fallback never applies
    # to them here. That combination makes an exact backend match a safe,
    # false-positive-free signal *only* for these two -- any other
    # FASTVIDEO_ATTENTION_BACKEND value (e.g. a backend meant for a
    # different model family sharing the same process/env var) can
    # legitimately resolve to something other than what was requested.
    _STRICT_BACKENDS = frozenset({
        AttentionBackendEnum.ATTN_QAT_TRAIN,
        AttentionBackendEnum.ATTN_QAT_INFER,
    })

    def _assert_local_attention_backend_engaged(self) -> None:
        """Guard against ``LocalAttention`` silently falling back to SDPA.

        ``Kandinsky5Attention.forward`` catches the ``AssertionError``
        ``LocalAttention`` raises when no pipeline forward context is set and
        falls back to plain ``F.scaled_dot_product_attention`` -- silently
        skipping whichever kernel ``FASTVIDEO_ATTENTION_BACKEND`` requested
        (the fake-quantized ``ATTN_QAT_TRAIN``/``ATTN_QAT_INFER`` kernels).
        ``LocalAttention.backend`` is resolved once at module construction,
        independent of whether a forward context is later set, so this only
        catches a backend that failed to resolve to what the env var
        requested -- it does not prove forward context was present for a
        given forward call. Pair it with always wrapping the actual
        transformer call in ``set_forward_context`` (see the denoising loops
        below), which is what prevents the runtime fallback.

        Scoped to ``_STRICT_BACKENDS``: other backend names can legitimately
        differ from what ``LocalAttention.backend`` resolved to (silent,
        logged, intentional fallback when unsupported by this layer -- see
        ``_STRICT_BACKENDS``'s comment), so asserting on those would flag
        expected behavior as a bug.
        """
        backend_env = envs.FASTVIDEO_ATTENTION_BACKEND
        if not backend_env:
            return
        expected = backend_name_to_enum(backend_env)
        if expected is None or expected not in self._STRICT_BACKENDS:
            return
        for module in self.transformer.modules():
            if isinstance(module, LocalAttention):
                assert module.backend == expected, (
                    f"Kandinsky5 local attention resolved to backend {module.backend}, expected "
                    f"{expected} from FASTVIDEO_ATTENTION_BACKEND={backend_env}. This likely means "
                    "LocalAttention's missing-forward-context guard silently fell back to SDPA.")
                return

    def _resolve_target_dtype(self, fastvideo_args: FastVideoArgs) -> torch.dtype:
        """Resolve the transformer's actual compute dtype.

        Trust ``pipeline_config.dit_precision`` directly for a normal,
        non-FSDP load -- ``TransformerLoader.load()`` asserts every
        parameter matches it exactly (``fastvideo/models/loader/
        component_loader.py``), so this holds for standalone T2V/I2V
        inference regardless of which precision (including fp32) was
        requested.

        FSDP2-wrapped modules are the one case where that assertion doesn't
        carry forward: ``maybe_load_fsdp_model`` hardcodes its
        ``MixedPrecisionPolicy`` to ``param_dtype=torch.bfloat16`` for every
        FSDP-wrapped load, independent of ``dit_precision`` -- this covers
        both the live transformer ``ValidationCallback`` reuses from
        training (whose ``pipeline_config.dit_precision`` reflects the
        fp32 master-weight load dtype, not the actual bf16 compute dtype)
        and multi-GPU ``use_fsdp_inference=True`` runs. Detect the real
        compute dtype from the parameters only in that case, mirroring the
        Cosmos 2.5 denoising stage.
        """
        declared_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        try:
            from torch.distributed.fsdp import FSDPModule
        except Exception:  # pragma: no cover - FSDP not always available
            return declared_dtype

        if not isinstance(self.transformer, FSDPModule):
            return declared_dtype

        target_dtype = torch.bfloat16  # safe default
        for p in self.transformer.parameters():
            if p.dtype != torch.float32:
                target_dtype = p.dtype
                break
        return target_dtype

    @staticmethod
    def fast_sta_nabla(
        T: int,
        H: int,
        W: int,
        wT: int = 3,
        wH: int = 3,
        wW: int = 3,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """
        Create a sparse temporal attention (STA) mask for efficient video generation.

        This method generates a mask that limits attention to nearby frames and spatial positions, reducing
        computational complexity for video generation.

        Args:
            T (int): Number of temporal frames
            H (int): Height in latent space
            W (int): Width in latent space
            wT (int): Temporal attention window size
            wH (int): Height attention window size
            wW (int): Width attention window size
            device (str): Device to create tensor on

        Returns:
            torch.Tensor: Sparse attention mask of shape (T*H*W, T*H*W)
        """
        max_extent = int(torch.tensor([T, H, W], device=device).amax().item())
        r = torch.arange(0, max_extent, 1, dtype=torch.int16, device=device)
        mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
        sta_t, sta_h, sta_w = (
            mat[:T, :T].flatten(),
            mat[:H, :H].flatten(),
            mat[:W, :W].flatten(),
        )
        sta_t = sta_t <= wT // 2
        sta_h = sta_h <= wH // 2
        sta_w = sta_w <= wW // 2
        sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
        sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
        return sta.reshape(T * H * W, T * H * W)

    def get_sparse_params(self, sample: torch.Tensor, device: torch.device) -> dict[str, Any] | None:
        """
        Generate sparse attention parameters for the transformer based on sample dimensions.

        This method computes the sparse attention configuration needed for efficient video processing in the
        transformer model.

        Args:
            sample (torch.Tensor): Input sample tensor
            device (torch.device): Device to place tensors on

        Returns:
            Dict: Dictionary containing sparse attention parameters
        """
        assert self.transformer.config.patch_size[0] == 1
        _, T, H, W, _ = sample.shape
        T, H, W = (
            T // self.transformer.config.patch_size[0],
            H // self.transformer.config.patch_size[1],
            W // self.transformer.config.patch_size[2],
        )
        if self.transformer.config.attention_type == "nabla":
            sta_mask = self.fast_sta_nabla(
                T,
                H // 8,
                W // 8,
                self.transformer.config.attention_wT,
                self.transformer.config.attention_wH,
                self.transformer.config.attention_wW,
                device=device,
            )

            sparse_params = {
                "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
                "attention_type": self.transformer.config.attention_type,
                "to_fractal": True,
                "P": self.transformer.config.attention_P,
                "wT": self.transformer.config.attention_wT,
                "wW": self.transformer.config.attention_wW,
                "wH": self.transformer.config.attention_wH,
                "add_sta": self.transformer.config.attention_add_sta,
                "visual_shape": (T, H, W),
                "method": self.transformer.config.attention_method,
            }
        else:
            sparse_params = None

        return sparse_params

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None:
            raise ValueError("timesteps must be prepared before Kandinsky5 denoising.")
        if batch.latents is None:
            raise ValueError("latents must be prepared before Kandinsky5 denoising.")
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(fastvideo_args.model_paths["transformer"], fastvideo_args)
            fastvideo_args.model_loaded["transformer"] = True
        self._assert_local_attention_backend_engaged()

        device = get_local_torch_device()
        target_dtype = self._resolve_target_dtype(fastvideo_args)
        autocast_enabled = target_dtype != torch.float32 and not fastvideo_args.disable_autocast
        latents = batch.latents
        num_channels = getattr(
            self.transformer,
            "in_visual_dim",
            fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim,
        )

        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        pooled = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        if batch.prompt_attention_mask is None or not batch.prompt_attention_mask:
            raise ValueError("Kandinsky5 requires Qwen prompt attention masks.")
        text_rope_pos = self._text_rope_pos(batch.prompt_attention_mask[0].to(device), device)

        neg_prompt_embeds = None
        neg_pooled = None
        negative_text_rope_pos = None
        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg_prompt_embeds = batch.negative_prompt_embeds[0].to(device=device, dtype=target_dtype)
            neg_pooled = batch.negative_prompt_embeds[1].to(device=device, dtype=target_dtype)
            if batch.negative_attention_mask is None or not batch.negative_attention_mask:
                raise ValueError("Kandinsky5 requires Qwen negative attention masks for CFG.")
            negative_text_rope_pos = self._text_rope_pos(batch.negative_attention_mask[0].to(device), device)

        height = int(batch.height)
        width = int(batch.width)
        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        num_latent_frames = (int(batch.num_frames) - 1) // temporal_ratio + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // spatial_ratio // 2, device=device),
            torch.arange(width // spatial_ratio // 2, device=device),
        ]
        scale_factor = self._scale_factor(height, width)

        sparse_params = self.get_sparse_params(latents, device)

        # I2V keeps the first (conditioning) frame fixed during denoising.
        # Key off the actual image conditioning, not transformer.visual_cond:
        # official T2V checkpoints also ship visual_cond=True, and skipping
        # frame 0 for them leaves it as undenoised noise.
        cond_frames = 1 if batch.image_latent is not None else 0

        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        with tqdm(total=batch.num_inference_steps, desc="Kandinsky5 Denoising") as progress_bar:
            for i, timestep in enumerate(batch.timesteps):
                if hasattr(self, "interrupt") and self.interrupt:
                    break

                t_expand = timestep.unsqueeze(0).repeat(latents.shape[0]).to(device=device, dtype=target_dtype)
                attn_metadata = None
                if sparse_params is not None:
                    attn_metadata = NablaAttentionMetadataBuilder().build(
                        current_timestep=i,
                        sta_mask=sparse_params["sta_mask"],
                        P=sparse_params["P"],
                        visual_shape=sparse_params["visual_shape"],
                    )
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled)
                                if device.type == "cuda" else contextlib.nullcontext())
                with set_forward_context(current_timestep=i, attn_metadata=attn_metadata,
                                         forward_batch=batch), autocast_ctx:
                    pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype=target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        timestep=t_expand,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=sparse_params,
                        return_dict=True,
                    ).sample

                    if neg_prompt_embeds is not None and neg_pooled is not None:
                        uncond_pred_velocity = self.transformer(
                            hidden_states=latents.to(dtype=target_dtype),
                            encoder_hidden_states=neg_prompt_embeds,
                            pooled_projections=neg_pooled,
                            timestep=t_expand,
                            visual_rope_pos=visual_rope_pos,
                            text_rope_pos=negative_text_rope_pos,
                            scale_factor=scale_factor,
                            sparse_params=sparse_params,
                            return_dict=True,
                        ).sample
                        pred_velocity = uncond_pred_velocity + batch.guidance_scale * (pred_velocity -
                                                                                       uncond_pred_velocity)

                latents[:, cond_frames:, :, :, :num_channels] = self.scheduler.step(
                    pred_velocity[:, cond_frames:],
                    timestep,
                    latents[:, cond_frames:, :, :, :num_channels],
                    return_dict=False,
                )[0]

                if batch.return_trajectory_latents:
                    trajectory_timesteps.append(timestep)
                    # latents is mutated in place, so snapshot a channels-first copy.
                    trajectory_latents.append(latents[..., :num_channels].permute(0, 4, 1, 2, 3).cpu())

                if i == len(batch.timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if trajectory_latents:
            batch.trajectory_latents = torch.stack(trajectory_latents, dim=1)
            batch.trajectory_timesteps = torch.stack(trajectory_timesteps, dim=0).cpu()

        batch.latents = latents[:, :, :, :, :num_channels]
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.min_list_length(2))
        return result


class Kandinsky5DmdDenoisingStage(Kandinsky5DenoisingStage):
    """DMD (few fixed steps, no CFG) variant of Kandinsky5DenoisingStage.

    Reuses the parent's RoPE/scale_factor/sparse-params helpers; only the
    denoising loop differs: a fixed short timestep schedule
    (``pipeline_config.dmd_denoising_steps``) with a single forward pass per
    step and no classifier-free-guidance branch, matching DMD's distilled
    few-step generator.

    ``dmd_denoising_steps`` values (e.g. ``[1000, 750, 500, 250]``) are
    literal *final* target timesteps -- that's how ``DMD2Method`` on the
    training side resolves them: nearest-sigma lookup against a scheduler
    that has never had ``set_timesteps`` called on it, so e.g. "750" maps to
    sigma 0.75. This stage therefore keeps a private
    ``FlowMatchEulerDiscreteScheduler`` (same ``shift`` as the pipeline
    scheduler) and drives it directly via predict-x0 + re-noise, exactly
    mirroring ``DMD2Method._student_rollout``'s "simulate" branch and Wan's
    own ``DmdDenoisingStage``. It must NOT reuse the pipeline's shared
    ``scheduler`` object through ``scheduler.step()``: by the time this
    stage runs, ``TimestepPreparationStage`` has already called
    ``scheduler.set_timesteps(timesteps=dmd_denoising_steps)`` on it, which
    re-applies the flow-match ``shift`` warp on top of values that are
    already final (e.g. sigma 0.75 -> 0.9375 at shift=5) -- a double shift
    that leaves every step far noisier than the student was trained for, so
    the sampled video is still mostly noise after the last step instead of
    converged.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self._sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=scheduler.shift)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("latents must be prepared before Kandinsky5 DMD denoising.")
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(fastvideo_args.model_paths["transformer"], fastvideo_args)
            fastvideo_args.model_loaded["transformer"] = True
        self._assert_local_attention_backend_engaged()

        device = get_local_torch_device()
        target_dtype = self._resolve_target_dtype(fastvideo_args)
        autocast_enabled = target_dtype != torch.float32 and not fastvideo_args.disable_autocast
        latents = batch.latents
        num_channels = getattr(
            self.transformer,
            "in_visual_dim",
            fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim,
        )

        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        pooled = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        if batch.prompt_attention_mask is None or not batch.prompt_attention_mask:
            raise ValueError("Kandinsky5 DMD requires Qwen prompt attention masks.")
        text_rope_pos = self._text_rope_pos(batch.prompt_attention_mask[0].to(device), device)

        height = int(batch.height)
        width = int(batch.width)
        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        num_latent_frames = (int(batch.num_frames) - 1) // temporal_ratio + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // spatial_ratio // 2, device=device),
            torch.arange(width // spatial_ratio // 2, device=device),
        ]
        scale_factor = self._scale_factor(height, width)
        sparse_params = self.get_sparse_params(latents, device)

        dmd_steps = fastvideo_args.pipeline_config.dmd_denoising_steps
        if not dmd_steps:
            raise ValueError("Kandinsky5 DMD denoising requires "
                             "pipeline_config.dmd_denoising_steps to be set.")
        timesteps = torch.tensor(dmd_steps, dtype=torch.long, device=device)

        # I2V keeps the first (conditioning) frame fixed during denoising.
        # Key off the actual image conditioning, not transformer.visual_cond:
        # official T2V checkpoints also ship visual_cond=True, and skipping
        # frame 0 for them leaves it as undenoised noise (see the same fix
        # in Kandinsky5DenoisingStage.forward() above).
        cond_frames = 1 if batch.image_latent is not None else 0

        with tqdm(total=len(timesteps), desc="Kandinsky5 DMD Denoising") as progress_bar:
            for i, timestep in enumerate(timesteps):
                if hasattr(self, "interrupt") and self.interrupt:
                    continue

                t_expand = timestep.unsqueeze(0).repeat(latents.shape[0]).to(device=device, dtype=target_dtype)
                attn_metadata = None
                if sparse_params is not None:
                    attn_metadata = NablaAttentionMetadataBuilder().build(
                        current_timestep=i,
                        sta_mask=sparse_params["sta_mask"],
                        P=sparse_params["P"],
                        visual_shape=sparse_params["visual_shape"],
                    )
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled)
                                if device.type == "cuda" else contextlib.nullcontext())
                with set_forward_context(current_timestep=i, attn_metadata=attn_metadata,
                                         forward_batch=batch), autocast_ctx:
                    pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype=target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        timestep=t_expand,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=sparse_params,
                        return_dict=True,
                    ).sample

                # Channel-last [B, T', H, W, C] -> channel-first [B, T', C, H, W]
                # to match pred_noise_to_pred_video/add_noise's expected layout
                # (same convention as Kandinsky5Model.add_noise/predict_x0).
                sample_cf = latents[:, cond_frames:, :, :, :num_channels].permute(0, 1, 4, 2, 3)
                velocity_cf = pred_velocity[:, cond_frames:].permute(0, 1, 4, 2, 3)
                b, t = sample_cf.shape[:2]
                step_timestep = timestep.reshape(1).to(device=device)
                pred_x0_cf = pred_noise_to_pred_video(
                    pred_noise=velocity_cf.flatten(0, 1),
                    noise_input_latent=sample_cf.flatten(0, 1),
                    timestep=step_timestep,
                    scheduler=self._sample_scheduler,
                ).unflatten(0, (b, t))

                if i < len(timesteps) - 1:
                    next_timestep = timesteps[i + 1].reshape(1).to(device=device)
                    noise = randn_tensor(
                        sample_cf.shape,
                        generator=batch.generator,
                        device=device,
                        dtype=pred_x0_cf.dtype,
                    )
                    next_cf = self._sample_scheduler.add_noise(
                        pred_x0_cf.flatten(0, 1),
                        noise.flatten(0, 1),
                        next_timestep,
                    ).unflatten(0, (b, t))
                else:
                    next_cf = pred_x0_cf

                latents[:, cond_frames:, :, :, :num_channels] = next_cf.permute(0, 1, 3, 4, 2).to(latents.dtype)
                progress_bar.update()

        batch.latents = latents[:, :, :, :, :num_channels]
        return batch


class Kandinsky5DecodingStage(DecodingStage):

    def __init__(self, vae: ParallelTiledVAE, pipeline=None) -> None:
        super().__init__(vae=vae, pipeline=pipeline)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("latents must be available before Kandinsky5 decoding.")
        # Kandinsky5 latents are channels-last [B, T, H, W, C]; the base stage
        # (and the trajectory latents recorded by the denoising stage) work
        # channels-first.
        batch.latents = batch.latents.permute(0, 4, 1, 2, 3).contiguous()
        return super().forward(batch, fastvideo_args)


class Kandinsky5ImageEncodingStage(EncodingStage):
    """Encode the conditioning image into a VAE latent for I2V."""

    def __init__(self, vae: ParallelTiledVAE, pipeline=None) -> None:
        super().__init__(vae=vae)

    @staticmethod
    def _preprocess(image, height: int, width: int) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            image = resize(image, height, width)
            image = numpy_to_pt(pil_to_numpy(image))  # always lands in [0, 1]
            return normalize(image)  # [0, 1] -> [-1, 1]
        # Tensor input: no reliable way to tell [0, 1] from an already
        # normalized [-1, 1] tensor whose values happen to be non-negative,
        # so mirror diffusers' heuristic and say what we assumed.
        if image.min() >= 0:
            logger.warning("Kandinsky5 conditioning image tensor has no negative values; "
                           "assuming range [0, 1] and normalizing to [-1, 1]. "
                           "Pass a [-1, 1] tensor with negative values to skip normalization.")
            image = normalize(image)  # [0, 1] -> [-1, 1]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-2:] != (height, width):
            image = torch.nn.functional.interpolate(image.float(),
                                                    size=(height, width),
                                                    mode="bilinear",
                                                    antialias=True)
        return image

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.pil_image is None:
            raise ValueError("Kandinsky5 I2V requires an input image.")

        if not fastvideo_args.model_loaded["vae"]:
            vae = getattr(self, "vae", None)
            if vae is None:
                loader = VAELoader()
                vae = loader.load(fastvideo_args.model_paths["vae"], fastvideo_args)
                self.vae = vae
            fastvideo_args.model_loaded["vae"] = True

        device = get_local_torch_device()
        vae = self.vae.to(device)
        self.vae = vae
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = vae_dtype != torch.float32 and not fastvideo_args.disable_autocast

        # [B, C, H, W] -> [B, C, 1, H, W]
        image = self._preprocess(batch.pil_image, int(batch.height), int(batch.width))
        image = image.to(device=device, dtype=torch.float32).unsqueeze(2)

        # Encode the single conditioning frame without tiling (matches diffusers).
        # The untested causal-VAE spatial_tiled_encode path corrupts the latent.
        prev_use_tiling = vae.use_tiling
        vae.use_tiling = False
        try:
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if not vae_autocast_enabled:
                    image = image.to(vae_dtype)
                # Sample with the batch generator (diffusers parity); mode()
                # would make seed-for-seed reproduction of the reference
                # pipeline impossible.
                generator = batch.generator
                if isinstance(generator, list) and len(generator) != image.shape[0]:
                    generator = generator[0]
                image_latent = vae.encode(image).sample(generator=generator)
        finally:
            vae.use_tiling = prev_use_tiling

        image_latent = image_latent * vae.scaling_factor
        # [B, C, 1, H, W] -> [B, 1, H, W, C] to match channels-last latents
        batch.image_latent = image_latent.permute(0, 2, 3, 4, 1).contiguous()

        # Place the conditioning latent into the prepared latents: frame 0 of
        # the main channels, the visual_cond channel block, and the mask.
        # NOTE: the official kandinsky-5 repo leaves the visual_cond block
        # zeros (generation_utils.py generate()), while the diffusers port
        # copies the image latent into it. A same-seed A/B on
        # Kandinsky-5.0-I2V-Pro-distilled-5s-Diffusers showed the Diffusers
        # export requires the copy: zeroing the block produces smeared faces
        # mid-video. Keep the diffusers semantics for Diffusers-format
        # checkpoints.
        latents = batch.latents
        image_latent = batch.image_latent.to(device=latents.device, dtype=latents.dtype)
        num_channels = image_latent.shape[-1]
        latents[:, 0:1, :, :, :num_channels] = image_latent
        if latents.shape[-1] > num_channels:
            latents[:, 0:1, :, :, num_channels:2 * num_channels] = image_latent
            latents[:, 0:1, :, :, 2 * num_channels:] = 1.0
        batch.latents = latents

        if fastvideo_args.vae_cpu_offload:
            vae.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, V.not_none)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        # This stage runs after latent preparation and writes into its output.
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky5NormalizationStage(PipelineStage):
    """Normalize the first latent frames to reduce I2V conditioning artifacts."""

    COND_FRAMES = 4
    REFERENCE_FRAMES = 5

    @staticmethod
    def _adaptive_mean_std(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        source_mean = source.mean(dim=(1, 2, 3, 4), keepdim=True)
        source_std = source.std(dim=(1, 2, 3, 4), keepdim=True)
        # Magic constants limit how far the first frames may drift.
        ref_mean = torch.clamp(reference.mean(dim=(1, 2, 3, 4), keepdim=True), source_mean - 0.05, source_mean + 0.1)
        ref_std = torch.clamp(reference.std(dim=(1, 2, 3, 4), keepdim=True), source_std - 0.1, source_std + 0.25)
        normalized = (source - source_mean) / source_std
        return normalized * ref_std + ref_mean

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latents = batch.latents
        n = self.COND_FRAMES
        if latents is None or latents.shape[1] <= n:
            return batch

        reference = latents[:, n:n + min(self.REFERENCE_FRAMES, latents.shape[1] - 1)]
        latents[:, :n] = self._adaptive_mean_std(latents[:, :n].clone(), reference)
        batch.latents = latents
        return batch
