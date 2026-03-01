# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import gc
from typing import Any, Literal

import torch

import fastvideo.envs as envs
from fastvideo.configs.sample import SamplingParam
from fastvideo.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.training.training_utils import (
    compute_density_for_timestep_sampling,
    get_sigmas,
    normalize_dit_input,
    shift_timestep,
)
from fastvideo.utils import is_vmoba_available, is_vsa_available, set_random_seed

from fastvideo.distillation.adapters.base import DistillAdapter
from fastvideo.distillation.roles import RoleHandle

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionMetadataBuilder,
    )
    from fastvideo.attention.backends.vmoba import VideoMobaAttentionMetadataBuilder
except Exception:
    VideoSparseAttentionMetadataBuilder = None  # type: ignore[assignment]
    VideoMobaAttentionMetadataBuilder = None  # type: ignore[assignment]


class WanAdapter(DistillAdapter):
    """
    Phase 1 target adapter: provide Wan-specific primitives without calling
    legacy distillation pipeline algorithm helpers (e.g. pipeline-private forward wrappers).
    """

    def __init__(
        self,
        *,
        prompt_handle: RoleHandle,
        training_args: Any,
        noise_scheduler: Any,
        vae: Any,
    ) -> None:
        self.prompt_handle = prompt_handle
        self.training_args = training_args
        self.noise_scheduler = noise_scheduler
        self.vae = vae

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.device = get_local_torch_device()

        self.noise_random_generator: torch.Generator | None = None
        self.noise_gen_cuda: torch.Generator | None = None

        self.negative_prompt_embeds: torch.Tensor | None = None
        self.negative_prompt_attention_mask: torch.Tensor | None = None

        self._init_timestep_mechanics()

    def _get_training_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _init_timestep_mechanics(self) -> None:
        self.timestep_shift = float(self.training_args.pipeline_config.flow_shift)
        self.num_train_timestep = int(self.noise_scheduler.num_train_timesteps)
        self.min_timestep = int(self.training_args.min_timestep_ratio * self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio * self.num_train_timestep)

        boundary_ratio = getattr(self.training_args, "boundary_ratio", None)
        self.boundary_timestep: float | None = (
            float(boundary_ratio) * float(self.num_train_timestep)
            if boundary_ratio is not None
            else None
        )

    @property
    def num_train_timesteps(self) -> int:
        return int(self.num_train_timestep)

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = shift_timestep(timestep, self.timestep_shift, self.num_train_timestep)
        return timestep.clamp(self.min_timestep, self.max_timestep)

    def on_train_start(self) -> None:
        seed = self.training_args.seed
        if seed is None:
            raise ValueError("training_args.seed must be set for distillation")

        global_rank = int(getattr(self.world_group, "rank", 0))
        sp_world_size = int(getattr(self.training_args, "sp_size", 1) or 1)
        if sp_world_size > 1:
            sp_group_seed = int(seed) + (global_rank // sp_world_size)
            set_random_seed(sp_group_seed)
        else:
            set_random_seed(int(seed) + global_rank)

        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.noise_gen_cuda = torch.Generator(device=self.device).manual_seed(int(seed))

        self.ensure_negative_conditioning()

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        """Return RNG generators that should be checkpointed for exact resume."""

        generators: dict[str, torch.Generator] = {}
        if self.noise_random_generator is not None:
            generators["noise_cpu"] = self.noise_random_generator
        if self.noise_gen_cuda is not None:
            generators["noise_cuda"] = self.noise_gen_cuda

        return generators

    def ensure_negative_conditioning(self) -> None:
        if self.negative_prompt_embeds is not None:
            return

        training_args = self.training_args
        world_group = self.world_group
        device = self.device
        dtype = self._get_training_dtype()

        neg_embeds: torch.Tensor | None = None
        neg_mask: torch.Tensor | None = None

        if world_group.rank_in_group == 0:
            sampling_param = SamplingParam.from_pretrained(training_args.model_path)
            negative_prompt = sampling_param.negative_prompt

            args_copy = copy.deepcopy(training_args)
            args_copy.inference_mode = True

            prompt_transformer = self.prompt_handle.require_module("transformer")
            prompt_pipeline = WanPipeline.from_pretrained(
                training_args.model_path,
                args=args_copy,
                inference_mode=True,
                loaded_modules={"transformer": prompt_transformer},
                tp_size=training_args.tp_size,
                sp_size=training_args.sp_size,
                num_gpus=training_args.num_gpus,
                pin_cpu_memory=training_args.pin_cpu_memory,
                dit_cpu_offload=True,
            )

            batch_negative = ForwardBatch(
                data_type="video",
                prompt=negative_prompt,
                prompt_embeds=[],
                prompt_attention_mask=[],
            )
            result_batch = prompt_pipeline.prompt_encoding_stage(  # type: ignore[attr-defined]
                batch_negative,
                training_args,
            )

            neg_embeds = result_batch.prompt_embeds[0].to(device=device, dtype=dtype)
            neg_mask = result_batch.prompt_attention_mask[0].to(device=device, dtype=dtype)

            del prompt_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        meta = torch.zeros((2,), device=device, dtype=torch.int64)
        if world_group.rank_in_group == 0:
            assert neg_embeds is not None
            assert neg_mask is not None
            meta[0] = neg_embeds.ndim
            meta[1] = neg_mask.ndim
        world_group.broadcast(meta, src=0)
        embed_ndim, mask_ndim = (int(meta[0].item()), int(meta[1].item()))

        max_ndim = 8
        embed_shape = torch.full((max_ndim,), -1, device=device, dtype=torch.int64)
        mask_shape = torch.full((max_ndim,), -1, device=device, dtype=torch.int64)
        if world_group.rank_in_group == 0:
            assert neg_embeds is not None
            assert neg_mask is not None
            embed_shape[:embed_ndim] = torch.tensor(
                list(neg_embeds.shape), device=device, dtype=torch.int64
            )
            mask_shape[:mask_ndim] = torch.tensor(
                list(neg_mask.shape), device=device, dtype=torch.int64
            )
        world_group.broadcast(embed_shape, src=0)
        world_group.broadcast(mask_shape, src=0)

        embed_sizes = tuple(int(x) for x in embed_shape[:embed_ndim].tolist())
        mask_sizes = tuple(int(x) for x in mask_shape[:mask_ndim].tolist())

        if world_group.rank_in_group != 0:
            neg_embeds = torch.empty(embed_sizes, device=device, dtype=dtype)
            neg_mask = torch.empty(mask_sizes, device=device, dtype=dtype)
        assert neg_embeds is not None
        assert neg_mask is not None

        world_group.broadcast(neg_embeds, src=0)
        world_group.broadcast(neg_mask, src=0)

        self.negative_prompt_embeds = neg_embeds
        self.negative_prompt_attention_mask = neg_mask

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.noise_random_generator is None:
            raise RuntimeError("WanAdapter.on_train_start() must be called before prepare_batch()")

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=batch_size,
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        return self.noise_scheduler.timesteps[indices].to(device=device)

    def _build_attention_metadata(self, training_batch: TrainingBatch) -> TrainingBatch:
        latents_shape = training_batch.raw_latent_shape
        patch_size = self.training_args.pipeline_config.dit_config.patch_size
        current_vsa_sparsity = training_batch.current_vsa_sparsity
        assert latents_shape is not None
        assert training_batch.timesteps is not None

        if envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            if not is_vsa_available() or VideoSparseAttentionMetadataBuilder is None:
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is VIDEO_SPARSE_ATTN, but fastvideo_kernel "
                    "is not correctly installed or detected."
                )
            training_batch.attn_metadata = VideoSparseAttentionMetadataBuilder().build(  # type: ignore[misc]
                raw_latent_shape=latents_shape[2:5],
                current_timestep=training_batch.timesteps,
                patch_size=patch_size,
                VSA_sparsity=current_vsa_sparsity,
                device=self.device,
            )
        elif envs.FASTVIDEO_ATTENTION_BACKEND == "VMOBA_ATTN":
            if not is_vmoba_available() or VideoMobaAttentionMetadataBuilder is None:
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is VMOBA_ATTN, but fastvideo_kernel "
                    "(or flash_attn>=2.7.4) is not correctly installed."
                )
            moba_params = self.training_args.moba_config.copy()
            moba_params.update(
                {
                    "current_timestep": training_batch.timesteps,
                    "raw_latent_shape": training_batch.raw_latent_shape[2:5],
                    "patch_size": patch_size,
                    "device": self.device,
                }
            )
            training_batch.attn_metadata = VideoMobaAttentionMetadataBuilder().build(**moba_params)  # type: ignore[misc]
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _prepare_dit_inputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        latents = training_batch.latents
        batch_size = latents.shape[0]
        if self.noise_gen_cuda is None:
            raise RuntimeError("WanAdapter.on_train_start() must be called before prepare_batch()")

        noise = torch.randn(
            latents.shape,
            generator=self.noise_gen_cuda,
            device=latents.device,
            dtype=latents.dtype,
        )
        timesteps = self._sample_timesteps(batch_size, latents.device)
        if int(getattr(self.training_args, "sp_size", 1) or 1) > 1:
            self.sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        training_batch.conditional_dict = {
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_attention_mask": training_batch.encoder_attention_mask,
        }

        if self.negative_prompt_embeds is not None and self.negative_prompt_attention_mask is not None:
            neg_embeds = self.negative_prompt_embeds
            neg_mask = self.negative_prompt_attention_mask
            if neg_embeds.shape[0] == 1 and batch_size > 1:
                neg_embeds = neg_embeds.expand(batch_size, *neg_embeds.shape[1:]).contiguous()
            if neg_mask.shape[0] == 1 and batch_size > 1:
                neg_mask = neg_mask.expand(batch_size, *neg_mask.shape[1:]).contiguous()
            training_batch.unconditional_dict = {
                "encoder_hidden_states": neg_embeds,
                "encoder_attention_mask": neg_mask,
            }

        training_batch.latents = training_batch.latents.permute(0, 2, 1, 3, 4)
        return training_batch

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        self.ensure_negative_conditioning()

        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch(current_vsa_sparsity=current_vsa_sparsity)
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        infos = raw_batch.get("info_list")

        if latents_source == "zeros":
            batch_size = encoder_hidden_states.shape[0]
            vae_config = self.training_args.pipeline_config.vae_config.arch_config
            num_channels = vae_config.z_dim
            spatial_compression_ratio = vae_config.spatial_compression_ratio
            latent_height = self.training_args.num_height // spatial_compression_ratio
            latent_width = self.training_args.num_width // spatial_compression_ratio
            latents = torch.zeros(
                batch_size,
                num_channels,
                self.training_args.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError(
                    "vae_latent not found in batch and latents_source='data'"
                )
            latents = raw_batch["vae_latent"]
            latents = latents[:, :, : self.training_args.num_latent_t]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        training_batch.latents = latents
        training_batch.encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(device, dtype=dtype)
        training_batch.infos = infos

        training_batch.latents = normalize_dit_input("wan", training_batch.latents, self.vae)
        training_batch = self._prepare_dit_inputs(training_batch)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        b, t = clean_latents.shape[:2]
        noisy = self.noise_scheduler.add_noise(
            clean_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep,
        ).unflatten(0, (b, t))
        return noisy

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for Wan distillation")
        return {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
            "return_dict": False,
        }

    def _get_transformer(self, handle: RoleHandle, timestep: torch.Tensor) -> torch.nn.Module:
        transformer = handle.require_module("transformer")
        transformer_2 = handle.modules.get("transformer_2")
        if (
            transformer_2 is not None
            and self.boundary_timestep is not None
            and float(timestep.item()) < float(self.boundary_timestep)
        ):
            return transformer_2
        return transformer

    def predict_x0(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        device_type = self.device.type
        dtype = noisy_latents.dtype
        text_dict = batch.conditional_dict if conditional else getattr(batch, "unconditional_dict", None)
        if text_dict is None:
            raise RuntimeError("Missing unconditional_dict; ensure_negative_conditioning() may have failed")

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            input_kwargs = self._build_distill_input_kwargs(noisy_latents, timestep, text_dict)
            transformer = self._get_transformer(handle, timestep)
            pred_noise = transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
            pred_x0 = pred_noise_to_pred_video(
                pred_noise=pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latents.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler,
            ).unflatten(0, pred_noise.shape[:2])
        return pred_x0

    def predict_noise(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        device_type = self.device.type
        dtype = noisy_latents.dtype
        text_dict = batch.conditional_dict if conditional else getattr(batch, "unconditional_dict", None)
        if text_dict is None:
            raise RuntimeError("Missing unconditional_dict; ensure_negative_conditioning() may have failed")

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            input_kwargs = self._build_distill_input_kwargs(noisy_latents, timestep, text_dict)
            transformer = self._get_transformer(handle, timestep)
            pred_noise = transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
        return pred_noise

    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        timesteps, attn_metadata = ctx
        with set_forward_context(current_timestep=timesteps, attn_metadata=attn_metadata):
            (loss / max(1, int(grad_accum_rounds))).backward()
