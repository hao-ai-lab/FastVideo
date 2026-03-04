# SPDX-License-Identifier: Apache-2.0

"""WanGame bidirectional model plugin (per-role instance)."""

from __future__ import annotations

import copy
from typing import Any, Literal, TYPE_CHECKING

import torch

import fastvideo.envs as envs
from fastvideo.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing,
)
from fastvideo.training.training_utils import (
    compute_density_for_timestep_sampling,
    get_sigmas,
    normalize_dit_input,
    shift_timestep,
)
from fastvideo.utils import (
    is_vmoba_available,
    is_vsa_available,
    set_random_seed,
)

from fastvideo.distillation.models.base import ModelBase
from fastvideo.distillation.utils.module_state import (
    apply_trainable,
)
from fastvideo.distillation.utils.moduleloader import (
    load_module_from_path,
)

if TYPE_CHECKING:
    from fastvideo.distillation.utils.distill_config import (
        DistillTrainingConfig,
    )

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionMetadataBuilder,
    )
    from fastvideo.attention.backends.vmoba import (
        VideoMobaAttentionMetadataBuilder,
    )
except Exception:
    VideoSparseAttentionMetadataBuilder = None  # type: ignore[assignment]
    VideoMobaAttentionMetadataBuilder = None  # type: ignore[assignment]


class WanGameModel(ModelBase):
    """WanGame per-role model: owns transformer + noise_scheduler."""

    _transformer_cls_name: str = (
        "WanGameActionTransformer3DModel"
    )

    def __init__(
        self,
        *,
        init_from: str,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 3.0,
        enable_gradient_checkpointing_type: str
        | None = None,
    ) -> None:
        self._init_from = str(init_from)
        self._trainable = bool(trainable)

        self.transformer = self._load_transformer(
            init_from=self._init_from,
            trainable=self._trainable,
            disable_custom_init_weights=(
                disable_custom_init_weights
            ),
            enable_gradient_checkpointing_type=(
                enable_gradient_checkpointing_type
            ),
        )

        self.noise_scheduler = (
            FlowMatchEulerDiscreteScheduler(
                shift=float(flow_shift)
            )
        )

        # Filled by init_preprocessors (student only).
        self.vae: Any = None
        self.training_config: DistillTrainingConfig | None = (
            None
        )
        self.dataloader: Any = None
        self.validator: Any = None
        self.start_step: int = 0

        self.world_group: Any = None
        self.sp_group: Any = None
        self.device: Any = None

        self.noise_random_generator: (
            torch.Generator | None
        ) = None
        self.noise_gen_cuda: torch.Generator | None = None

        self.timestep_shift: float = float(flow_shift)
        self.num_train_timestep: int = int(
            self.noise_scheduler.num_train_timesteps
        )
        self.min_timestep: int = 0
        self.max_timestep: int = self.num_train_timestep

    def _load_transformer(
        self,
        *,
        init_from: str,
        trainable: bool,
        disable_custom_init_weights: bool,
        enable_gradient_checkpointing_type: str | None,
    ) -> torch.nn.Module:
        transformer = load_module_from_path(
            model_path=init_from,
            module_type="transformer",
            disable_custom_init_weights=(
                disable_custom_init_weights
            ),
            override_transformer_cls_name=(
                self._transformer_cls_name
            ),
        )
        transformer = apply_trainable(
            transformer, trainable=trainable
        )
        if (
            trainable
            and enable_gradient_checkpointing_type
        ):
            transformer = apply_activation_checkpointing(
                transformer,
                checkpointing_type=(
                    enable_gradient_checkpointing_type
                ),
            )
        return transformer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(
        self, training_config: DistillTrainingConfig
    ) -> None:
        """Load VAE, build dataloader, seed RNGs."""
        self.training_config = training_config

        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.device = get_local_torch_device()

        self._init_timestep_mechanics()

        # Optional validator.
        validation_cfg = getattr(
            training_config, "_validation_cfg", None
        )
        if validation_cfg:
            validation_enabled = bool(
                validation_cfg.get(
                    "enabled", bool(validation_cfg)
                )
            )
            if validation_enabled:
                from fastvideo.distillation.validators.wangame import (
                    WanGameValidator,
                )

                self.validator = WanGameValidator(
                    training_config=training_config
                )

        from fastvideo.dataset.dataloader.schema import (
            pyarrow_schema_wangame,
        )
        from fastvideo.distillation.utils.dataloader import (
            build_parquet_wangame_train_dataloader,
        )

        self.dataloader = (
            build_parquet_wangame_train_dataloader(
                training_config.data,
                parquet_schema=pyarrow_schema_wangame,
            )
        )
        self.start_step = 0

    # ------------------------------------------------------------------
    # ModelBase overrides: timestep helpers
    # ------------------------------------------------------------------

    @property
    def num_train_timesteps(self) -> int:
        return int(self.num_train_timestep)

    def shift_and_clamp_timestep(
        self, timestep: torch.Tensor
    ) -> torch.Tensor:
        timestep = shift_timestep(
            timestep,
            self.timestep_shift,
            self.num_train_timestep,
        )
        return timestep.clamp(
            self.min_timestep, self.max_timestep
        )

    # ------------------------------------------------------------------
    # ModelBase overrides: lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        seed = tc.data.seed
        if seed is None:
            raise ValueError(
                "training.data.seed must be set "
                "for distillation"
            )

        global_rank = int(
            getattr(self.world_group, "rank", 0)
        )
        sp_world_size = int(tc.distributed.sp_size or 1)
        if sp_world_size > 1:
            sp_group_seed = int(seed) + (
                global_rank // sp_world_size
            )
            set_random_seed(sp_group_seed)
        else:
            set_random_seed(int(seed) + global_rank)

        self.noise_random_generator = torch.Generator(
            device="cpu"
        ).manual_seed(int(seed))
        self.noise_gen_cuda = torch.Generator(
            device=self.device
        ).manual_seed(int(seed))

    def get_rng_generators(
        self,
    ) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}
        if self.noise_random_generator is not None:
            generators["noise_cpu"] = (
                self.noise_random_generator
            )
        if self.noise_gen_cuda is not None:
            generators["noise_cuda"] = self.noise_gen_cuda
        return generators

    # ------------------------------------------------------------------
    # ModelBase overrides: runtime primitives
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch(
            current_vsa_sparsity=current_vsa_sparsity
        )
        infos = raw_batch.get("info_list")

        if latents_source == "zeros":
            clip_feature = raw_batch["clip_feature"]
            batch_size = int(clip_feature.shape[0])
            vae_config = (
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            num_channels = int(vae_config.z_dim)
            spatial_compression_ratio = int(
                vae_config.spatial_compression_ratio
            )
            latent_height = (
                int(tc.data.num_height)
                // spatial_compression_ratio
            )
            latent_width = (
                int(tc.data.num_width)
                // spatial_compression_ratio
            )
            latents = torch.zeros(
                batch_size,
                num_channels,
                int(tc.data.num_latent_t),
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError(
                    "vae_latent not found in batch "
                    "and latents_source='data'"
                )
            latents = raw_batch["vae_latent"]
            latents = latents[
                :, :, : tc.data.num_latent_t
            ]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown latents_source: "
                f"{latents_source!r}"
            )

        if "clip_feature" not in raw_batch:
            raise ValueError(
                "clip_feature must be present for WanGame"
            )
        image_embeds = raw_batch["clip_feature"].to(
            device, dtype=dtype
        )

        if "first_frame_latent" not in raw_batch:
            raise ValueError(
                "first_frame_latent must be present "
                "for WanGame"
            )
        image_latents = raw_batch["first_frame_latent"]
        image_latents = image_latents[
            :, :, : tc.data.num_latent_t
        ]
        image_latents = image_latents.to(
            device, dtype=dtype
        )

        pil_image = raw_batch.get("pil_image")
        if isinstance(pil_image, torch.Tensor):
            training_batch.preprocessed_image = (
                pil_image.to(device=device)
            )
        else:
            training_batch.preprocessed_image = pil_image

        keyboard_cond = raw_batch.get("keyboard_cond")
        if (
            isinstance(keyboard_cond, torch.Tensor)
            and keyboard_cond.numel() > 0
        ):
            training_batch.keyboard_cond = (
                keyboard_cond.to(device, dtype=dtype)
            )
        else:
            training_batch.keyboard_cond = None

        mouse_cond = raw_batch.get("mouse_cond")
        if (
            isinstance(mouse_cond, torch.Tensor)
            and mouse_cond.numel() > 0
        ):
            training_batch.mouse_cond = mouse_cond.to(
                device, dtype=dtype
            )
        else:
            training_batch.mouse_cond = None

        temporal_compression_ratio = (
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        expected_num_frames = (
            (tc.data.num_latent_t - 1)
            * temporal_compression_ratio
            + 1
        )
        if (
            training_batch.keyboard_cond is not None
            and int(training_batch.keyboard_cond.shape[1])
            != int(expected_num_frames)
        ):
            raise ValueError(
                "keyboard_cond temporal dim mismatch: "
                f"got {int(training_batch.keyboard_cond.shape[1])}, "
                f"expected {int(expected_num_frames)}"
            )
        if (
            training_batch.mouse_cond is not None
            and int(training_batch.mouse_cond.shape[1])
            != int(expected_num_frames)
        ):
            raise ValueError(
                "mouse_cond temporal dim mismatch: "
                f"got {int(training_batch.mouse_cond.shape[1])}, "
                f"expected {int(expected_num_frames)}"
            )

        training_batch.latents = latents
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.image_embeds = image_embeds
        training_batch.image_latents = image_latents
        training_batch.infos = infos

        training_batch.latents = normalize_dit_input(
            "wan", training_batch.latents, self.vae
        )
        training_batch = self._prepare_dit_inputs(
            training_batch
        )
        training_batch = self._build_attention_metadata(
            training_batch
        )

        training_batch.attn_metadata_vsa = copy.deepcopy(
            training_batch.attn_metadata
        )
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        training_batch.mask_lat_size = (
            self._build_i2v_mask_latents(image_latents)
        )
        viewmats, intrinsics, action_labels = (
            self._process_actions(training_batch)
        )
        training_batch.viewmats = viewmats
        training_batch.Ks = intrinsics
        training_batch.action = action_labels

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

    def predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        device_type = self.device.type
        dtype = noisy_latents.dtype

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(
                f"Unknown attn_kind: {attn_kind!r}"
            )

        with torch.autocast(
            device_type, dtype=dtype
        ), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            cond_inputs = (
                self._select_cfg_condition_inputs(
                    batch,
                    conditional=conditional,
                    cfg_uncond=cfg_uncond,
                )
            )
            input_kwargs = (
                self._build_distill_input_kwargs(
                    noisy_latents,
                    timestep,
                    image_embeds=cond_inputs[
                        "image_embeds"
                    ],
                    image_latents=cond_inputs[
                        "image_latents"
                    ],
                    mask_lat_size=cond_inputs[
                        "mask_lat_size"
                    ],
                    viewmats=cond_inputs["viewmats"],
                    Ks=cond_inputs["Ks"],
                    action=cond_inputs["action"],
                    mouse_cond=cond_inputs["mouse_cond"],
                    keyboard_cond=cond_inputs[
                        "keyboard_cond"
                    ],
                )
            )
            transformer = self._get_transformer(timestep)
            pred_noise = transformer(
                **input_kwargs
            ).permute(0, 2, 1, 3, 4)
            pred_x0 = pred_noise_to_pred_video(
                pred_noise=pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latents.flatten(
                    0, 1
                ),
                timestep=timestep,
                scheduler=self.noise_scheduler,
            ).unflatten(0, pred_noise.shape[:2])
        return pred_x0

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        device_type = self.device.type
        dtype = noisy_latents.dtype

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(
                f"Unknown attn_kind: {attn_kind!r}"
            )

        with torch.autocast(
            device_type, dtype=dtype
        ), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            cond_inputs = (
                self._select_cfg_condition_inputs(
                    batch,
                    conditional=conditional,
                    cfg_uncond=cfg_uncond,
                )
            )
            input_kwargs = (
                self._build_distill_input_kwargs(
                    noisy_latents,
                    timestep,
                    image_embeds=cond_inputs[
                        "image_embeds"
                    ],
                    image_latents=cond_inputs[
                        "image_latents"
                    ],
                    mask_lat_size=cond_inputs[
                        "mask_lat_size"
                    ],
                    viewmats=cond_inputs["viewmats"],
                    Ks=cond_inputs["Ks"],
                    action=cond_inputs["action"],
                    mouse_cond=cond_inputs["mouse_cond"],
                    keyboard_cond=cond_inputs[
                        "keyboard_cond"
                    ],
                )
            )
            transformer = self._get_transformer(timestep)
            pred_noise = transformer(
                **input_kwargs
            ).permute(0, 2, 1, 3, 4)
        return pred_noise

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        timesteps, attn_metadata = ctx
        with set_forward_context(
            current_timestep=timesteps,
            attn_metadata=attn_metadata,
        ):
            (
                loss / max(1, int(grad_accum_rounds))
            ).backward()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_training_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _init_timestep_mechanics(self) -> None:
        assert self.training_config is not None
        tc = self.training_config
        self.timestep_shift = float(
            tc.pipeline_config.flow_shift  # type: ignore[union-attr]
        )
        self.num_train_timestep = int(
            self.noise_scheduler.num_train_timesteps
        )
        self.min_timestep = 0
        self.max_timestep = self.num_train_timestep

    def _sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.noise_random_generator is None:
            raise RuntimeError(
                "on_train_start() must be called "
                "before prepare_batch()"
            )
        assert self.training_config is not None
        tc = self.training_config

        u = compute_density_for_timestep_sampling(
            weighting_scheme=tc.model.weighting_scheme,
            batch_size=batch_size,
            generator=self.noise_random_generator,
            logit_mean=tc.model.logit_mean,
            logit_std=tc.model.logit_std,
            mode_scale=tc.model.mode_scale,
        )
        indices = (
            u
            * self.noise_scheduler.config.num_train_timesteps
        ).long()
        return self.noise_scheduler.timesteps[indices].to(
            device=device
        )

    def _build_attention_metadata(
        self, training_batch: TrainingBatch
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        latents_shape = training_batch.raw_latent_shape
        patch_size = (
            tc.pipeline_config.dit_config.patch_size  # type: ignore[union-attr]
        )
        current_vsa_sparsity = (
            training_batch.current_vsa_sparsity
        )
        assert latents_shape is not None
        assert training_batch.timesteps is not None

        if (
            envs.FASTVIDEO_ATTENTION_BACKEND
            == "VIDEO_SPARSE_ATTN"
        ):
            if (
                not is_vsa_available()
                or VideoSparseAttentionMetadataBuilder
                is None
            ):
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is "
                    "VIDEO_SPARSE_ATTN, but "
                    "fastvideo_kernel is not correctly "
                    "installed or detected."
                )
            training_batch.attn_metadata = VideoSparseAttentionMetadataBuilder().build(  # type: ignore[misc]
                raw_latent_shape=latents_shape[2:5],
                current_timestep=(
                    training_batch.timesteps
                ),
                patch_size=patch_size,
                VSA_sparsity=current_vsa_sparsity,
                device=self.device,
            )
        elif (
            envs.FASTVIDEO_ATTENTION_BACKEND
            == "VMOBA_ATTN"
        ):
            if (
                not is_vmoba_available()
                or VideoMobaAttentionMetadataBuilder
                is None
            ):
                raise ImportError(
                    "FASTVIDEO_ATTENTION_BACKEND is "
                    "VMOBA_ATTN, but fastvideo_kernel "
                    "(or flash_attn>=2.7.4) is not "
                    "correctly installed."
                )
            moba_params = tc.model.moba_config.copy()
            moba_params.update(
                {
                    "current_timestep": (
                        training_batch.timesteps
                    ),
                    "raw_latent_shape": (
                        training_batch.raw_latent_shape[
                            2:5
                        ]
                    ),
                    "patch_size": patch_size,
                    "device": self.device,
                }
            )
            training_batch.attn_metadata = VideoMobaAttentionMetadataBuilder().build(**moba_params)  # type: ignore[misc]
        else:
            training_batch.attn_metadata = None

        return training_batch

    def _prepare_dit_inputs(
        self, training_batch: TrainingBatch
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        latents = training_batch.latents
        assert isinstance(latents, torch.Tensor)
        batch_size = latents.shape[0]

        if self.noise_gen_cuda is None:
            raise RuntimeError(
                "on_train_start() must be called "
                "before prepare_batch()"
            )

        noise = torch.randn(
            latents.shape,
            generator=self.noise_gen_cuda,
            device=latents.device,
            dtype=latents.dtype,
        )
        timesteps = self._sample_timesteps(
            batch_size, latents.device
        )
        if int(tc.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(timesteps, src=0)

        sigmas = get_sigmas(
            self.noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = (
            (1.0 - sigmas) * latents + sigmas * noise
        )

        training_batch.noisy_model_input = (
            noisy_model_input
        )
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape

        training_batch.latents = (
            training_batch.latents.permute(0, 2, 1, 3, 4)
        )
        return training_batch

    def _build_i2v_mask_latents(
        self, image_latents: torch.Tensor
    ) -> torch.Tensor:
        assert self.training_config is not None
        tc = self.training_config
        temporal_compression_ratio = (
            tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )
        num_frames = (
            (tc.data.num_latent_t - 1)
            * temporal_compression_ratio
            + 1
        )

        (
            batch_size,
            _num_channels,
            _t,
            latent_height,
            latent_width,
        ) = image_latents.shape
        mask_lat_size = torch.ones(
            batch_size,
            1,
            num_frames,
            latent_height,
            latent_width,
        )
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=temporal_compression_ratio,
        )
        mask_lat_size = torch.cat(
            [first_frame_mask, mask_lat_size[:, :, 1:]],
            dim=2,
        )
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        return mask_lat_size.to(
            device=image_latents.device,
            dtype=image_latents.dtype,
        )

    def _process_actions(
        self, training_batch: TrainingBatch
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        keyboard_cond = getattr(
            training_batch, "keyboard_cond", None
        )
        mouse_cond = getattr(
            training_batch, "mouse_cond", None
        )
        if keyboard_cond is None or mouse_cond is None:
            raise ValueError(
                "WanGame batch must provide "
                "keyboard_cond and mouse_cond"
            )

        from fastvideo.models.dits.hyworld.pose import (
            process_custom_actions,
        )

        batch_size = int(
            training_batch.noisy_model_input.shape[0]  # type: ignore[union-attr]
        )
        viewmats_list: list[torch.Tensor] = []
        intrinsics_list: list[torch.Tensor] = []
        action_labels_list: list[torch.Tensor] = []
        for b in range(batch_size):
            v, i, a = process_custom_actions(
                keyboard_cond[b], mouse_cond[b]
            )
            viewmats_list.append(v)
            intrinsics_list.append(i)
            action_labels_list.append(a)

        viewmats = torch.stack(viewmats_list, dim=0).to(
            device=self.device, dtype=torch.bfloat16
        )
        intrinsics = torch.stack(
            intrinsics_list, dim=0
        ).to(device=self.device, dtype=torch.bfloat16)
        action_labels = torch.stack(
            action_labels_list, dim=0
        ).to(device=self.device, dtype=torch.bfloat16)

        num_latent_t = int(
            training_batch.noisy_model_input.shape[2]  # type: ignore[union-attr]
        )
        if int(action_labels.shape[1]) != num_latent_t:
            raise ValueError(
                "Action conditioning temporal dim "
                "mismatch: "
                f"action={tuple(action_labels.shape)} "
                f"vs latent_t={num_latent_t}"
            )
        if int(viewmats.shape[1]) != num_latent_t:
            raise ValueError(
                "Viewmats temporal dim mismatch: "
                f"viewmats={tuple(viewmats.shape)} "
                f"vs latent_t={num_latent_t}"
            )

        return viewmats, intrinsics, action_labels

    def _build_distill_input_kwargs(
        self,
        noisy_video_latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_embeds: torch.Tensor,
        image_latents: torch.Tensor,
        mask_lat_size: torch.Tensor,
        viewmats: torch.Tensor | None,
        Ks: torch.Tensor | None,
        action: torch.Tensor | None,
        mouse_cond: torch.Tensor | None,
        keyboard_cond: torch.Tensor | None,
    ) -> dict[str, Any]:
        hidden_states = torch.cat(
            [
                noisy_video_latents.permute(
                    0, 2, 1, 3, 4
                ),
                mask_lat_size,
                image_latents,
            ],
            dim=1,
        )
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": None,
            "timestep": timestep.to(
                device=self.device, dtype=torch.bfloat16
            ),
            "encoder_hidden_states_image": image_embeds,
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
            "return_dict": False,
        }

    def _select_cfg_condition_inputs(
        self,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, Any]:
        image_embeds = batch.image_embeds
        image_latents = batch.image_latents
        mask_lat_size = batch.mask_lat_size
        if image_embeds is None:
            raise RuntimeError(
                "WanGameModel requires "
                "TrainingBatch.image_embeds"
            )
        if image_latents is None:
            raise RuntimeError(
                "WanGameModel requires "
                "TrainingBatch.image_latents"
            )
        if mask_lat_size is None:
            raise RuntimeError(
                "WanGameModel requires "
                "TrainingBatch.mask_lat_size"
            )

        viewmats = getattr(batch, "viewmats", None)
        Ks = getattr(batch, "Ks", None)
        action = getattr(batch, "action", None)
        mouse_cond = getattr(batch, "mouse_cond", None)
        keyboard_cond = getattr(
            batch, "keyboard_cond", None
        )

        if conditional or cfg_uncond is None:
            return {
                "image_embeds": image_embeds,
                "image_latents": image_latents,
                "mask_lat_size": mask_lat_size,
                "viewmats": viewmats,
                "Ks": Ks,
                "action": action,
                "mouse_cond": mouse_cond,
                "keyboard_cond": keyboard_cond,
            }

        on_missing_raw = cfg_uncond.get(
            "on_missing", "error"
        )
        if not isinstance(on_missing_raw, str):
            raise ValueError(
                "method_config.cfg_uncond.on_missing "
                "must be a string, got "
                f"{type(on_missing_raw).__name__}"
            )
        on_missing = on_missing_raw.strip().lower()
        if on_missing not in {"error", "ignore"}:
            raise ValueError(
                "method_config.cfg_uncond.on_missing "
                "must be one of {error, ignore}, got "
                f"{on_missing_raw!r}"
            )

        supported_channels = {"image", "action"}
        for channel, policy_raw in cfg_uncond.items():
            if channel in {"on_missing"}:
                continue
            if channel in supported_channels:
                continue
            if policy_raw is None:
                continue
            if not isinstance(policy_raw, str):
                raise ValueError(
                    "method_config.cfg_uncond values "
                    "must be strings, got "
                    f"{channel}="
                    f"{type(policy_raw).__name__}"
                )
            policy = policy_raw.strip().lower()
            if policy == "keep":
                continue
            if on_missing == "ignore":
                continue
            raise ValueError(
                "WanGameModel does not support "
                "cfg_uncond channel "
                f"{channel!r} (policy={policy!r}). "
                "Set cfg_uncond.on_missing=ignore or "
                "remove the channel."
            )

        def _get_policy(channel: str) -> str:
            raw = cfg_uncond.get(channel, "keep")
            if raw is None:
                return "keep"
            if not isinstance(raw, str):
                raise ValueError(
                    "method_config.cfg_uncond values "
                    "must be strings, got "
                    f"{channel}={type(raw).__name__}"
                )
            policy = raw.strip().lower()
            if policy not in {"keep", "zero", "drop"}:
                raise ValueError(
                    "method_config.cfg_uncond values "
                    "must be one of "
                    "{keep, zero, drop}, got "
                    f"{channel}={raw!r}"
                )
            return policy

        image_policy = _get_policy("image")
        if image_policy == "zero":
            image_embeds = torch.zeros_like(image_embeds)
            image_latents = torch.zeros_like(
                image_latents
            )
            mask_lat_size = torch.zeros_like(
                mask_lat_size
            )
        elif image_policy == "drop":
            raise ValueError(
                "cfg_uncond.image=drop is not supported "
                "for WanGame I2V; use "
                "cfg_uncond.image=zero or keep."
            )

        action_policy = _get_policy("action")
        if action_policy == "zero":
            if (
                viewmats is None
                or Ks is None
                or action is None
            ):
                if on_missing == "ignore":
                    pass
                else:
                    raise ValueError(
                        "cfg_uncond.action=zero requires "
                        "action conditioning tensors, "
                        "but TrainingBatch is missing "
                        "{viewmats, Ks, action}."
                    )
            else:
                viewmats = torch.zeros_like(viewmats)
                Ks = torch.zeros_like(Ks)
                action = torch.zeros_like(action)
            if mouse_cond is not None:
                mouse_cond = torch.zeros_like(mouse_cond)
            if keyboard_cond is not None:
                keyboard_cond = torch.zeros_like(
                    keyboard_cond
                )
        elif action_policy == "drop":
            viewmats = None
            Ks = None
            action = None
            mouse_cond = None
            keyboard_cond = None

        return {
            "image_embeds": image_embeds,
            "image_latents": image_latents,
            "mask_lat_size": mask_lat_size,
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
        }

    def _get_transformer(
        self, timestep: torch.Tensor
    ) -> torch.nn.Module:
        return self.transformer
