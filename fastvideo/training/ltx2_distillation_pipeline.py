# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS, DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
    DEFAULT_LTX2_AUDIO_HOP_LENGTH, DEFAULT_LTX2_AUDIO_MEL_BINS,
    DEFAULT_LTX2_AUDIO_SAMPLE_RATE)
from fastvideo.pipelines import ForwardBatch
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.ltx2_training_pipeline import LTX2TrainingPipeline
from fastvideo.training.training_utils import EMA_FSDP, get_scheduler
from fastvideo.training.training_utils import shift_timestep
from fastvideo.utils import shallow_asdict

logger = init_logger(__name__)


class LTX2DistillationPipeline(DistillationPipeline):
    """DMD distillation pipeline for LTX-2."""

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    with_audio: bool = False

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ):
        modules = super(DistillationPipeline, self).load_modules(
            fastvideo_args, loaded_modules)
        training_args = cast(TrainingArgs, fastvideo_args)

        if training_args.real_score_model_path:
            logger.info("Loading real score transformer from: %s",
                        training_args.real_score_model_path)
            training_args.override_transformer_cls_name = (
                "LTX2Transformer3DModel")
            self.real_score_transformer = self.load_module_from_path(
                training_args.real_score_model_path, "transformer",
                training_args)
            modules["real_score_transformer"] = self.real_score_transformer
            self.real_score_transformer_2 = None
        else:
            raise ValueError(
                "real_score_model_path is required for DMD distillation pipeline"
            )

        if training_args.fake_score_model_path:
            logger.info("Loading fake score transformer from: %s",
                        training_args.fake_score_model_path)
            training_args.override_transformer_cls_name = (
                "LTX2Transformer3DModel")
            self.fake_score_transformer = self.load_module_from_path(
                training_args.fake_score_model_path, "transformer",
                training_args)
            modules["fake_score_transformer"] = self.fake_score_transformer
            self.fake_score_transformer_2 = None
        else:
            raise ValueError(
                "fake_score_model_path is required for DMD distillation pipeline"
            )

        training_args.override_transformer_cls_name = None
        return modules

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing LTX-2 distillation pipeline...")
        LTX2TrainingPipeline.initialize_training_pipeline(self, training_args)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.training_args.pipeline_config.flow_shift)
        self.modules["scheduler"] = self.noise_scheduler

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        if self.training_args.boundary_ratio is not None:
            self.boundary_timestep = (
                self.training_args.boundary_ratio *
                self.noise_scheduler.num_train_timesteps)
        else:
            self.boundary_timestep = None

        self.real_score_transformer.requires_grad_(False)
        self.real_score_transformer.eval()
        if self.real_score_transformer_2 is not None:
            self.real_score_transformer_2.requires_grad_(False)
            self.real_score_transformer_2.eval()

        if training_args.enable_gradient_checkpointing_type is not None:
            from fastvideo.training.activation_checkpoint import (
                apply_activation_checkpointing)

            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            self.real_score_transformer = apply_activation_checkpointing(
                self.real_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        fake_score_params = list(
            filter(lambda p: p.requires_grad,
                   self.fake_score_transformer.parameters()))
        fake_score_lr = (training_args.fake_score_learning_rate
                         if training_args.fake_score_learning_rate > 0 else
                         training_args.learning_rate)
        betas = tuple(float(x.strip())
                      for x in training_args.fake_score_betas.split(","))

        self.fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=fake_score_lr,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        self.fake_score_lr_scheduler = get_scheduler(
            training_args.fake_score_lr_scheduler,
            optimizer=self.fake_score_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        self.generator_update_interval = training_args.generator_update_interval
        self.denoising_step_list = torch.tensor(
            self.training_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device(),
        )
        if training_args.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.noise_scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )).to(get_local_torch_device())
            self.denoising_step_list = timesteps[1000 -
                                                 self.denoising_step_list]

        self.num_train_timestep = self.noise_scheduler.num_train_timesteps
        self.min_timestep = int(self.training_args.min_timestep_ratio *
                                self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio *
                                self.num_train_timestep)
        self.real_score_guidance_scale = (
            self.training_args.real_score_guidance_scale)

        self.generator_ema: EMA_FSDP | None = None
        self.generator_ema_2: EMA_FSDP | None = None
        if (self.training_args.ema_decay
                is not None) and (self.training_args.ema_decay > 0.0):
            self.generator_ema = EMA_FSDP(self.transformer,
                                          decay=self.training_args.ema_decay)
            logger.info("Initialized generator EMA with decay=%s",
                        self.training_args.ema_decay)
        else:
            logger.info("Generator EMA disabled (ema_decay <= 0.0)")

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline

        logger.info("Initializing LTX-2 validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        validation_pipeline = LTX2Pipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "text_encoder": self.get_module("text_encoder"),
                "tokenizer": self.get_module("tokenizer"),
                "vae": self.get_module("vae"),
                "audio_vae": self.get_module("audio_vae"),
                "vocoder": self.get_module("vocoder"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=training_args.dit_cpu_offload,
        )
        self.validation_pipeline = validation_pipeline

    def _prepare_validation_batch(self, sampling_param, training_args,
                                  validation_batch, num_inference_steps):
        """Prepare validation batch for LTX2 without forcing train-time shapes."""
        if "prompt" in validation_batch:
            sampling_param.prompt = validation_batch["prompt"]
        elif "caption" in validation_batch:
            sampling_param.prompt = validation_batch["caption"]

        # Respect per-sample validation settings when provided; otherwise keep
        # defaults loaded from model sampling config.
        if validation_batch.get("height") is not None:
            sampling_param.height = int(validation_batch["height"])
        if validation_batch.get("width") is not None:
            sampling_param.width = int(validation_batch["width"])
        if validation_batch.get("num_frames") is not None:
            sampling_param.num_frames = int(validation_batch["num_frames"])

        if validation_batch.get("num_inference_steps") is not None:
            sampling_param.num_inference_steps = int(
                validation_batch["num_inference_steps"])
        else:
            sampling_param.num_inference_steps = num_inference_steps

        if validation_batch.get("guidance_scale") is not None:
            sampling_param.guidance_scale = float(
                validation_batch["guidance_scale"])
        elif training_args.validation_guidance_scale:
            sampling_param.guidance_scale = float(
                training_args.validation_guidance_scale)

        if validation_batch.get("negative_prompt") is not None:
            sampling_param.negative_prompt = validation_batch[
                "negative_prompt"]
        if validation_batch.get("fps") is not None:
            sampling_param.fps = int(validation_batch["fps"])

        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        return ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

    def _get_ltx2_data_sources(self, data_path: str) -> dict[str, str]:
        data_root = Path(data_path).expanduser().resolve()
        if (data_root / ".precomputed").exists():
            data_root = data_root / ".precomputed"
        sources = {"latents": "latents", "conditions": "conditions"}
        audio_dir = data_root / "audio_latents"
        if audio_dir.exists() and any(audio_dir.rglob("*.pt")):
            sources["audio_latents"] = "audio_latents"
        return sources

    def _has_precomputed_pt_data(self, data_path: str) -> bool:
        """Check whether precomputed .pt data exists for LTX2."""
        data_root = Path(data_path).expanduser().resolve()
        if (data_root / ".precomputed").exists():
            data_root = data_root / ".precomputed"
        latents_dir = data_root / "latents"
        conditions_dir = data_root / "conditions"
        return (latents_dir.exists() and conditions_dir.exists()
                and any(latents_dir.rglob("*.pt"))
                and any(conditions_dir.rglob("*.pt")))

    def _normalize_dit_input(self, training_batch):
        return training_batch

    def _sample_sigmas(self, batch_size: int, seq_length: int,
                       device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
        min_tokens = 1024
        max_tokens = 4096
        min_shift = 0.95
        max_shift = 2.05
        slope = (max_shift - min_shift) / (max_tokens - min_tokens)
        shift = slope * seq_length + (min_shift - slope * min_tokens)
        normal_samples = torch.randn(
            (batch_size, ),
            generator=self.noise_random_generator,
            device="cpu",
        ) + shift
        return torch.sigmoid(normal_samples).to(device=device, dtype=dtype)

    def _build_data_free_audio_latents(
            self, batch_size: int, device: torch.device,
            dtype: torch.dtype) -> torch.Tensor:
        fps = 24.0
        duration = float(self.training_args.num_frames) / fps
        audio_shape = AudioLatentShape.from_duration(
            batch=batch_size,
            duration=duration,
            channels=DEFAULT_LTX2_AUDIO_CHANNELS,
            mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
            sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
            hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
            audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
        )
        return torch.zeros(audio_shape.to_torch_shape(),
                           device=device,
                           dtype=dtype)

    def _get_next_batch(self, training_batch):
        with self.tracker.timed("timing/get_next_batch"):
            batch = next(self.train_loader_iter, None)  # type: ignore
            if batch is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                self.train_loader_iter = iter(self.train_dataloader)
                batch = next(self.train_loader_iter)

            if self._use_parquet_data:
                training_batch = self._get_next_batch_distill_parquet(
                    batch, training_batch)
            else:
                training_batch = self._get_next_batch_distill_pt(
                    batch, training_batch)

        return training_batch

    def _get_next_batch_distill_parquet(self, batch, training_batch):
        """Extract distillation data from a parquet-format batch."""
        device = get_local_torch_device()
        dtype = torch.bfloat16

        prompt_embeds = batch["text_embedding"].to(device)
        prompt_attention_mask = batch["text_attention_mask"].to(
            device, dtype=torch.int64)
        video_embeds, audio_embeds, attention_mask = (
            self.text_encoder.run_connectors(
                prompt_embeds, prompt_attention_mask))

        batch_size = video_embeds.shape[0]
        vae_config = (self.training_args.pipeline_config
                      .vae_config.arch_config)
        num_channels = vae_config.z_dim
        scr = vae_config.spatial_compression_ratio
        latent_h = self.training_args.num_height // scr
        latent_w = self.training_args.num_width // scr
        latents = torch.zeros(
            batch_size, num_channels,
            self.training_args.num_latent_t,
            latent_h, latent_w,
            device=device, dtype=dtype)

        training_batch.latents = latents
        training_batch.encoder_hidden_states = video_embeds.to(
            device, dtype=dtype)
        training_batch.encoder_attention_mask = attention_mask.to(
            device, dtype=dtype)

        training_batch.audio_latents = (
            self._build_data_free_audio_latents(
                batch_size=batch_size, device=device, dtype=dtype))
        training_batch.audio_encoder_hidden_states = audio_embeds.to(
            device, dtype=dtype)
        training_batch.audio_encoder_attention_mask = (
            attention_mask.to(device))

        training_batch.infos = []
        training_batch.raw_latent_shape = latents.shape
        return training_batch

    def _get_next_batch_distill_pt(self, batch, training_batch):
        """Extract distillation data from a precomputed .pt batch."""
        device = get_local_torch_device()
        dtype = torch.bfloat16

        conditions = batch["conditions"]
        if ("video_prompt_embeds" in conditions
                and "audio_prompt_embeds" in conditions
                and "prompt_attention_mask" in conditions):
            video_embeds = conditions["video_prompt_embeds"].to(
                device)
            audio_embeds = conditions["audio_prompt_embeds"].to(
                device)
            attention_mask = conditions["prompt_attention_mask"].to(
                device, dtype=torch.int64)
        else:
            prompt_embeds = conditions["prompt_embeds"].to(device)
            prompt_attention_mask = conditions[
                "prompt_attention_mask"].to(device, dtype=torch.int64)
            video_embeds, audio_embeds, attention_mask = (
                self.text_encoder.run_connectors(
                    prompt_embeds, prompt_attention_mask))

        if self.training_args.simulate_generator_forward:
            batch_size = video_embeds.shape[0]
            vae_config = (self.training_args.pipeline_config
                          .vae_config.arch_config)
            num_channels = vae_config.z_dim
            scr = vae_config.spatial_compression_ratio
            latent_h = self.training_args.num_height // scr
            latent_w = self.training_args.num_width // scr
            latents = torch.zeros(
                batch_size, num_channels,
                self.training_args.num_latent_t,
                latent_h, latent_w,
                device=device, dtype=dtype)
        else:
            latents = batch["latents"]["latents"].to(
                device, dtype=dtype)
            latents = latents[:, :, :self.training_args.num_latent_t]

        training_batch.latents = latents
        training_batch.encoder_hidden_states = video_embeds.to(
            device, dtype=dtype)
        training_batch.encoder_attention_mask = attention_mask.to(
            device, dtype=dtype)

        if self.training_args.simulate_generator_forward:
            training_batch.audio_latents = (
                self._build_data_free_audio_latents(
                    batch_size=latents.shape[0],
                    device=device, dtype=dtype))
            training_batch.audio_encoder_hidden_states = (
                audio_embeds.to(device, dtype=dtype))
            training_batch.audio_encoder_attention_mask = (
                attention_mask.to(device))
        elif self.with_audio and "audio_latents" in batch:
            audio_latents = batch["audio_latents"]["latents"].to(
                device, dtype=dtype)
            training_batch.audio_latents = audio_latents
            training_batch.audio_encoder_hidden_states = (
                audio_embeds.to(device, dtype=dtype))
            training_batch.audio_encoder_attention_mask = (
                attention_mask.to(device))

        idxs = batch.get("idx")
        if idxs is not None and torch.is_tensor(idxs):
            training_batch.infos = [
                {"idx": int(i)} for i in idxs.tolist()]
        else:
            training_batch.infos = []
        training_batch.raw_latent_shape = latents.shape
        return training_batch

    def _prepare_dit_inputs(self, training_batch):
        training_batch = LTX2TrainingPipeline._prepare_dit_inputs(
            self, training_batch)

        conditional_dict = {
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_attention_mask": training_batch.encoder_attention_mask,
        }
        if training_batch.audio_encoder_hidden_states is not None:
            conditional_dict["audio_encoder_hidden_states"] = (
                training_batch.audio_encoder_hidden_states)
            conditional_dict["audio_encoder_attention_mask"] = (
                training_batch.audio_encoder_attention_mask)

        if getattr(self, "negative_prompt_embeds", None) is not None:
            unconditional_dict = {
                "encoder_hidden_states": self.negative_prompt_embeds,
                "encoder_attention_mask": self.negative_prompt_attention_mask,
            }
            if training_batch.audio_encoder_hidden_states is not None:
                # There is no separate negative audio prompt in training batches;
                # reuse audio conditioning for the uncond branch.
                unconditional_dict["audio_encoder_hidden_states"] = (
                    training_batch.audio_encoder_hidden_states)
                unconditional_dict["audio_encoder_attention_mask"] = (
                    training_batch.audio_encoder_attention_mask)
            training_batch.unconditional_dict = unconditional_dict

        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}
        training_batch.conditional_dict = conditional_dict
        self.video_latent_shape = training_batch.latents.shape
        self.video_latent_shape_sp = training_batch.latents.shape
        return training_batch

    def _sigma_from_timestep(self, timestep: torch.Tensor,
                             n_dim: int) -> torch.Tensor:
        schedule_timesteps = self.noise_scheduler.timesteps.to(
            device=self.device)
        sigmas = self.noise_scheduler.sigmas.to(device=self.device,
                                                dtype=torch.float32)
        timestep = timestep.to(device=self.device,
                               dtype=schedule_timesteps.dtype).flatten()

        step_indices = []
        for t in timestep:
            # Prefer exact match; otherwise use nearest scheduler timestep.
            exact = (schedule_timesteps == t).nonzero(as_tuple=False)
            if exact.numel() > 0:
                idx = int(exact[0].item())
            else:
                idx = int(torch.argmin(torch.abs(schedule_timesteps - t)).item())
            step_indices.append(idx)

        sigma = sigmas[step_indices].flatten()
        return sigma

    def _timestep_tokens_from_sigma(self, hidden_states: torch.Tensor,
                                    sigma: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        if hasattr(self.transformer, "patchifier"):
            from fastvideo.models.dits.ltx2 import VideoLatentShape
            video_shape = VideoLatentShape.from_torch_shape(hidden_states.shape)
            token_count = self.transformer.patchifier.get_token_count(
                video_shape)
        else:
            token_count = hidden_states.shape[2] * hidden_states.shape[
                3] * hidden_states.shape[4]
        return sigma.reshape(batch_size, 1).expand(batch_size, token_count)

    def _build_distill_input_kwargs(
        self,
        noise_input,
        timestep,
        text_dict,
        training_batch,
        audio_noise_input=None,
    ):
        if text_dict is None:
            raise ValueError("text_dict cannot be None for distillation")

        batch_size = noise_input.shape[0]
        sigma = self._sigma_from_timestep(
            timestep.reshape(-1).expand(batch_size).to(self.device), n_dim=5)
        video_timestep = self._timestep_tokens_from_sigma(
            noise_input, sigma.reshape(batch_size)).to(
                noise_input.device, dtype=noise_input.dtype)

        training_batch.input_kwargs = {
            "hidden_states": noise_input,
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": video_timestep,
            "return_dict": False,
        }
        if ("audio_encoder_hidden_states" in text_dict
                and audio_noise_input is not None):
            audio_sigma = self._sigma_from_timestep(
                timestep.reshape(-1).expand(batch_size).to(self.device),
                n_dim=4).reshape(batch_size, 1).expand(
                    batch_size, audio_noise_input.shape[2])
            training_batch.input_kwargs.update({
                "audio_hidden_states":
                audio_noise_input,
                "audio_encoder_hidden_states":
                text_dict["audio_encoder_hidden_states"],
                "audio_timestep":
                audio_sigma.to(noise_input.device, dtype=noise_input.dtype),
                "audio_encoder_attention_mask":
                text_dict["audio_encoder_attention_mask"],
            })

        training_batch.noise_latents = noise_input
        return training_batch

    def _generator_forward(self, training_batch):
        latents = training_batch.latents
        dtype = latents.dtype
        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep = self.denoising_step_list[index]
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep

        noise = torch.randn(self.video_latent_shape, device=self.device, dtype=dtype)
        sigma = self._sigma_from_timestep(
            timestep.expand(latents.shape[0]).to(self.device), n_dim=5).to(dtype)
        noisy_latent = (1.0 - sigma) * latents + sigma * noise

        noisy_audio = None
        if training_batch.audio_latents is not None:
            audio_sigma = self._sigma_from_timestep(
                timestep.expand(latents.shape[0]).to(self.device),
                n_dim=4).to(dtype)
            audio_noise = torch.randn_like(training_batch.audio_latents)
            noisy_audio = ((1.0 - audio_sigma) * training_batch.audio_latents +
                           audio_sigma * audio_noise)

        training_batch = self._build_distill_input_kwargs(
            noisy_latent,
            timestep,
            training_batch.conditional_dict,
            training_batch,
            audio_noise_input=noisy_audio,
        )
        with set_forward_context(current_timestep=timestep,
                                 attn_metadata=training_batch.attn_metadata_vsa):
            pred = self.transformer(**training_batch.input_kwargs)
        if isinstance(pred, tuple):
            return pred[0], pred[1]
        return pred, None

    def _generator_multi_step_simulation_forward(self, training_batch):
        latents = training_batch.latents
        dtype = latents.dtype
        target_idx = torch.randint(0,
                                   len(self.denoising_step_list), [1],
                                   device=self.device,
                                   dtype=torch.long).item()
        target_timestep = self.denoising_step_list[target_idx:target_idx + 1]

        noisy_input = torch.randn(self.video_latent_shape,
                                  device=self.device,
                                  dtype=dtype)
        noisy_audio = None
        if training_batch.audio_latents is not None:
            noisy_audio = torch.randn_like(training_batch.audio_latents)
        with torch.no_grad():
            for step_idx in range(target_idx):
                timestep = self.denoising_step_list[step_idx:step_idx + 1]
                tmp_batch = self._build_distill_input_kwargs(
                    noisy_input,
                    timestep,
                    training_batch.conditional_dict,
                    training_batch,
                    audio_noise_input=noisy_audio,
                )
                with set_forward_context(
                        current_timestep=timestep,
                        attn_metadata=training_batch.attn_metadata_vsa):
                    pred = self.transformer(**tmp_batch.input_kwargs)
                if isinstance(pred, tuple):
                    pred_video, pred_audio = pred
                else:
                    pred_video, pred_audio = pred, None

                next_timestep = self.denoising_step_list[step_idx + 1:step_idx
                                                         + 2]
                next_sigma = self._sigma_from_timestep(
                    next_timestep.expand(latents.shape[0]).to(self.device),
                    n_dim=5).to(dtype)
                noise = torch.randn_like(pred_video)
                noisy_input = (1.0 - next_sigma) * pred_video + next_sigma * noise
                if pred_audio is not None and noisy_audio is not None:
                    next_audio_sigma = self._sigma_from_timestep(
                        next_timestep.expand(latents.shape[0]).to(self.device),
                        n_dim=4).to(dtype)
                    audio_noise = torch.randn_like(pred_audio)
                    noisy_audio = ((1.0 - next_audio_sigma) * pred_audio +
                                   next_audio_sigma * audio_noise)

        training_batch = self._build_distill_input_kwargs(
            noisy_input,
            target_timestep,
            training_batch.conditional_dict,
            training_batch,
            audio_noise_input=noisy_audio,
        )
        with set_forward_context(current_timestep=target_timestep,
                                 attn_metadata=training_batch.attn_metadata_vsa):
            pred = self.transformer(**training_batch.input_kwargs)
        if isinstance(pred, tuple):
            pred_video, pred_audio = pred
        else:
            pred_video, pred_audio = pred, None
        training_batch.dmd_latent_vis_dict[
            "generator_timestep"] = target_timestep.float().detach()
        return pred_video, pred_audio

    def _dmd_forward(self, generator_pred_video, training_batch):
        if isinstance(generator_pred_video, tuple):
            original_latent, original_audio = generator_pred_video
        else:
            original_latent, original_audio = generator_pred_video, None
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)
            timestep = shift_timestep(timestep, self.timestep_shift,
                                      self.num_train_timestep)
            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            sigma = self._sigma_from_timestep(
                timestep.expand(original_latent.shape[0]).to(self.device),
                n_dim=5).to(original_latent.dtype)
            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=original_latent.dtype)
            noisy_latent = ((1.0 - sigma) * original_latent +
                            sigma * noise).detach()
            noisy_audio = None
            audio_noise = None
            if original_audio is not None:
                audio_sigma = self._sigma_from_timestep(
                    timestep.expand(original_latent.shape[0]).to(self.device),
                    n_dim=4).to(original_audio.dtype)
                audio_noise = torch.randn_like(original_audio)
                noisy_audio = ((1.0 - audio_sigma) * original_audio +
                               audio_sigma * audio_noise).detach()

            training_batch = self._build_distill_input_kwargs(
                noisy_latent,
                timestep,
                training_batch.conditional_dict,
                training_batch,
                audio_noise_input=noisy_audio,
            )
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            fake_pred = current_fake_score_transformer(
                **training_batch.input_kwargs)
            if isinstance(fake_pred, tuple):
                fake_pred_video, fake_pred_audio = fake_pred
            else:
                fake_pred_video, fake_pred_audio = fake_pred, None

            training_batch = self._build_distill_input_kwargs(
                noisy_latent,
                timestep,
                training_batch.conditional_dict,
                training_batch,
                audio_noise_input=noisy_audio,
            )
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            real_cond = current_real_score_transformer(
                **training_batch.input_kwargs)
            if isinstance(real_cond, tuple):
                real_cond_video, real_cond_audio = real_cond
            else:
                real_cond_video, real_cond_audio = real_cond, None

            if training_batch.unconditional_dict is not None:
                training_batch = self._build_distill_input_kwargs(
                    noisy_latent,
                    timestep,
                    training_batch.unconditional_dict,
                    training_batch,
                    audio_noise_input=noisy_audio,
                )
                real_uncond = current_real_score_transformer(
                    **training_batch.input_kwargs)
                if isinstance(real_uncond, tuple):
                    real_uncond_video, real_uncond_audio = real_uncond
                else:
                    real_uncond_video, real_uncond_audio = real_uncond, None
                real_score_pred_video = real_cond_video + (
                    real_cond_video - real_uncond_video
                ) * self.real_score_guidance_scale
                if real_cond_audio is not None and real_uncond_audio is not None:
                    real_score_pred_audio = real_cond_audio + (
                        real_cond_audio - real_uncond_audio
                    ) * self.real_score_guidance_scale
                else:
                    real_score_pred_audio = real_cond_audio
            else:
                real_score_pred_video = real_cond_video
                real_score_pred_audio = real_cond_audio

            denom = torch.abs(original_latent -
                              real_score_pred_video).mean().clamp_min(1e-6)
            grad = (fake_pred_video - real_score_pred_video) / denom
            grad = torch.nan_to_num(grad)
            grad_audio = None
            if (original_audio is not None and fake_pred_audio is not None
                    and real_score_pred_audio is not None):
                audio_denom = torch.abs(original_audio -
                                        real_score_pred_audio).mean().clamp_min(
                                            1e-6)
                grad_audio = (fake_pred_audio - real_score_pred_audio
                              ) / audio_denom
                grad_audio = torch.nan_to_num(grad_audio)

        dmd_loss = 0.5 * torch.nn.functional.mse_loss(
            original_latent.float(),
            (original_latent.float() - grad.float()).detach())
        if grad_audio is not None and original_audio is not None:
            audio_dmd_loss = 0.5 * torch.nn.functional.mse_loss(
                original_audio.float(),
                (original_audio.float() - grad_audio.float()).detach(),
            )
            dmd_loss = dmd_loss + audio_dmd_loss

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video":
            original_latent.detach(),
            "real_score_pred_video":
            real_score_pred_video.detach(),
            "faker_score_pred_video":
            fake_pred_video.detach(),
            "dmd_timestep":
            timestep.detach(),
        })
        return dmd_loss

    def faker_score_forward(self, training_batch):
        with torch.no_grad():
            if self.training_args.simulate_generator_forward:
                generator_pred_video, generator_pred_audio = self._generator_multi_step_simulation_forward(
                    training_batch)
            else:
                generator_pred_video, generator_pred_audio = self._generator_forward(
                    training_batch)

        fake_score_timestep = torch.randint(0,
                                            self.num_train_timestep, [1],
                                            device=self.device,
                                            dtype=torch.long)
        fake_score_timestep = shift_timestep(fake_score_timestep,
                                             self.timestep_shift,
                                             self.num_train_timestep)
        fake_score_timestep = fake_score_timestep.clamp(
            self.min_timestep, self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)
        sigma = self._sigma_from_timestep(
            fake_score_timestep.expand(generator_pred_video.shape[0]).to(
                self.device),
            n_dim=5).to(generator_pred_video.dtype)
        noisy_generator_pred_video = ((1.0 - sigma) * generator_pred_video +
                                      sigma * fake_score_noise)
        noisy_generator_pred_audio = None
        fake_score_audio_noise = None
        if generator_pred_audio is not None:
            audio_sigma = self._sigma_from_timestep(
                fake_score_timestep.expand(generator_pred_video.shape[0]).to(
                    self.device),
                n_dim=4).to(generator_pred_audio.dtype)
            fake_score_audio_noise = torch.randn_like(generator_pred_audio)
            noisy_generator_pred_audio = ((1.0 - audio_sigma) *
                                          generator_pred_audio +
                                          audio_sigma * fake_score_audio_noise)

        training_batch = self._build_distill_input_kwargs(
            noisy_generator_pred_video,
            fake_score_timestep,
            training_batch.conditional_dict,
            training_batch,
            audio_noise_input=noisy_generator_pred_audio,
        )
        current_fake_score_transformer = self._get_fake_score_transformer(
            fake_score_timestep)
        with set_forward_context(current_timestep=fake_score_timestep,
                                 attn_metadata=training_batch.attn_metadata):
            pred = current_fake_score_transformer(**training_batch.input_kwargs)
        if isinstance(pred, tuple):
            fake_score_denoised, fake_score_audio_denoised = pred
        else:
            fake_score_denoised, fake_score_audio_denoised = pred, None

        pred_velocity = (noisy_generator_pred_video - fake_score_denoised) / sigma
        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((pred_velocity.float() - target.float())**2)
        if (fake_score_audio_denoised is not None
                and noisy_generator_pred_audio is not None
                and generator_pred_audio is not None
                and fake_score_audio_noise is not None):
            audio_sigma = self._sigma_from_timestep(
                fake_score_timestep.expand(generator_pred_video.shape[0]).to(
                    self.device),
                n_dim=4).to(fake_score_audio_denoised.dtype)
            pred_audio_velocity = (
                noisy_generator_pred_audio - fake_score_audio_denoised) / audio_sigma
            audio_target = fake_score_audio_noise - generator_pred_audio
            flow_matching_loss = flow_matching_loss + torch.mean(
                (pred_audio_velocity.float() - audio_target.float())**2)

        training_batch.fake_score_latent_vis_dict = {
            "training_batch_fakerscore_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video":
            generator_pred_video,
            "fake_score_timestep":
            fake_score_timestep,
        }
        return training_batch, flow_matching_loss


def main(args) -> None:
    logger.info("Starting LTX-2 distillation pipeline...")
    pipeline = LTX2DistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("LTX-2 distillation pipeline completed")


if __name__ == "__main__":
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
