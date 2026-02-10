# SPDX-License-Identifier: Apache-2.0
import os
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.distributed as dist

from fastvideo.dataset import build_ltx2_precomputed_dataloader
from fastvideo.distributed import get_local_torch_device, get_sp_group, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import VideoLatentShape
from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.trackers import (DummyTracker, TrackerType, Trackers,
                                         initialize_trackers)
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases, get_scheduler)
from fastvideo.utils import set_random_seed

logger = init_logger(__name__)


class LTX2TrainingPipeline(TrainingPipeline):
    """Training pipeline for LTX-2 text-to-video (optional audio)."""

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    text_encoder: torch.nn.Module
    with_audio: bool = False
    tracker: TrackerType

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # TODO (David): Change to port LTX2 scheduler into self.modules["scheduler"]
        if "scheduler" in self.modules:
            del self.modules["scheduler"]

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing LTX-2 training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.transformer_2 = self.get_module("transformer_2", None)
        self.text_encoder = self.get_module("text_encoder")
        self.text_encoder.eval()
        self.text_encoder.to(self.device)
        self.seed = training_args.seed

        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            from fastvideo.training.activation_checkpoint import (
                apply_activation_checkpointing)
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        self.set_trainable()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters()))
        betas_str = training_args.betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_args.learning_rate,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        data_sources = self._get_ltx2_data_sources(training_args.data_path)
        self.with_audio = "audio_latents" in data_sources
        self.train_dataset, self.train_dataloader = (
            build_ltx2_precomputed_dataloader(
                training_args.data_path,
                training_args.train_batch_size,
                num_data_workers=training_args.dataloader_num_workers,
                data_sources=data_sources,
                drop_last=True,
                seed=self.seed,
            ))

        self.num_update_steps_per_epoch = max(
            1,
            len(self.train_dataloader) //
            training_args.gradient_accumulation_steps)
        self.num_train_epochs = max(
            1, training_args.max_train_steps // self.num_update_steps_per_epoch)
        self.current_epoch = 0

        trackers = list(training_args.trackers)
        if not trackers and training_args.tracker_project_name:
            trackers.append(Trackers.WANDB.value)
        if self.global_rank != 0:
            trackers = []

        tracker_log_dir = training_args.output_dir or os.getcwd()
        if trackers:
            tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

        tracker_config = training_args.__dict__ if trackers else None
        tracker_run_name = training_args.wandb_run_name or None
        project = training_args.tracker_project_name or "fastvideo"
        self.tracker = initialize_trackers(
            trackers,
            experiment_name=project,
            config=tracker_config,
            log_dir=tracker_log_dir,
            run_name=tracker_run_name,
        ) if trackers else DummyTracker()

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
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

    def _get_ltx2_data_sources(self, data_path: str) -> dict[str, str]:
        data_root = Path(data_path).expanduser().resolve()
        if (data_root / ".precomputed").exists():
            data_root = data_root / ".precomputed"
        sources = {"latents": "latents", "conditions": "conditions"}
        audio_dir = data_root / "audio_latents"
        if audio_dir.exists() and any(audio_dir.rglob("*.pt")):
            sources["audio_latents"] = "audio_latents"
        return sources

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        with self.tracker.timed("timing/get_next_batch"):
            batch = next(self.train_loader_iter, None)  # type: ignore
            if batch is None:
                self.current_epoch += 1
                logger.info("Starting epoch %s", self.current_epoch)
                self.train_loader_iter = iter(self.train_dataloader)
                batch = next(self.train_loader_iter)

            latents = batch["latents"]["latents"].to(get_local_torch_device(),
                                                     dtype=torch.bfloat16)
            latents = latents[:, :, :self.training_args.num_latent_t]
            conditions = batch["conditions"]
            if ("video_prompt_embeds" in conditions
                    and "audio_prompt_embeds" in conditions
                    and "prompt_attention_mask" in conditions):
                video_embeds = conditions["video_prompt_embeds"].to(
                    get_local_torch_device())
                audio_embeds = conditions["audio_prompt_embeds"].to(
                    get_local_torch_device())
                attention_mask = conditions["prompt_attention_mask"].to(
                    get_local_torch_device(), dtype=torch.int64)
            else:
                prompt_embeds = conditions["prompt_embeds"].to(
                    get_local_torch_device())
                prompt_attention_mask = conditions["prompt_attention_mask"].to(
                    get_local_torch_device(), dtype=torch.int64)

                video_embeds, audio_embeds, attention_mask = (
                    self.text_encoder.run_connectors(prompt_embeds,
                                                     prompt_attention_mask))

            training_batch.latents = latents.to(get_local_torch_device(),
                                                dtype=torch.bfloat16)
            training_batch.encoder_hidden_states = video_embeds.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.encoder_attention_mask = attention_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)

            if self.with_audio and "audio_latents" in batch:
                audio_latents = batch["audio_latents"]["latents"].to(
                    get_local_torch_device(), dtype=torch.bfloat16)
                training_batch.audio_latents = audio_latents
                training_batch.audio_encoder_hidden_states = audio_embeds.to(
                    get_local_torch_device(), dtype=torch.bfloat16)
                training_batch.audio_encoder_attention_mask = attention_mask.to(
                    get_local_torch_device())

            idxs = batch.get("idx")
            if idxs is not None and torch.is_tensor(idxs):
                training_batch.infos = [{"idx": int(i)} for i in idxs.tolist()]
            else:
                training_batch.infos = []
            training_batch.raw_latent_shape = latents.shape

        return training_batch

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
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

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert training_batch.latents is not None
        latents = training_batch.latents

        batch_size = latents.shape[0]
        video_shape = VideoLatentShape.from_torch_shape(latents.shape)
        if hasattr(self.transformer, "patchifier"):
            token_count = self.transformer.patchifier.get_token_count(
                video_shape)
        else:
            token_count = latents.shape[2] * latents.shape[3] * latents.shape[4]

        sigmas = self._sample_sigmas(batch_size, token_count, latents.device,
                                     latents.dtype)
        noise = torch.randn(latents.shape,
                            generator=self.noise_gen_cuda,
                            device=latents.device,
                            dtype=latents.dtype)
        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = (
            1.0 - sigmas_expanded) * latents + sigmas_expanded * noise

        conditioning_mask = None
        first_frame_p = self.training_args.ltx2_first_frame_conditioning_p
        if (first_frame_p > 0 and torch.rand(
                1,
                generator=self.noise_random_generator,
        ).item() < first_frame_p):
            conditioning_mask = torch.zeros(
                (batch_size, 1, latents.shape[2], latents.shape[3],
                 latents.shape[4]),
                dtype=torch.bool,
                device=latents.device,
            )
            conditioning_mask[:, :, 0:1] = True
            noisy_model_input = torch.where(conditioning_mask, latents,
                                            noisy_model_input)

        if conditioning_mask is None:
            mask_patch = torch.zeros(
                (batch_size, token_count),
                dtype=torch.bool,
                device=latents.device,
            )
        elif hasattr(self.transformer, "patchifier"):
            mask_patch = self.transformer.patchifier.patchify(
                conditioning_mask.float()).sum(dim=-1) > 0
        else:
            mask_patch = conditioning_mask[:, 0].reshape(batch_size, -1)

        timesteps = torch.where(
            mask_patch,
            torch.zeros_like(mask_patch, dtype=latents.dtype),
            sigmas.view(-1, 1).expand_as(mask_patch).to(latents.dtype),
        )

        training_batch.noisy_model_input = noisy_model_input
        training_batch.timesteps = timesteps
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.conditioning_mask = conditioning_mask
        if hasattr(
                training_batch,
                "audio_latents") and training_batch.audio_latents is not None:
            audio_latents = training_batch.audio_latents
            audio_noise = torch.randn(audio_latents.shape,
                                      generator=self.noise_gen_cuda,
                                      device=audio_latents.device,
                                      dtype=audio_latents.dtype)
            audio_sigmas_expanded = sigmas.view(-1, 1, 1, 1)
            audio_noisy = (1.0 - audio_sigmas_expanded) * audio_latents + (
                audio_sigmas_expanded * audio_noise)
            audio_timesteps = sigmas.view(-1, 1).expand(batch_size,
                                                        audio_latents.shape[2])
            training_batch.audio_noisy_model_input = audio_noisy
            training_batch.audio_timesteps = audio_timesteps.to(
                dtype=audio_latents.dtype)
            training_batch.audio_noise = audio_noise

        return training_batch

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.attn_metadata = None
        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.timesteps is not None
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "return_dict":
            False,
        }
        if training_batch.audio_noisy_model_input is not None:
            training_batch.input_kwargs.update({
                "audio_hidden_states":
                training_batch.audio_noisy_model_input,
                "audio_encoder_hidden_states":
                training_batch.audio_encoder_hidden_states,
                "audio_timestep":
                training_batch.audio_timesteps.to(get_local_torch_device(),
                                                  dtype=torch.bfloat16),
                "audio_encoder_attention_mask":
                training_batch.audio_encoder_attention_mask,
            })
        return training_batch

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor,
                    conditioning_mask: torch.Tensor | None) -> torch.Tensor:
        loss = (pred.float() - target.float())**2
        if conditioning_mask is None:
            return loss.mean()
        loss_mask = (~conditioning_mask).float()
        loss_mask = loss_mask.expand(-1, pred.shape[1], -1, -1, -1)
        denom = loss_mask.mean()
        if denom.item() == 0.0:
            return loss.sum() * 0.0
        loss = loss * loss_mask
        return loss.mean() / denom

    def _transformer_forward_and_compute_loss(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        input_kwargs = training_batch.input_kwargs
        assert input_kwargs is not None
        assert training_batch.sigmas is not None
        assert training_batch.latents is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.noise is not None

        with self.tracker.timed("timing/forward_backward"), set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            with torch.autocast("cuda", dtype=training_batch.latents.dtype
                                ), torch.autograd.set_detect_anomaly(True):
                outputs = self.transformer(**input_kwargs)
            if isinstance(outputs, tuple):
                video_denoised, audio_denoised = outputs
            else:
                video_denoised = outputs
                audio_denoised = None

            sigmas_expanded = training_batch.sigmas.view(-1, 1, 1, 1, 1)
            video_pred_velocity = (training_batch.noisy_model_input -
                                   video_denoised) / sigmas_expanded
            video_target = training_batch.noise - training_batch.latents
            loss = self._masked_mse(video_pred_velocity, video_target,
                                    training_batch.conditioning_mask)

            if audio_denoised is not None and training_batch.audio_latents is not None:
                audio_sigmas = training_batch.sigmas.view(-1, 1, 1, 1)
                audio_pred_velocity = (training_batch.audio_noisy_model_input -
                                       audio_denoised) / audio_sigmas
                audio_target = training_batch.audio_noise - training_batch.audio_latents
                audio_loss = (audio_pred_velocity.float() -
                              audio_target.float())**2
                loss = loss + audio_loss.mean()
                logger.info("Audio loss: %s", audio_loss.mean())
            else:
                logger.warning("Audio denoised is None")

            loss = loss / self.training_args.gradient_accumulation_steps
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            avg_loss = loss.detach().clone()
            logger.info("finished backward")

        with self.tracker.timed("timing/reduce_loss"):
            world_group = get_world_group()
            avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        training_batch.total_loss += avg_loss.item()
        return training_batch

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        max_grad_norm = self.training_args.max_grad_norm
        if max_grad_norm is not None:
            with self.tracker.timed("timing/clip_grad_norm"):
                grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                    [p for p in self.transformer.parameters()],
                    max_grad_norm,
                    foreach=None,
                )
                grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch


def main(args) -> None:
    logger.info("Starting LTX-2 training pipeline...")
    pipeline = LTX2TrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
