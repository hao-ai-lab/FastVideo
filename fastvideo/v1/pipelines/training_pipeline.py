import os
from typing import Any, Dict, List, Optional

from fastvideo.v1.distributed import (get_sp_group,
                                      init_distributed_environment,
                                      initialize_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           PipelineStage, TextEncodingStage,
                                           TimestepPreparationStage)

logger = init_logger(__name__)


class WanTrainingPipeline(ComposedPipelineBase):  # == distill_one_step

    def __init__(self,
                 model_path: str,
                 fastvideo_args: FastVideoArgs,
                 config: Optional[dict[str, Any]] = None):
        # Initialize distributed environment
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        rank = int(os.environ.get("RANK", -1))

        if local_rank == -1 or world_size == -1 or rank == -1:
            raise ValueError(
                "Local rank, world size, and rank must be set. Use torchrun to launch the script."
            )
        print('test')

        init_distributed_environment(local_rank, world_size, rank)
        initialize_model_parallel(tensor_model_parallel_size=world_size,
                                  sequence_model_parallel_size=world_size)

        self.model_path = model_path
        self._stages: List[PipelineStage] = []
        self._stage_name_mapping: Dict[str, PipelineStage] = {}

        if self._required_config_modules is None:
            raise NotImplementedError(
                "Subclass must set _required_config_modules")

        if config is None:
            # Load configuration
            logger.info("Loading pipeline configuration...")
            self.config = self._load_config(model_path)
        else:
            self.config = config

        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(fastvideo_args)

        self.initialize_pipeline(fastvideo_args)

        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(fastvideo_args)

    def get_teacher_latent():
        pass

    def create_validation_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def create_training_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ):
        sp_group = get_sp_group()
        world_size = sp_group.world_size
        rank = sp_group.rank
        args = fastvideo_args

        # Train!
        total_batch_size = (world_size * args.gradient_accumulation_steps /
                            args.sp_size * args.train_sp_batch_size)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(train_dataset))
        logger.info("  Dataloader size = %s", len(train_dataloader))
        logger.info("  Num Epochs = %s", args.num_train_epochs)
        logger.info("  Resume training from step %s", init_steps)
        logger.info("  Instantaneous batch size per device = %s",
                    args.train_batch_size)
        logger.info(
            "  Total train batch size (w. data & sequence parallel, accumulation) = %s",
            total_batch_size)
        logger.info("  Gradient Accumulation steps = %s",
                    args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s", args.max_train_steps)
        logger.info(
            "  Total training parameters per FSDP shard = %s B",
            sum(p.numel()
                for p in transformer.parameters() if p.requires_grad) / 1e9)
        # print dtype
        logger.info("  Master weight dtype: %s",
                    transformer.parameters().__next__().dtype)

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            assert NotImplementedError(
                "resume_from_checkpoint is not supported now.")
            # TODO

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=init_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=local_rank > 0,
        )

        loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )

        step_times = deque(maxlen=100)

        # todo future
        for i in range(init_steps):
            next(loader)

        # log_validation(args, transformer, device,
        #             torch.bfloat16, 0, scheduler_type=args.scheduler_type, shift=args.shift, num_euler_timesteps=args.num_euler_timesteps, linear_quadratic_threshold=args.linear_quadratic_threshold,ema=False)
        def get_num_phases(multi_phased_distill_schedule, step):
            # step-phase,step-phase
            multi_phases = multi_phased_distill_schedule.split(",")
            phase = multi_phases[-1].split("-")[-1]
            for step_phases in multi_phases:
                phase_step, phase = step_phases.split("-")
                if step <= int(phase_step):
                    return int(phase)
            return phase

        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.time()
            assert args.multi_phased_distill_schedule is not None
            num_phases = get_num_phases(args.multi_phased_distill_schedule,
                                        step)

            loss, grad_norm, pred_norm = distill_one_step(
                transformer,
                args.model_type,
                teacher_transformer,
                ema_transformer,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                solver,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.max_grad_norm,
                uncond_prompt_embed,
                uncond_prompt_mask,
                args.num_euler_timesteps,
                num_phases,
                args.not_apply_cfg_solver,
                args.distill_cfg,
                args.ema_decay,
                args.pred_decay_weight,
                args.pred_decay_type,
                args.hunyuan_teacher_disable_cfg,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
                "phases": num_phases,
            })
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss":
                        loss,
                        "learning_rate":
                        lr_scheduler.get_last_lr()[0],
                        "step_time":
                        step_time,
                        "avg_step_time":
                        avg_step_time,
                        "grad_norm":
                        grad_norm,
                        "pred_fro_norm":
                        pred_norm["fro"],  # codespell:ignore
                        "pred_largest_singular_value":
                        pred_norm["largest singular value"],
                        "pred_absolute_mean":
                        pred_norm["absolute mean"],
                        "pred_absolute_max":
                        pred_norm["absolute max"],
                    },
                    step=step,
                )
            if step % args.checkpointing_steps == 0:
                if args.use_lora:
                    # Save LoRA weights
                    save_lora_checkpoint(transformer, optimizer, rank,
                                         args.output_dir, step)
                else:
                    # Your existing checkpoint saving code
                    if args.use_ema:
                        save_checkpoint(ema_transformer, rank, args.output_dir,
                                        step)
                    else:
                        save_checkpoint(transformer, rank, args.output_dir,
                                        step)
                dist.barrier()
            if args.log_validation and step % args.validation_steps == 0:
                log_validation(
                    args,
                    transformer,
                    device,
                    torch.bfloat16,
                    step,
                    scheduler_type=args.scheduler_type,
                    shift=args.shift,
                    num_euler_timesteps=args.num_euler_timesteps,
                    linear_quadratic_threshold=args.linear_quadratic_threshold,
                    linear_range=args.linear_range,
                    ema=False,
                )
                if args.use_ema:
                    log_validation(
                        args,
                        ema_transformer,
                        device,
                        torch.bfloat16,
                        step,
                        scheduler_type=args.scheduler_type,
                        shift=args.shift,
                        num_euler_timesteps=args.num_euler_timesteps,
                        linear_quadratic_threshold=args.
                        linear_quadratic_threshold,
                        linear_range=args.linear_range,
                        ema=True,
                    )

        if args.use_lora:
            save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                                 args.max_train_steps)
        else:
            save_checkpoint(transformer, rank, args.output_dir,
                            args.max_train_steps)

        if get_sequence_parallel_state():
            destroy_sequence_parallel_group()
