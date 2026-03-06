

## 1) File Structure

fastvideo/train/
  trainer.py                # Training loop; calls method.train_one_step()
  models/
      base.py              # BaseModel ABC: predict_x0, add_noise, backward, ...
      wan/
          wan.py           # Wan model loader
      wangame/
          wangame.py       # WanGame model loader
          wangame_causal.py
  methods/
      base.py              # DistillMethod base; methods provide train_one_step
      distribution_matching/
          dmd2.py          # DMD2 distillation (student/teacher/critic)
          self_forcing.py  # Self-forcing distillation
      fine_tuning/
          finetune.py      # SFT finetuning (student only)
          dfsft.py         # Distribution-free SFT
      knowledge_distillation/
      consistency_model/
  callbacks/
      callback.py          # CallbackDict registry
      grad_clip.py         # Gradient clipping + optional per-module norm logging
      validation.py        # Periodic validation via inference pipeline
      ema.py               # EMA weight averaging
  entrypoint/
      train.py             # YAML-only CLI entrypoint (torchrun -m fastvideo.train.entrypoint.train)
      dcp_to_diffusers.py  # Checkpoint conversion
  utils/
      config.py            # YAML parser -> RunConfig
      builder.py           # build_from_config: instantiate models, method, dataloader
      instantiate.py       # _target_ based instantiation
      training_config.py   # TrainingConfig dataclass (all training settings with defaults)
      dataloader.py        # Dataset / dataloader construction
      moduleloader.py      # Dynamic module import
      module_state.py      # apply_trainable(): requires_grad + train/eval
      optimizer.py         # Optimizer construction
      tracking.py          # W&B tracker (owned by trainer)
      checkpoint.py        # Save/resume with DCP
      validation.py        # Validation helpers

By this design, we only need a YAML config to train different models using different methods.
Models declare `_target_` to select the model class; methods declare `_target_` to select the method class.
Current code: https://github.com/FoundationResearch/FastVideo/tree/distill1/fastvideo/train

DMD2 Distillation, Self-Forcing, SFT, and DFSFT are tested on Wan / WanGame.

Current supported models: Wan, WanGame.
Current supported methods: DMD2, Self-Forcing, SFT, DFSFT.

Feedbacks are highly welcome!


## 2) Example YAML (DMD2 8-step)

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: false
    disable_custom_init_weights: true
  critic:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
    disable_custom_init_weights: true

method:
  _target_: fastvideo.train.methods.distribution_matching.dmd2.DMD2Method
  rollout_mode: simulate
  generator_update_interval: 5
  real_score_guidance_scale: 3.5
  dmd_denoising_steps: [1000, 850, 700, 550, 350, 275, 200, 125]

  # Critic optimizer (required)
  fake_score_learning_rate: 8.0e-6
  fake_score_betas: [0.0, 0.999]
  fake_score_lr_scheduler: constant

training:
  distributed:
    num_gpus: 8
    sp_size: 1
    tp_size: 1

  data:
    data_path: data/Wan-Syn_77x448x832_600k
    dataloader_num_workers: 4
    train_batch_size: 1
    training_cfg_rate: 0.0
    seed: 1000
    num_latent_t: 20
    num_height: 448
    num_width: 832
    num_frames: 77

  optimizer:
    learning_rate: 2.0e-6
    betas: [0.0, 0.999]
    weight_decay: 0.01
    lr_scheduler: constant
    lr_warmup_steps: 0

  loop:
    max_train_steps: 4000
    gradient_accumulation_steps: 1

  checkpoint:
    output_dir: outputs/wan2.1_dmd2_8steps
    training_state_checkpointing_steps: 1000
    checkpoints_total_limit: 3

  tracker:
    project_name: distillation_wan
    run_name: wan2.1_dmd2_8steps

  model:
    enable_gradient_checkpointing_type: full

callbacks:
  grad_clip:
    max_grad_norm: 1.0
  validation:
    pipeline_target: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
    dataset_file: examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json
    every_steps: 100
    sampling_steps: [8]
    sampler_kind: sde
    sampling_timesteps: [1000, 850, 700, 550, 350, 275, 200, 125]
    guidance_scale: 6.0

pipeline:
  flow_shift: 8
```
