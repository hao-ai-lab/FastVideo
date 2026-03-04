## 1) File Structure

```text
fastvideo/distillation/
  trainer.py # Builds the training loop; calls method-provided train_one_step
  dispatch.py # Auto-dispatch via @register_method/@register_model; builds DistillRuntime from config
  roles.py # Wraps model resources in RoleHandle to tag roles (teacher/student/critic/...)
  models/
      components.py # intermediate variable during dispatch-time model construction; records all components
      wan.py # loads Wan. model-loading logic differs across families
      ...
  adapters/
      base.py # turns loaded components into runtime primitives: predict_x0/add_noise/backward/...
      wan.py # wan-specific adapter.
      ...
  methods/
      base.py # DistillMethod base; methods must provide e.g. train-one-step for trainer to call
      distribution_matching/
          dmd2.py # DMD2 distillation (student/teacher/critic)
      fine_tuning/
          finetune.py # SFT finetuning (student only)
      knowledge_distillation/
      consistency_model/
  validators/
      base.py # inference differs by model; validators are model-specific.
      wan.py  # validator backed by WanPipeline.
    utils/
      config.py # yaml parser
      dataloader.py
      moduleloader.py
      module_state.py # apply_trainable(...): standardizes requires_grad + train/eval
      tracking.py # wandb tracker (owned by trainer)
      checkpoint.py # save/resume

```

``` yaml
recipe:
  family: wan
  method: dmd2

roles:
  student:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    trainable: false
  critic:
    path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

training:
  seed: 0
  output_dir: /path/to/out
  data_path: /path/to/parquet
  max_train_steps: 4000
  train_batch_size: 1
  dataloader_num_workers: 4
  num_gpus: 8

  log_validation: true
  validation_steps: 50
  validation_dataset_file: /path/to/validation.json
  validation_sampling_steps: "8"     # legacy-style string；method 可选择覆写
  validation_guidance_scale: "5.0"   # legacy-style string；method 可选择覆写

  trackers: ["wandb"]
  tracker_project_name: my_project
  wandb_run_name: my_run

pipeline_config:
  flow_shift: 3
  sampler_kind: ode

method_config:
  rollout_mode: simulate
  dmd_denoising_steps: [999, 750, 500, 250, 0]
  generator_update_interval: 1
  real_score_guidance_scale: 1.0
  attn_kind: dense
```