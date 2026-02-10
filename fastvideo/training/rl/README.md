# RL (GRPO) Training Pipeline

GRPO training for Wan2.1 1.3B: we generate **rollouts** with the inference pipeline (inference mode), then train the policy on those rollouts (reward + advantage + GRPO loss). Currently we use the **OCR reward** and the **text-prompt dataset** only.

## How to run

**Single-GPU:**

```bash
# From repo root
bash examples/training/rl/finetune_t2v_grpo.sh
```

**Multi-GPU (e.g. 4 GPUs):**

```bash
bash examples/training/rl/finetune_t2v_grpo_4gpu.sh
```

Scripts call `fastvideo/training/wan_rl_training_pipeline.py` (Wan RL entry point).

## Code structure

```
FastVideo/
├── examples/training/rl/           # Runscripts
│   ├── finetune_t2v_grpo.sh        # Single-GPU
│   ├── finetune_t2v_grpo_4gpu.sh   # Multi-GPU (4)
│   └── validation.json
│
├── data/ocr/                       # Text prompt dataset (train.txt, test.txt)
│
├── fastvideo/
│   ├── training/
│   │   ├── wan_rl_training_pipeline.py   # Wan RL entry → RLPipeline
│   │   └── rl/                            # RL core
│   │       ├── rl_pipeline.py            # RLPipeline: collect, reward, advantage, GRPO loss
│   │       ├── rl_utils.py
│   │       ├── stat_tracking.py          # Per-prompt advantage normalization
│   │       ├── wan_grpo_utils.py
│   │       └── rewards/                  # Reward models
│   │           ├── rewards.py            # MultiRewardAggregator, create_reward_models
│   │           ├── ocr.py                # OCR reward (current)
│   │           └── base.py
│   │
│   ├── dataset/
│   │   └── rl_prompt_dataset.py    # RL prompt dataloader (text / geneval), KRepeatSampler
│   │
│   └── pipelines/stages/
│       └── denoising.py            # Rollout generation: logprob + trajectory in inference path
```

## Where things live

| Component            | Location |
|---------------------|----------|
| Runscripts          | `examples/training/rl/finetune_t2v_grpo.sh`, `finetune_t2v_grpo_4gpu.sh` |
| Wan RL entry        | `fastvideo/training/wan_rl_training_pipeline.py` |
| RL pipeline (GRPO)  | `fastvideo/training/rl/rl_pipeline.py` |
| Rewards (OCR)       | `fastvideo/training/rl/rewards/` (e.g. `ocr.py`, `rewards.py`) |
| Prompt dataset      | data: `data/ocr/` code: `fastvideo/dataset/rl_prompt_dataset.py` |
| Rollout / sampling  | `fastvideo/pipelines/stages/denoising.py` (inference path + logprob) |

---

## Environment

On our machine, use the conda environment **fastvideo_shijie**. Activate it before running the scripts (e.g. `conda activate fastvideo_shijie`). This environment is not defined in a file in the repo; it is assumed to exist on the host.

---

## Component overview

**Runscripts.** The entry points for GRPO training are the bash scripts under `examples/training/rl/`. `finetune_t2v_grpo.sh` runs single-GPU training and `finetune_t2v_grpo_4gpu.sh` runs on four GPUs. Both invoke `fastvideo/training/wan_rl_training_pipeline.py` via `torchrun` and pass model, dataset, RL, and validation arguments. `validation.json` in the same directory configures validation prompts used during training.

**Wan RL entry point.** `fastvideo/training/wan_rl_training_pipeline.py` defines `WanRLTrainingPipeline`, which extends `RLPipeline` with Wan-specific setup. It builds the inference pipeline used for rollouts (Wan pipeline with UniPC scheduler and shared transformer/VAE/text encoder), and wires it into the RL loop. All RL logic lives in the base `RLPipeline`; this module only handles Wan pipeline creation and config.

**RL pipeline (GRPO).** The core training loop is in `fastvideo/training/rl/rl_pipeline.py`. `RLPipeline` subclasses the generic `TrainingPipeline` and implements trajectory collection (generating videos with log-probabilities), reward computation via the reward models, per-prompt advantage normalization, and the GRPO policy loss (with clipping and KL). It also manages the value model and its optimizer when used. Rollouts are produced by the sampling pipeline; log-probabilities come from the denoising stage’s `sde_step_with_logprob` in `fastvideo/pipelines/stages/denoising.py`.

**Rewards.** Reward models live under `fastvideo/training/rl/rewards/`. `base.py` defines `BaseRewardModel`, the abstract interface for video reward models (input shape [B, T, C, H, W]). `rewards.py` provides `MultiRewardAggregator` (combining several reward models with weights) and `ValueModel` / `create_reward_models`, used by the pipeline to build the reward stack. The current default reward is the OCR scorer in `ocr.py`, which uses PaddleOCR on sampled frames and compares to the prompt text (e.g. text in quotes) to produce a scalar reward per video.

**Per-prompt stat tracking.** `fastvideo/training/rl/stat_tracking.py` implements `PerPromptStatTracker`, which keeps reward statistics per unique prompt and computes normalized advantages (e.g. per-prompt mean and std for GRPO). The pipeline clears these stats after each advantage computation so normalization uses only the current batch.

**RL utilities.** `fastvideo/training/rl/rl_utils.py` holds shared helpers for RL training, including `compute_gae` (Generalized Advantage Estimation) and reward-statistics helpers used by the pipeline. `fastvideo/training/rl/wan_grpo_utils.py` contains Wan-specific GRPO helpers: it adapts the SDE step and sampling flow from FlowGRPO to FastVideo’s scheduler and `WanPipeline`, and is used for rollout generation with log-probabilities.

**Prompt dataset.** The RL prompt dataloader is implemented in `fastvideo/dataset/rl_prompt_dataset.py`. It supports text-file prompts (one per line) and GenEval-style JSONL via `TextPromptDataset` and `GenevalPromptDataset`, and provides `KRepeatSampler` so each prompt is repeated k times per step for GRPO. The data files themselves (e.g. `train.txt`, `test.txt`) live under `data/ocr/` when using the text-prompt dataset.

**Rollout generation.** Video rollouts and their log-probabilities are produced in the inference path of the denoising stage. In `fastvideo/pipelines/stages/denoising.py`, `sde_step_with_logprob` performs the SDE step and returns the log-probability of the transition; the denoising stage calls it for both policy and reference (for KL) when running in RL mode, so the pipeline can compute GRPO loss.

**Transformer forward args (log_prob replay).** To ensure the first log_prob after trajectory collection matches the log_prob from the trajectory collection process, the transformer forward in `_compute_log_prob_for_timestep` (GRPO loss) must use the same inputs and `set_forward_context` as the DenoisingStage. The DenoisingStage therefore saves, when RL is enabled, the transformer forward args (per-step context: `current_timestep`, `attn_metadata`; trajectory-scope: `image_kwargs`, `pos_cond_kwargs`, `neg_cond_kwargs`, `action_kwargs`, `guidance_expand`, `video_latent`, `image_latent`, `forward_batch_snapshot`) into `ForwardBatch.RLData.transformer_forward_args`. These are copied to `TrainingBatch.transformer_forward_args` in `collect_trajectories` and used in `_compute_log_prob_for_timestep` to replay the same forward. Data structures live in `fastvideo/pipelines/pipeline_batch_info.py`: `RLTransformerForwardArgs`, `TrajectoryScopeForwardArgs`, `PerStepForwardContext`.

**Multiple batches per step (flow_grpo-style).** `--rl-num-batches-per-step` (default 1) controls how many sampling batches are collected per training step. Each batch is processed independently: get prompts, collect trajectories, compute rewards, then compute advantages (per-prompt stats within that batch). When > 1, one optimizer step and one log are performed per batch; when 1, `gradient_accumulation_steps` batches are accumulated then one optimizer step and one log. No concatenation or splitting; advantages are computed per batch so each trajectory's advantage uses only its group.

---

## Source codebase: flow_grpo

The FastVideo RL pipeline above was ported from the **flow_grpo** repo. This section documents the original flow_grpo layout so readers can map concepts and file locations between the two codebases.

### How to run (flow_grpo)

From the flow_grpo repo root:

```bash
bash train_wan.sh
```

This launches `accelerate launch` with `scripts/accelerate_configs/multi_gpu.yaml` and runs `scripts/train_wan2_1.py` with config `config/grpo.py:general_ocr_wan2_1`.

### Code structure (flow_grpo)

```
flow_grpo/
├── train_wan.sh                      # Wan GRPO runscript (accelerate + train_wan2_1.py)
├── config/
│   ├── base.py                       # Base config (ml_collections)
│   └── grpo.py                       # GRPO configs (e.g. general_ocr_wan2_1)
│
├── dataset/                          # Prompt datasets (data files)
│   ├── ocr/                          # Text prompts (train.txt, test.txt)
│   ├── geneval/                      # GenEval JSONL (train_metadata.jsonl, test_metadata.jsonl)
│   └── ...
│
├── scripts/
│   ├── train_wan2_1.py               # Wan2.1 GRPO training entry: loop, dataloader, rollout, reward, advantage, loss
│   └── accelerate_configs/
│       └── multi_gpu.yaml             # Multi-GPU / bf16 accelerate config
│
└── flow_grpo/
    ├── diffusers_patch/
    │   ├── wan_pipeline_with_logprob.py   # Rollout + logprob: sde_step_with_logprob, wan_pipeline_with_logprob
    │   └── wan_prompt_embedding.py        # encode_prompt for Wan
    ├── rewards.py                    # Reward functions (video_ocr, jpeg_compressibility, etc.)
    ├── stat_tracking.py              # PerPromptStatTracker (per-prompt advantage normalization)
    ├── prompts.py                    # Prompt sampling (general_ocr, from_file, etc.)
    └── ema.py                        # EMAModuleWrapper for policy EMA
```

### Where things live (flow_grpo)

| Component            | Location |
|----------------------|----------|
| Runscript            | `train_wan.sh` (repo root) |
| Training entry       | `scripts/train_wan2_1.py` |
| Wan pipeline + logprob | `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py` |
| Config               | `config/grpo.py` (and `config/base.py`) |
| Accelerate config    | `scripts/accelerate_configs/multi_gpu.yaml` |
| Rewards              | `flow_grpo/rewards.py` (e.g. video_ocr), `flow_grpo/ocr.py` |
| Per-prompt stats     | `flow_grpo/stat_tracking.py` |
| Prompts              | `flow_grpo/prompts.py` |
| Prompt dataset (code)| `TextPromptDataset`, `GenevalPromptDataset`, `DistributedKRepeatSampler` in `scripts/train_wan2_1.py` |
| Prompt dataset (data)| `dataset/ocr/`, `dataset/geneval/`, etc. |
| EMA                  | `flow_grpo/ema.py` |

### Component overview (flow_grpo)

**Runscript.** `train_wan.sh` at the repo root sets `PYTHONPATH` and `WANDB_API_KEY`, then runs `accelerate launch` with `--config_file scripts/accelerate_configs/multi_gpu.yaml` and `--main_process_port 29503`, invoking `scripts/train_wan2_1.py` with `--config config/grpo.py:general_ocr_wan2_1`. Optional single-GPU usage is commented out (e.g. `--num_processes 1`).

**Training script.** `scripts/train_wan2_1.py` is the Wan2.1 GRPO entry point. It defines the training loop: load config, build Wan pipeline and scheduler, create dataloader (with `TextPromptDataset` or `GenevalPromptDataset` and `DistributedKRepeatSampler`), and each step collects rollouts via `wan_pipeline_with_logprob`, computes rewards via `flow_grpo.rewards`, normalizes advantages with `PerPromptStatTracker`, computes log-probabilities for the GRPO loss, and updates the transformer (with optional EMA). It also contains `eval`, `save_ckpt`, `compute_log_prob`, and helper functions (`get_sigmas`, `set_adapter_and_freeze_params`, etc.).

**Wan pipeline with logprob.** `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py` provides `sde_step_with_logprob` (UniPC-style SDE step returning transition log-probability) and `wan_pipeline_with_logprob` (full pipeline call for generation with log-probabilities and optional KL reward). These are used by the training script for rollout generation and for GRPO loss computation.

**Config.** `config/grpo.py` defines ml_collections configs such as `general_ocr_wan2_1()`: dataset path (`dataset/ocr`), pretrained model (e.g. Wan2.1-T2V-1.3B-Diffusers), sample settings (steps, resolution, batch sizes, num_image_per_prompt, num_batches_per_epoch), train settings (learning rate, beta, clip_range, gradient_accumulation_steps, timestep_fraction), reward (`reward_fn`, e.g. video_ocr), prompt fn (e.g. general_ocr), per-prompt stat tracking, EMA, and save/eval paths. It imports and extends `config/base.py` for default keys.

**Accelerate config.** `scripts/accelerate_configs/multi_gpu.yaml` sets `distributed_type: MULTI_GPU`, `mixed_precision: bf16`, `downcast_bf16: yes`, and `num_processes: 8` (single machine). Used by `train_wan.sh` for multi-GPU launches.

**Rewards.** `flow_grpo/rewards.py` implements reward functions (e.g. jpeg compressibility, aesthetic score, CLIP score, pickscore, and video_ocr which uses `flow_grpo/ocr.py`). The config’s `reward_fn` (e.g. `{"video_ocr": 1.0}`) selects and weights them; the training script builds the reward callable from this and evaluates it on sampled videos/frames.

**Per-prompt stat tracking.** `flow_grpo/stat_tracking.py` defines `PerPromptStatTracker`, which keeps reward statistics per unique prompt and computes normalized advantages (per-prompt mean and std, or global std when `global_std=True`). Used in the training script for GRPO advantage normalization.

**Prompts.** `flow_grpo/prompts.py` provides prompt-sampling helpers (e.g. `general_ocr`, `from_file`) that load lines from assets or dataset files and return a random prompt. The config’s `prompt_fn` names the function used during training.

**Dataset.** Prompt data lives under `dataset/ocr/` (train.txt, test.txt) or `dataset/geneval/` (train_metadata.jsonl, test_metadata.jsonl). The dataset classes and `DistributedKRepeatSampler` (k repeats per prompt for GRPO) are defined in `scripts/train_wan2_1.py`; the script builds the dataloader from the config’s dataset path and split.

**EMA.** `flow_grpo/ema.py` provides `EMAModuleWrapper` for exponential moving average of the policy; the training script optionally maintains an EMA model and can save it with checkpoints.

---

# Debugging message rules

Debugging messages should be printed using the logger.info function
Newly added debugging messages should preceed with "RL_METRIC: " before its content
Debugging messages should be wrapped in myregion comments; the debug code snippet should start with "# myregion debug", and end with "# endregion"
