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
# Debugging message rules

Debugging messages should be printed using the logger.info function
Newly added debugging messages should preceed with "RL_METRIC: " before its content
Debugging messages should be wrapped in myregion comments; the debug code snippet should start with "# myregion debug", and end with "# endregion"
