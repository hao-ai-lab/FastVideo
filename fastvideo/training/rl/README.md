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
