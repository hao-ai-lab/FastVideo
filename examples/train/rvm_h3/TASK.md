# FastH3 RVM post-training

This branch implements reward-based velocity matching (RVM) for the released four-step FastH3 checkpoint.

## Scope

- Train a LoRA on the FastH3 student with on-policy, group-relative video rewards.
- Preserve FastH3's exact CFG-free four-step VSA rollout contract.
- Apply reward-weighted velocity matching to the video latent stream.
- Leave audio out of the reward term and optionally anchor its velocity to the released FastH3 policy.
- Provide one-GPU correctness checks, eight-GPU experiment configs, prompt preprocessing, reward setup, checkpoint finalization, inference, and evaluation scripts.
- Evaluate every 5% of optimizer progress on a deterministic set of at most 100 prompts.

## Non-goals

The initial implementation does not introduce a new teacher-guided objective, DMD critic, GRPO trajectory likelihood, or audio reward. The priority is a reproducible implementation of the published RVM training pattern adapted conservatively to H3's paired video/audio scheduler.

## Required acceptance gates

1. Training rollout uses the same four denoising intervals, CFG=1.0, VSA backend, sparsity, and prompt conditioning as FastH3 inference.
2. Zero-initialized LoRA reproduces the frozen checkpoint.
3. Equal rewards produce zero group-relative RVM signal.
4. Positive and negative advantages move the velocity prediction in opposite directions.
5. Video and audio losses are normalized per modality.
6. The complete reward mixture can score fixed videos deterministically before distributed training starts.
7. Validation runs at 5% optimizer-step intervals and never exceeds 100 prompts.
