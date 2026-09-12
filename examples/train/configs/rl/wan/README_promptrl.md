# PromptRL for Wan video generation

PromptRL trains a **prompt-refiner LoRA** (Qwen2.5-VL-3B-Instruct) and —
in the joint milestone — a **Wan LoRA** with shared group-relative
rewards.  The canonical target is Wan2.1 T2V 1.3B at 480×832 / 77
frames on eight 80GB training GPUs, with VideoScore2 running as a
pluggable reward service on a ninth GPU.

## Milestones

1. **Prompt-only** (`promptrl_prompt_only.yaml`): train the refiner LoRA
   while Wan stays frozen.  Rollout/transition storage is skipped.
2. **Joint** (`promptrl_joint.yaml`): train independent LoRA adapters
   for the refiner and Wan.  Shared advantages are detached before both
   policy losses, so no gradients cross between the models.  The refiner
   initializes from the prompt-only adapter
   (`models.refiner.init_adapter_from`); Wan's LoRA initializes at zero.

## Group layout per step

One original prompt is replicated across the eight training ranks; each
rank produces one independently seeded candidate:

| Ranks | Role               | Generation prompt        | Format reward |
|-------|--------------------|--------------------------|---------------|
| 0–1   | retained originals | original prompt          | 1             |
| 2–7   | refined            | sampled refiner output*  | 1 if valid    |

\* Malformed refinements (missing `<answer>...</answer>`) receive format
reward 0 and fall back to the original prompt for video generation; the
completion is preserved so it still receives its negative LM signal.
Every rank executes the refiner forward/backward path — retained
originals use zero refiner advantage — so distributed collectives stay
aligned.

Videos are scored against the **original** prompt (never the refined
one).  The composite reward `format + VideoScore2` is normalized within
each `(group, reward_tag)`; zero-variance groups yield zero advantages.

## Reward service

The trainer depends only on the reward-provider contract
(`score(samples) -> results`).  The built-in service collects each complete
group and evaluates every video with the official VideoScore2 procedure:
2 FPS video sampling, the published three-dimension prompt, temperature-0.7
generation, and confidence-weighted scores from the generated `1`–`5` token
logits. A fixed seed makes repeated requests reproducible, and completed
request IDs are cached for idempotent retries:

```bash
# reward host (ninth GPU)
CUDA_VISIBLE_DEVICES=0 python -m fastvideo.train.methods.rl.promptrl.rewards.serve \
    --model-id TIGER-Lab/VideoScore2 --port 8100

# training host (eight GPUs)
torchrun --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/rl/wan/promptrl_prompt_only.yaml
```

Endpoints: `GET /healthz`, `POST /v1/rewards:score` (multipart).  Any
timeout, duplicate/missing sample, non-finite score, or cardinality
mismatch fails the step consistently on every rank — zero rewards are
never substituted.  Videos transfer over HTTP only; no shared
filesystem is required.

## Prompt dataset

Raw JSONL or Parquet (not preprocessed):

```json
{"id": "p1", "prompt": "a cat riding a skateboard", "reward_tag": "animals"}
```

`prompt` is required; `id` and `reward_tag` are optional.

## Bundles and inference

The `promptrl_export` callback writes an inference bundle at train end
(`manifest.json`, `refiner/` PEFT adapter, `generator/` Wan LoRA
safetensors + prompt template/version, refiner sampling config, base
model identifiers, compatible FastVideo version).  Generate with the
typed API:

```bash
python examples/inference/promptrl/promptrl_wan_inference.py \
    --bundle outputs/wan2.1_promptrl_joint/bundle \
    --prompt "a cat riding a skateboard"
```

## Observability

Per-step metrics include group reward statistics, VideoScore2
components, refinement validity, refined-vs-original reward gap,
LM loss/KL/completion length, Wan policy loss/KL/ratio/clip fraction,
rollout/reward/backward latency, and peak GPU memory.  Only group
leaders log prompt samples or videos.

## Notes

- Distributed layout: `sp_size=1`, `hsdp_shard_dim=8`; Wan uses HSDP
  while the refiner is replicated with DDP-synchronized LoRA gradients.
- Wan rollouts use 20 flow-matching steps; only the last 8 are
  stochastic SDE transitions whose states/old log-probs are stored and
  recomputed (microbatch 1) in the joint loss.
- Reference policies are adapter-disabled forward contexts (PEFT
  `disable_adapter` for the refiner, `disable_lora` for Wan) — no
  duplicate frozen model copies.
- Checkpointing covers both adapters, both optimizers/schedulers, and
  all RNG states; prompt iteration is a deterministic function of the
  step index, so resume continues the exact prompt stream.
- Out of scope for v1: FLUX/image reproduction, full-parameter
  training, automatic prompt refinement inside `VideoGenerator`.
