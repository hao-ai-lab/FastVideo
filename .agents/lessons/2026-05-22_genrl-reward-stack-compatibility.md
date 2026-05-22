---
date: 2026-05-22
experiment: genrl_longcat_probe_001
category: porting
severity: critical
---

# GenRL Reward Stack Compatibility Must Be Preflighted

## What Happened
The GenRL LongCat Modal launch repeatedly failed before the first optimizer
step while initializing HPSv3 and VideoAlign rewards. Failures included missing
submodule dependencies, HPSv3 importing a removed Transformers alias,
FlashAttention being forced when unavailable, Qwen2-VL checkpoint key drift,
PEFT/LoRA-prefixed VideoAlign checkpoint key drift, and old Qwen2-VL forward
code expecting `model.embed_tokens`.

## Root Cause
The reward models are vendored research repositories with their own dependency
assumptions:

- FastVideo uses Transformers 4.57.x and Torch 2.10+.
- HPSv3 and VideoAlign were written around older Qwen2-VL / Transformers
  layouts.
- VideoAlign checkpoints are PEFT/LoRA-wrapped, so key drift appears under
  `base_model.model.*` prefixes, not only raw model keys.
- FastVideo treats FlashAttention as recommended/optional, while VideoAlign
  forced `flash_attention_2` unless explicitly disabled.

The initial preflight checked only shallow setup. It did not instantiate every
reward, load each checkpoint, run a dummy scoring call, and then move/clear the
reward models.

## Fix / Workaround
Use the parent FastVideo reward wrappers, not submodule edits, to patch
compatibility:

- Alias missing Transformers `VideoInput`.
- Remap old Qwen2-VL checkpoint keys to current `model.visual.*` and
  `model.language_model.*` names.
- Remap PEFT/LoRA-prefixed VideoAlign keys under
  `base_model.model.model.*`.
- Add runtime `embed_tokens` aliases through wrapper/base-model graphs.
- Install FlashAttention in the Modal image, while retaining an SDPA fallback.
- Modal preflight now exercises HPSv3 general, HPSv3 percentile, VideoAlign MQ,
  and VideoAlign TA with dummy videos before Wan sampling starts.

## Prevention
Before launching a costly RL run, require a reward-stack preflight that:

- validates W&B and HF identity,
- validates prompt JSONL files are real data, not Git LFS pointers,
- validates enough prompts exist for all distributed ranks,
- validates reward checkpoint completeness,
- imports version-sensitive dependencies and prints versions,
- instantiates every enabled reward model,
- loads each checkpoint strictly enough to catch missing/unexpected keys,
- runs one dummy reward call for each reward head,
- moves reward models back to CPU and clears caches.

Do not treat first visible checkpoint-key errors as isolated. For vendored
Qwen2-VL reward repos, inspect raw and PEFT-prefixed state dict paths together.
