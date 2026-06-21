# FastVideo Interleave Examples

This directory contains a small Python example for the reusable Interleave
orchestration helpers. It does not add FastVideo CLI commands or HTTP routes.

## Single-Prompt Trace

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  python examples/interleave/interleave_single_prompt.py \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "a brushed steel espresso machine on a marble counter, morning window light" \
    --output-dir outputs/interleave_single_prompt
```

The script uses `VideoGenerator` directly with a fallback single-prompt planner
and accept-all critic. It writes an image plus `trace.json`; the trace records
planner/generator/critic attempts and omits base64 image payloads by default.

## Planner And Critic

The real InterleaveThinker planner and critic wrappers are integrated through
the existing FastVideo training config system:

- `examples/train/configs/interleave_thinker/planner_sft_lora.yaml`
- `examples/train/configs/interleave_thinker/critic_sft_lora.yaml`
- `examples/train/configs/interleave_thinker/planner_smoke.yaml`
- `examples/train/configs/rl/interleave_thinker/critic_grpo.yaml`
- `examples/train/configs/rl/interleave_thinker/planner_grpo.yaml`

## Optional Gemini Backends

The RL reward config can use closed-source Google models through lazy wrappers:

- `fastvideo.train.methods.rl.rewards.GeminiNanoBananaEditScorer` generates
  edits with Nano Banana and scores them with Gemini.
- `fastvideo.entrypoints.interleave.generator.NanoBananaImageGeneratorBackend`
  implements the same image backend protocol as the local FastVideo generator.

Install the optional SDK and provide a key only when using these API backends:

```bash
uv pip install -e ".[eval-judge]"
export GEMINI_API_KEY=...
```

Supported Nano Banana aliases are:

- `nano-banana` -> `gemini-2.5-flash-image`
- `nano-banana-pro` -> `gemini-3-pro-image`
- `nano-banana-2` -> `gemini-3.1-flash-image`
