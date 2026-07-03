# FastVideo Interleave Example

This directory contains a small Python example for running an application-level
InterleaveThinker-style image generation trace on top of FastVideo. The helper
code lives under `apps/interleave_thinker` so it stays separate from FastVideo's
core generation package.

## Single-Prompt Trace

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
  python examples/interleave/interleave_single_prompt.py \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "a brushed steel espresso machine on a marble counter, morning window light" \
    --output-dir outputs/interleave_single_prompt
```

The script uses `VideoGenerator` directly through the local FastVideo image
backend. It writes the generated image and a `trace.json` file under the output
directory. The trace records the generation attempt and omits base64 image
payloads by default.

## How The Example Planner Works

`SinglePromptPlanner` is intentionally minimal: it takes the user instruction
and creates one generation step with that exact prompt. It does not decompose
the instruction, call a learned planner, or create multiple edit steps.

`AcceptAllCritic` is equally small. It accepts the first generated image, so the
example can produce a complete trace without requiring a learned critic model.
The orchestration code still records the planner, generator, and critic result
so a trace has the same shape when a richer app-specific planner or critic is
added later.

## Optional Gemini Backends

The app helper can also use closed-source Google image models through lazy
wrappers when you instantiate the backend directly:

- `apps.interleave_thinker.generator.NanoBananaImageGeneratorBackend`
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
