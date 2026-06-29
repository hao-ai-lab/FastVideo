# InterleaveThinker Inference Workflow

This page summarizes the inference half of the InterleaveThinker integration.
It adds reusable Python helpers for interleaved image generation/editing without
adding new `fastvideo` CLI commands or HTTP API routes.

The training half is intentionally deferred to a follow-up PR after this
inference PR lands upstream.

## Scope

This PR adds:

- typed request/response and trace schemas for interleaved workflows;
- a planner -> generator/edit -> critic orchestration loop;
- a FastVideo image backend adapter for `VideoGenerator`;
- an optional Nano Banana image backend with lazy Google SDK imports;
- config-driven single-prompt runs and prompt-set evaluation helpers;
- saved-trace metrics and report helpers;
- a runnable single-prompt example under `examples/interleave/`.

This PR does not add:

- Qwen3-VL planner or critic model wrappers;
- InterleaveThinker SFT or GRPO training methods;
- training YAML configs;
- standalone FastVideo subcommands or a new HTTP service.

## Integrated Surfaces

| Surface | Purpose |
|---------|---------|
| `fastvideo.workflow.interleave_thinker.schema` | Typed edit requests, generated-image records, planner/critic inputs, attempts, and traces. |
| `fastvideo.workflow.interleave_thinker.orchestrator` | Provider-based planner/generator/critic loop with retry/refinement support. |
| `fastvideo.workflow.interleave_thinker.generator` | Image backend protocol plus FastVideo and Nano Banana implementations. |
| `fastvideo.workflow.interleave_thinker.config` | Typed config loading for reusable interleave runs. |
| `fastvideo.workflow.interleave_thinker.runner` | Config-driven runner for the inference-only `single_prompt` planner and `accept_all`/`none` critics. |
| `fastvideo.workflow.interleave_thinker.evaluation` | Prompt-set execution and resumable trace writing. |
| `fastvideo.workflow.interleave_thinker.trace_eval` | Saved-trace metric extraction and JSON/HTML reports. |

The Qwen-backed `interleave_thinker` planner and critic config kinds are
reserved for the follow-up training PR. In this PR, selecting those kinds raises
a clear runtime error instead of importing training-only modules.

## Runtime Flow

1. A user supplies an instruction through a typed config or Python call.
2. A planner provider returns one or more `PlannedInterleaveStep` objects. The
   inference PR ships `SinglePromptPlanner`, which turns the instruction into a
   single generator step.
3. An image backend translates the step into a `GenerationRequest` or external
   image API request.
4. An optional critic reviews the generated image and can request a refined
   prompt for another attempt.
5. The orchestrator records every attempt in an `InterleaveTrace`.
6. Trace helpers write JSON without base64 image payloads by default.

## Validation Expectations

Before the PR is marked ready:

- run `pre-commit run --all-files` in an authoritative environment;
- run focused workflow tests covering backend translation, orchestration,
  config loading, prompt-set execution, and trace evaluation;
- run targeted Wan T2V SSIM on Modal L40S, or document why the workflow helper
  changes are N/A for Wan T2V pixel output;
- rely on upstream CI for pre-commit, Fastcheck, and full-suite gating before
  merge.

## Review Checklist

- Confirm no standalone `fastvideo` CLI command or HTTP API route was added.
- Confirm workflow imports do not require training-only InterleaveThinker model
  modules.
- Confirm optional Google SDK usage is lazy and only required for Nano Banana
  execution.
- Confirm fake-provider tests cover planner, generator, critic, trace, and
  evaluation behavior without credentials.
