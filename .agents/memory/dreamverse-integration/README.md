# Dreamverse Integration — Memory Index

Living knowledge base for the FastVideo ↔ Dreamverse ↔ Dynamo integration.
Tracks the public API refactor (PRs 0-17), the LTX-2 streaming server
upstream, the Dreamverse switch from `FastVideo-internal` to public
`FastVideo`, and the NVFP4 quantization landing.

**Last reconciled:** 2026-05-04 (FastVideo `will/ltx2_sr_port` @ `89a6484d`
post-#1258-merge rebase; Dreamverse `will/integrate-public-fastvideo` @ `ec8ef92`).
PRs #1257 (`will/api_7.6`) and #1258 (`will/api_7.7`) MERGED to main; PR #1284
(`will/api_7.8`) OPEN, awaiting review.

## Reading guide — what to load when

| Question / task | File |
|---|---|
| "What's running right now? What just landed?" | [state.md](state.md) |
| "Why is the schema typed this way? What's the philosophy?" | [design.md](design.md) |
| "What PRs landed? In flight? Planned?" | [pr-roadmap.md](pr-roadmap.md) |
| "Streaming server, `generate_async`, `build_app` routes?" | [streaming-server.md](streaming-server.md) |
| "How does Dreamverse use FastVideo? What about Dynamo?" | [cross-repo-surfaces.md](cross-repo-surfaces.md) |
| "NVFP4? Layer profiles? `LinearBase` fallback? AbsMaxFP8?" | [quantization.md](quantization.md) |
| "Why was decision X made? What's resolved vs. open?" | [decisions-log.md](decisions-log.md) |
| "What should I work on next? Priority order?" | [open-threads.md](open-threads.md) |
| "Who should be co-authored on commits in this scope?" | [authors.md](authors.md) |

## Repo + worktree paths

| Repo | Path | Active branch |
|---|---|---|
| FastVideo (public) | `/home/william5lin/FastVideo` | `will/ltx2_sr_port` |
| Dreamverse | `/home/william5lin/Dreamverse` | `will/integrate-public-fastvideo` |
| FastVideo-internal (read-only ref) | `/home/william5lin/FastVideo-internal` | their `main` |
| Dynamo (read-only ref) | `/home/william5lin/dynamo` | upstream |

## Glossary

- **NVFP4**: NVIDIA's specific block-scaled FP4 (e2m1 mantissa, fp32 alpha,
  `layout_128x4` scale layout, group size 16). Distinct from MX-FP4 / OCP-FP4.
- **`GeneratorConfig`**: typed init-time public config (model_path, engine,
  pipeline). Replaces flat `from_pretrained(**kwargs)`.
- **`GenerationRequest`**: typed per-call request (prompt, inputs, sampling,
  runtime, output, stage_overrides, state, plan, extensions). Replaces flat
  `generate_video(**kwargs)`.
- **`ServeConfig`** / **`RunConfig`**: top-level YAML envelopes. ServeConfig
  for `fastvideo serve`; RunConfig for offline `fastvideo generate`.
- **`InferencePreset`**: model-owned named preset (e.g. `ltx2_two_stage`)
  defining stage topology + per-stage defaults + valid override types.
- **`ContinuationState`**: opaque round-trip state envelope `{kind, payload}`.
  Hybrid: server-held for streaming WS, client-round-trip for stateless HTTP.
- **`generate_async`**: future canonical async exec API (PR 7.10) yielding
  `VideoProgressEvent` / `VideoPartialEvent` / `VideoFinalEvent`. Substrate
  for streaming server, OpenAI server, AND Dynamo backend.
- **`build_app`**: FastAPI app factory in
  `fastvideo.entrypoints.streaming.server`. Currently exposes only
  `/health` + `/v1/stream`. FE-required `/healthz`+`/readyz`+`/status`
  migration is open follow-up #1.
- **`LLMProvider`**: protocol abstraction for prompt enhancer providers
  (cerebras, cerebras_ifm, groq). Public schema currently restricts to
  `Literal["cerebras", "groq"]`; `cerebras_ifm` is internal-only.
- **`compat.py`**: legacy kwargs translation layer (~370 lines). Scheduled
  for death across PRs 14-17.
- **`prepare_for_compile`**: duck-type protocol method called via
  `getattr(module, "prepare_for_compile", None)` before `torch.compile`.
  Currently only Gemma3 implements it.
- **`SubprocessGpuPool`**: PR 7.6 public replacement for the internal
  `realtime/local_runtime.GPUPool`. Per-GPU subprocess workers, typed
  `GeneratorConfig` boundary.
- **PR 5.5**: streaming server subpackage skeleton — adds
  `fastvideo/entrypoints/streaming/` parallel to `openai/`.
- **PR 7.10**: the unlock PR. Closes Q-5 (audio re-encode), Q-9 (Dynamo
  progress), and PR 7.5's mid-segment cancellation TODO simultaneously.

## Live process map (as of 2026-05-03)

| Port | Service | Source |
|---|---|---|
| 8009 | `dreamverse-server` | running, `/readyz` 200, 1 warmed GPU worker |
| 5274 | `next-server` (dev) | running |
| 8000 | unknown FastAPI | not in handoff — verify before launching new BE |

## How this directory is maintained

- Source of truth for the integration story. Update when state changes.
- Each file has a "Last updated" header; bump when you edit.
- Cross-reference siblings via relative links; do NOT duplicate content.
- New entries: register in `../index.jsonl`.
- These files supersede the untracked source docs in the repo root and
  `.agents/exploration/` — see [state.md](state.md) "Untracked but
  present" section for disposition.

## Source documents (archived 2026-05-03)

The 7 source docs that this directory consolidates have been moved into
[`source-archive/`](source-archive/). They remain available for agents
who want the full unsynthesized rationale, but the synthesized memory
files in this dir are the canonical source of truth.

| Source doc | Lines | Synthesized into |
|---|---|---|
| [`source-archive/apirefactor.md`](source-archive/apirefactor.md) | 838 | [design.md](design.md) |
| [`source-archive/PR-plan.md`](source-archive/PR-plan.md) | 1145 | [pr-roadmap.md](pr-roadmap.md) |
| [`source-archive/dreamverse_review.md`](source-archive/dreamverse_review.md) | 390 | [state.md](state.md) + [decisions-log.md](decisions-log.md) |
| [`source-archive/handoff-nvfp4-launch-demo.md`](source-archive/handoff-nvfp4-launch-demo.md) | 518 | [state.md](state.md) + [quantization.md](quantization.md) + [open-threads.md](open-threads.md) |
| [`source-archive/streaming-server-upstream-plan.md`](source-archive/streaming-server-upstream-plan.md) | 539 | [streaming-server.md](streaming-server.md) + [decisions-log.md](decisions-log.md) |
| [`source-archive/dreamverse_integration.md`](source-archive/dreamverse_integration.md) | 285 | [cross-repo-surfaces.md](cross-repo-surfaces.md) |
| [`source-archive/video-generator-config-api-design.md`](source-archive/video-generator-config-api-design.md) | 93 | [design.md](design.md) (early-draft material) |
| `.agents/exploration/pr-link-review.md` | 29 | already promoted to `.agents/skills/review-pr-link/` (kept in exploration dir) |

See [`source-archive/README.md`](source-archive/README.md) for the
archive policy.
