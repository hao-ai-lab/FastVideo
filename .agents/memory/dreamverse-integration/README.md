# Dreamverse Integration — Memory Index

Living knowledge base for the FastVideo ↔ Dreamverse ↔ Dynamo integration.
Tracks the public API refactor (PRs 0-17), the LTX-2 streaming server
upstream, the Dreamverse switch from `FastVideo-internal` to public
`FastVideo`, and the NVFP4 quantization landing.

**Last reconciled:** 2026-05-06 (**D-26** EXECUTED — rebased
`will/dreamverse-monorepo` directly onto `origin/main` (`c17d33bf`)
via `git rebase origin/main`; 66 commits cleanly replayed; force-
pushed via `--force-with-lease`. New tip `83829c5e`. Local backup
branch `will/dreamverse-monorepo-pre-main-rebase-backup-20260506`
preserved at the pre-rebase tip `2ee839a3`. PR #1288 on
`will/ltx2_sr_port` is untouched. The branch is now ready to open
as a single PR against main.

**Earlier — D-21** EXECUTED — chunk-stutter root cause
analysis + NVENC build path + opt-in `--nvenc` flag + benchmark regression
test. 3 parallel explore agents confirmed apps/dreamverse matches
FastVideo-internal byte-for-byte on NVFP4 + torch.compile coverage; stutter
is NOT a regression. Software libx264 encoding consumes ~22% of segment
wall-time. Built ffmpeg with NVENC support; B200 silicon doesn't have NVENC
encoder hardware so the path is currently moot on this dev host but works
on RTX 50-series / T4 / A10 deploys. Benchmark captured libx264 ultrafast
at 611ms median (8.25x realtime in isolation). Open follow-ups D-22/D-23/D-24.

**Earlier — D-20** EXECUTED — segment-2 BrokenPipe
root cause was a TWO-direction silent drop of LTX-2 audio kwargs in
public `VideoGenerator` (inbound `SamplingParam.update()` rejected
`audio_num_frames`/`ltx2_audio_clean_latent`/etc. as unknown fields and
`logger.error`'d, outbound result dict didn't surface
`ltx2_audio_latents` from `output_batch.extra`). Ported the
FastVideo-internal extra-overrides routing block + made `update()`
strict + added regression test (7 tests, all pass) + landed 4 commits
on `will/dreamverse-monorepo` @ `5eaf0a13` (11 commits ahead of
`fbd823df`). End-to-end verified on GPU4: `Cached audio latents shape
=(1, 8, 126, 16) for segment 2`, `Segment 2: relayed av chunks=22,
bytes=3.8MB`, no BrokenPipeError. Public-API fix needs cherry-pick to
`will/ltx2_sr_port` for PR #1288 — see open-threads.md item D-20-CP.
**D-19** EXECUTED previously: Dreamverse migration landed on
`will/dreamverse-monorepo` @ `c1fe5d4c` (5 commits ahead of
`will/ltx2_sr_port` HEAD `fbd823df`). 164 files, 53,294 LOC, 31,725
files under `apps/dreamverse/`. e2e PASSES against migrated code (8/8
Playwright in 5.1s, `/proc/$PID/cwd` verified). Significant deviation
from [integration-plan.md](integration-plan.md): the plan's "DELETE
generic-merged from Dreamverse, import public substitutes" assumption
was invalid (public APIs aren't drop-ins) — generic-merged files now
carried PRODUCT-LOCAL inside `apps/dreamverse/server/`. Public
`fastvideo.entrypoints.streaming.*` reverts to `fbd823df` state. See
[decisions-log.md D-19](decisions-log.md#d-19) +
[D-20](decisions-log.md#d-20) for full context.).
FastVideo `will/ltx2_sr_port` @ HEAD (post-D-17 STACK.md removal +
integration-review.md addition + integration-plan.md addition + D-18
reconciliation). Dreamverse `will/integrate-public-fastvideo` @ `ec8ef92`.
PRs #1257 / #1258 / #1284 / #1286 MERGED to main. **PR #1287 CLOSED
(in favor of consolidation); PR #1288 OPEN as the single mega-PR
landing the entire `will/ltx2_sr_port` chain at once** (LTX-2 SR
runtime + NVFP4 + `generate_async`/Dynamo contract + agents memory dir).
Split branches kept as historical bookmarks; STACK.md model **abandoned** —
see [decisions-log.md D-17](decisions-log.md#d-17). Local backup
`will/ltx2_sr_port-pre-1286-rebase` @ `1baa60bb` preserves the
pre-rebase chain.

## Fresh-context onboarding (read in order)

If you're an agent picking up this work for the first time, do these
**5 things in this order**. Once done, you have full context to continue
any open thread, commit correctly, push, and propagate to the open PR.

1. **Confirm worktree state** — run the "First 60 seconds" block in
   [runbook.md](runbook.md). Tells you the branch is right, services
   are up, and PR #1286's head matches what this dir claims.

2. **Read [state.md](state.md)** — single-page snapshot of branch tips,
   live services, test status, pre-existing failures, "do not pop"
   stashes.

3. **Read [pr-roadmap.md](pr-roadmap.md)** — what PRs landed, what's in
   flight, what's planned. Identifies the active open PR (currently
   #1286) and where it sits in the dependency chain.

4. **Read [open-threads.md](open-threads.md)** — prioritized work items
   with effort estimates and dependencies. The "Recommended pull order"
   section is a ready-made TODO list if you need one.

5. **Skim [runbook.md](runbook.md) end-to-end** — operational how-to:
   verify, commit (with co-author trailers), push, propagate to PR
   #1286, maintain the memory dir, and the "Common pitfalls" section
   that catches the recurring traps.

Skip the deep-context docs (design / streaming-server / cross-repo /
quantization / decisions-log) until you need them — they're indexed in
the "Deep-dive reading guide" below.

Final check: run the "Self-test" block at the bottom of
[runbook.md](runbook.md). If you can answer all 8 questions from this
dir alone, you're ready. If you can't, the gap is a memory-dir bug —
file it in [open-threads.md](open-threads.md) before continuing.

## Deep-dive reading guide

| Question / task | File |
|---|---|
| "What's running right now? What just landed?" | [state.md](state.md) |
| "How do I commit / push / propagate to PR #1286?" | [runbook.md](runbook.md) |
| "Why is the schema typed this way? What's the philosophy?" | [design.md](design.md) |
| "What PRs landed? In flight? Planned?" | [pr-roadmap.md](pr-roadmap.md) |
| "Streaming server, `generate_async`, `build_app` routes?" | [streaming-server.md](streaming-server.md) |
| "How does Dreamverse use FastVideo? What about Dynamo?" | [cross-repo-surfaces.md](cross-repo-surfaces.md) |
| "NVFP4? Layer profiles? `LinearBase` fallback? AbsMaxFP8?" | [quantization.md](quantization.md) |
| "Why was decision X made? What's resolved vs. open?" | [decisions-log.md](decisions-log.md) |
| "What should I work on next? Priority order?" | [open-threads.md](open-threads.md) |
| "Who should be co-authored on commits in this scope?" | [authors.md](authors.md) |
| "How do we execute the Dreamverse → FastVideo monorepo merge?" | [integration-plan.md](integration-plan.md) ← **CURRENT** |
| "Historical drift audit + Option-D evaluation (deprecated by D-18)" | [integration-review.md](integration-review.md) (DEPRECATED) |

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
