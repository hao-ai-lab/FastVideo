# Dreamverse Integration Review Log

This document tracks design decisions, open questions, and integration-time
choices made while landing the public-side stacked PRs (7.7 → 8) and switching
Dreamverse from `FastVideo-internal` to public `FastVideo`. The user will
review this carefully — entries are deliberately verbose about *why*.

## Goal

Replace Dreamverse's dependency on `FastVideo-internal` with the public
`FastVideo` package, using the upstreamed streaming server stack
(`fastvideo.entrypoints.streaming.*`) where Dreamverse currently has local
copies or imports private modules.

## Surfaces Dreamverse currently uses from FastVideo-internal

(from `/home/william5lin/Dreamverse/server/`, scanned 2026-04-26):

| Dreamverse import | Internal path | Public replacement |
|---|---|---|
| `fastvideo.entrypoints.realtime.local_runtime.RealtimeRuntimeConfig` | `FastVideo-internal/fastvideo/entrypoints/realtime/local_runtime.py` | (none) — Dreamverse rewires through `streaming.gpu_pool.SubprocessGpuPool` |
| `fastvideo.entrypoints.realtime.local_runtime.GPUPool` | same as above | `fastvideo.entrypoints.streaming.gpu_pool.SubprocessGpuPool` (PR 7.6) |
| `fastvideo.configs.pipelines.base.PipelineConfig` | already in public | unchanged |
| `fastvideo.entrypoints.video_generator.VideoGenerator` | already in public | unchanged |
| `fastvideo.layers.quantization.fp4_config.FP4Config` | already in public | unchanged |
| `fastvideo.utils.maybe_download_model` | already in public | unchanged |
| `fastvideo.models.audio.ltx2_audio_processing.AudioProcessor` | already in public | unchanged |
| `fastvideo.models.loader.component_loader.ComponentLoader` | already in public | unchanged |
| `fastvideo.models.dits.ltx2.*` | already in public | unchanged |
| local copy: `Dreamverse/server/prompt_enhancer.py` (1933 lines) | mirrors `FastVideo-internal/.../prompt_enhancer.py` | `fastvideo.entrypoints.streaming.prompt.*` (PR 7.7) |
| local copy: `Dreamverse/server/prompt_safety.py` | mirrors `FastVideo-internal/.../prompt_safety.py` | `fastvideo.entrypoints.streaming.prompt.safety` (PR 7.8) |
| local copy: `Dreamverse/server/session_logger.py` | mirrors `FastVideo-internal/.../session_logger.py` | `fastvideo.entrypoints.streaming.session_logger` (PR 7.8) |
| local copy: `Dreamverse/server/rewrite_prompt_payload.py` | mirrors `FastVideo-internal/.../rewrite_prompt_payload.py` | `fastvideo.entrypoints.streaming.prompt.rewrite` (PR 7.8) |
| local copy: `Dreamverse/server/mock_server.py` (1200 lines) | mirrors `FastVideo-internal/.../mock_server.py` | `fastvideo.entrypoints.streaming.mock_server` (PR 7.8) |
| local copy: `Dreamverse/server/session_init_image.py` | mirrors `FastVideo-internal/.../session_init_image.py` | `fastvideo.entrypoints.streaming.session_init_image` (PR 7.5 — already public) |

## Design decisions made (auto-resolved)

### D-1: Realtime runtime → streaming GpuPool migration shape

**Context.** Dreamverse's `server/runtime/gpu_pool.py` thin-wraps
`fastvideo.entrypoints.realtime.local_runtime.GPUPool`, which takes a
`RealtimeRuntimeConfig(model_registry=…, default_model_id=…, default_height=…,
default_width=…, default_num_frames=…, default_num_inference_steps=…,
startup_warmup_*…)`. The public `streaming.gpu_pool.SubprocessGpuPool` takes a
typed `GeneratorConfig` + `GpuPoolConfig` + `WarmupConfig`.

The shapes differ in two important ways:

1. The internal version had a multi-model registry (`model_id → model_config`
   dict). The public version is single-model (one `GeneratorConfig`).
2. The internal version flattened a few sampling defaults (height/width/frames/
   steps) into the runtime config. The public version expects them as part of
   the per-request `SamplingConfig`.

**Decision.** Dreamverse will:
1. Drop the multi-model registry on the integration branch (it is not used in
   production today — Dreamverse boots one model per replica).
2. Construct a `GeneratorConfig` for the chosen model from `MODEL_REGISTRY[id]`
   and pass it to `SubprocessGpuPool`.
3. Move the `default_height` / `default_width` / `default_num_frames` /
   `default_num_inference_steps` defaults into a server-side
   `default_request: GenerationRequest` template the session controller fills
   from per-request input.

**Why.** Multi-model is feasible to add back later (one pool per model id,
acquire by `(session_id, model_id)`), but not on the migration branch — that
would couple the upstream switch to a feature redesign. Punting keeps the
upstream switch a pure mechanical refactor.

**Risk.** If a Dreamverse code path silently relied on the registry to swap
models per-session, the migration branch will surface that as a missing-model
error. The integration tests must exercise at least one segment per supported
model id before merging the Dreamverse branch.

### D-2: PR 7.7 prompt enhancer API surface narrower than the internal one

**Context.** The upstreamed `PromptEnhancer.enhance/auto_extend/rewrite` returns
`LLMResponse(content, provider, model, latency_ms, fallback_used)`. The internal
`enhance_prompt` / `generate_auto_prompt` / `rewrite_prompt_sequence` returns
`EnhanceResult(prompt, fallback_used, error, provider, model, latency_ms)` /
`RewriteResult(prompts, …, rollout_id, rollout_label, raw_response_text)`.

**Decision.** The Dreamverse integration branch will adapt at the call site:
- `enhancer.enhance_prompt(...)` → `enhancer.enhance(prompt)` + a thin shim
  that maps the structured response into the existing `EnhanceResult` shape
  for the session-controller code path. Move the shim to
  `Dreamverse/server/prompting/_internal_compat.py`.
- The locked-segment / next-segment-index plumbing the internal version
  built into the user payload becomes Dreamverse-side template logic in
  the shim.
- The JSON-shaped responses the internal prompts assume (`{"next_prompt":
  "..."}` / `{"segment_prompts": [...]}`) become Dreamverse-side
  parsing in the shim, since the public `LLMResponse` is intentionally raw.

**Why.** The public surface stays minimal and provider-agnostic; the
LTX-2-specific orchestration (locked segments, rollout id/label, JSON
schemas) is an internal-UI concern, not something every public consumer
should wear. Dreamverse keeps its existing call shape; the public stays
clean.

**Open question for review:** Should we promote some of this into
`fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` (or similar)
once a second consumer appears? Logging here so we have the option.

### D-3: Multi-stage provider race (Dreamverse) vs sequential fallback (public)

**Context.** The internal enhancer runs all providers in a stage in parallel
and returns the first to succeed (`_run_provider_race`). The public
enhancer runs providers strictly sequentially with retryable-error fallback.

**Decision.** Public stays sequential for PR 7.7. The race-based fallback is
a Dreamverse-specific tail-latency optimization that depends on parallel API
budgets; promoting it would force every public consumer to have multiple
provider keys configured. Dreamverse can keep `_run_provider_race` as an
internal optimization on its side.

**Risk.** First-segment latency on Dreamverse may regress slightly when
Cerebras is having a bad minute (sequential fallback waits the full
20s timeout before trying Groq). If this is a real production concern,
add a public knob like `concurrency: int = 1` on `PromptEnhancer` that
gates a race path — but only after measuring.

### D-4: Skipping PR 7.9 router for the integration branch

**Context.** The internal stack ships a `router/main.py` that load-balances
across replicas with health checks. Dreamverse's deployment uses a single
replica per region (per `gpu_pool.py:_parse_requested_gpu_limit`).

**Decision.** Land PR 7.9 on the public side (so the surface is upstreamed)
but skip wiring it into the Dreamverse integration branch. Dreamverse's
`server/main.py` does not import from `router/`.

### D-5: Audio re-encode (PR 7.10) needed for streaming, deferred

**Context.** The internal streaming server's per-step path runs an audio
re-encode (`_re_encode_audio` inside `_stream_av_fmp4_events` /
`do_step_ltx2`) so each fMP4 segment ships with continuation-conditioning
audio. The whole-segment `pool.run()` path the public streaming server
currently uses doesn't need this. The PR plan defers re-encode integration
to PR 7.10 (`generate_async` / per-step streaming).

**Decision.** Land PR 7.10's `generate_async` on the public side. The
Dreamverse integration branch initially keeps using `pool.run()` (whole
segment, no re-encode); a follow-up branch swaps it to
`generate_async` + audio re-encode once that path is exercised end-to-end.

### D-6: `realtime/local_runtime.py` is *not* upstreamed

**Context.** It is the FastVideo-internal precursor to `streaming.gpu_pool`.
Upstreaming both would create two GPU pool implementations in the public
repo.

**Decision.** Don't upstream `realtime/local_runtime.py`. Dreamverse switches
to `streaming.gpu_pool.SubprocessGpuPool` on the integration branch. The
internal module can be deleted from FastVideo-internal at a follow-up.

## Open questions for user review

Each section below is a place the auto-decision could plausibly be wrong.
Please flip / annotate these in review.

### Q-1 Multi-model GPU pool (D-1)

Does any current Dreamverse production flow load multiple model ids
concurrently? If yes, we need to either (a) keep `realtime/local_runtime`
alive on the internal side until the public side gains a multi-model pool,
or (b) build the multi-model abstraction upstream as part of PR 7.6 follow-up
work.

### Q-2 Promoting LTX-2 prompt orchestration (D-2)

The locked-segments / next-segment-index / JSON-response orchestration is
LTX-2-specific. If Cosmos / Wan / Hunyuan ever grow a similar continuation
flow, we'll regret keeping the orchestration on the consumer side. Worth
promoting now?

### Q-3 Race-based provider fallback (D-3)

The sequential fallback in the public enhancer adds up to `timeout_ms` of
extra latency per failing provider before the next is tried. For Dreamverse
that's 20s. Should we land the race path now behind a `concurrency: int = 1`
knob, or wait until we have data?

### Q-4 Router upstream skip on Dreamverse branch (D-4)

We're upstreaming PR 7.9 (router) but not consuming it in the Dreamverse
integration branch. Is that right? Dreamverse currently has no router
component, so the answer is probably yes — but flagging.

### Q-5 generate_async cutover for the streaming path (D-5)

The plan leaves Dreamverse using `pool.run` (whole segment) initially.
Audio re-encode for cross-segment continuity is deferred to a follow-up.
Is that acceptable for the first switch, or does Dreamverse audio quality
regress relative to the internal path until 7.10 is wired in?

## PR-by-PR execution log

### PR 7.6 — already opened (#1257)

`will/api_7.6` rebased onto `origin/main`, with subprocess-pool robustness
review fixes pushed (boot_ok event, dead-worker detection, parallel shutdown,
reader-exit pending-job cleanup). 17/17 gpu_pool tests + 89/89 streaming
tests green at head.

### PR 7.7 — already opened (#1258)

`will/api_7.7` rebased onto the new 7.6 + LLM provider review fixes applied
locally (per-instance `retryable`, 4xx-non-retryable, json-decode wrap,
shared `_openai_compat.complete_openai_compatible`, `dataclasses.replace`
for the fallback marker). 29/29 prompt tests + 120/120 streaming tests green.
**Pending push** — the user opted to push this branch themselves.

### PR 7.8 — rebased onto new 7.7

`will/api_7.8` two commits replayed cleanly on the new 7.7. Adds
`fastvideo/entrypoints/streaming/{prompt/safety,prompt/rewrite,session_logger,
mock_server}.py` plus `test_auxiliaries.py`. 141/141 streaming tests green.

Notable gap vs internal version: the public `PromptSafetyFilter` ships one
classifier slot (`unsafe` label, single threshold) whereas the internal
version chained an NSFW filter and a hate-speech filter with marker-based
label matching. Multi-classifier composition is left to Dreamverse —
operators chain two filters explicitly. See **D-7** below.

### PR 7.9 — rebased onto new 7.8

`will/api_7.9` three commits replayed cleanly. Adds streaming router
(`router/{config,registry,main}.py`), `fastvideo router-serve` CLI
subcommand, and `test_router.py`. 151/151 streaming tests green.

Caveat: router/main.py uses the deprecated FastAPI `app.on_event("shutdown")`
hook — emits a DeprecationWarning. Migration to lifespan handlers is a
pre-merge cleanup item but not a blocker.

### PR 7.10 — rebased onto new 7.9

`will/api_7.10` three commits replayed with two trivial conflicts (line
wrap in `server.py`, redundant test in `test_cli_translation.py`). Adds
`VideoEvent` hierarchy, `VideoGenerator.generate_async`,
`default_health_check_request`, plus `test_generate_async.py` (273-line
contract test). 184/184 streaming + contract tests green.

### PR 8 — rebased onto new 7.10

`will/api_8` four commits → three (the 4th was a duplicate
`streaming.md` doc that 7.5 already shipped, dropped during rebase).
Adds `docs/design/server_contracts/{dynamo,index,openai}.md`,
`mkdocs.yml` entries, and `fastvideo/tests/contract/test_{dreamverse,
dynamo}_shape.py`. 206/206 streaming + contract tests green.

### Dreamverse `will/integrate-public-fastvideo`

Branch created from Dreamverse `master`. Single change: `pyproject.toml`
swaps `fastvideo = { path = "../FastVideo-internal", editable = true }`
to point at `../FastVideo`. Comment added linking back to this review
doc.

**Verified:** every TRACKED `from fastvideo.*` import in Dreamverse
(`server/video_generation.py` only) resolves against the public
package — except `fastvideo.layers.quantization.fp4_config.FP4Config`
(see **D-7** / Q-6 below).

**Untracked WIP** in `Dreamverse/server/{config,prompting,runtime,session}/`
imports `fastvideo.entrypoints.realtime.local_runtime` (D-6); this
branch does not migrate that WIP. The user's existing untracked work
stays untouched and will need a separate follow-up to consume
`streaming.gpu_pool.SubprocessGpuPool`.

## Test ladder (built-up to e2e per user request)

Each rung verifies the integration switch at one layer. Run from the
narrowest to the broadest before running the full e2e against real
GPU + model weights.

| # | Layer | Command | Status against the switched stack |
|---|---|---|---|
| 1 | Public FastVideo unit + contract tests | `pytest fastvideo/tests/api/ fastvideo/tests/entrypoints/streaming/ fastvideo/tests/contract/` | 358/358 passing on `will/api_8` |
| 2 | Public FastVideo FP4 lazy-import | `pytest fastvideo/tests/ops/quantization/test_fp4_config.py` | 3/3 passing |
| 3 | Dreamverse Python tests | `cd Dreamverse && uv run pytest server/tests/ -k "not stress and not benchmark and not health_endpoint"` | 73/73 passing against public FastVideo |
| 4 | Dreamverse FE unit/integration (vitest) | `cd Dreamverse/apps/web && npm test` | 54/86 passing — 32 failures are pre-existing copy-mismatches in `reducer.test.ts` etc., not caused by the switch |
| 5 | Backend HTTP smoke (Playwright) | `cd Dreamverse/apps/web && PLAYWRIGHT_SKIP_WEBSERVER=1 PLAYWRIGHT_BASE_URL=http://127.0.0.1:8009 npx playwright test e2e/backend-health.spec.ts` | 4/4 passing (5th correctly skipped because devtools-only route is off) |
| 6 | Frontend shell smoke (Playwright) | `npx playwright test e2e/frontend-shell.spec.ts` | Pending — requires Next.js dev server to be reachable; was stuck during this run, needs a clean restart |
| 7 | Full e2e preset generation | `npx playwright test e2e/preset-prompt-generation.spec.ts` | **8/8 passing** end-to-end after restart with `CUDA_VISIBLE_DEVICES=4 ENABLE_TORCH_COMPILE=0 FASTVIDEO_GPU_COUNT=1 FASTVIDEO_ENABLE_DEVTOOLS=1`. BE warmup + GPU 4 idle slot let `/readyz` flip green; the spec verifies preset → WS → backend handshake → "Generating video…" state. |

### How to reproduce e2e tier 7 from cold

```
# 1. BE — picks an idle GPU and skips torch.compile (avoids the
#    aarch64 cross-compiler bug in the conda env's triton stack).
cd ~/Dreamverse
set -a; source ~/.env; set +a
CUDA_VISIBLE_DEVICES=4 ENABLE_TORCH_COMPILE=0 \
  FASTVIDEO_ENABLE_DEVTOOLS=1 FASTVIDEO_GPU_COUNT=1 \
  uv run dreamverse-server &

# 2. Wait for /readyz (~2 min for warmup x2 segments)
until curl -fsS http://127.0.0.1:8009/readyz >/dev/null; do sleep 5; done

# 3. FE
cd ~/Dreamverse/apps/web
BACKEND_URL=http://127.0.0.1:8009 NEXT_PUBLIC_INCLUDE_DEVTOOLS=1 \
  npm run dev:devtools &

# 4. Playwright
cd ~/Dreamverse/apps/web
PLAYWRIGHT_SKIP_WEBSERVER=1 \
  PLAYWRIGHT_BASE_URL=http://127.0.0.1:5274 \
  BACKEND_URL=http://127.0.0.1:8009 \
  npx playwright test --project=chromium --reporter=list
```

### Surfaced during the e2e debug pass (logged here for follow-up)

* **`SamplingParam has no field ltx2_image_crf`** — Dreamverse's
  `server/video_generation.py:406` passes `ltx2_image_crf=0.0` to a
  `SamplingParam(...)` constructor. The internal SamplingParam (in
  `fastvideo/configs/sample/base.py`) declared this field; the public
  `fastvideo.api.sampling_param.SamplingParam` does not. Currently
  the BE logs an `ERROR` and silently drops the kwarg; warmup still
  succeeds because the field is non-load-bearing for FP4-disabled
  inference. Either re-add the field to the public schema or update
  Dreamverse to stop passing it. **D-8.**

* **`aarch64-conda-linux-gnu-cc` triton compile failure** — the conda
  env we boot from injects an ARM cross-compiler ahead of `gcc` on
  `$PATH`, so `torch._inductor`'s triton launcher fails compilation.
  Setting `ENABLE_TORCH_COMPILE=0` bypasses it. Long-term fix: clean
  the conda env's compiler shadowing or add a `CC=gcc` override in
  Dreamverse's worker bootstrap. **D-9.**

* **GPU pool starts but warmup OOMs on a shared GPU** — when
  `CUDA_VISIBLE_DEVICES` lands on a GPU another tenant is using
  (107 GiB-pegged training run on GPU 0 in this case), LTX-2 warmup
  fails with OOM. Picking an idle GPU (4-7 here) is a manual step.
  A pre-warm probe that checks free memory before booting the pool
  would prevent this. **D-10.**

* **ffmpeg fragment write `Broken pipe`** — when the WS client closes
  before the backend finishes streaming the first segment, ffmpeg
  hits `[Errno 32] Broken pipe`. Currently Dreamverse's
  `gpu_pool.handle_command` re-raises this as a session error,
  which then propagates to "User step failed". Cosmetic for now —
  swallowing pipe-broken on intentional disconnect would clean up
  the logs. **D-11.**

## Additional integration gaps surfaced during the switch

### D-7: `FP4Config` is private-only

**Context.** `Dreamverse/server/video_generation.py:271` imports
`fastvideo.layers.quantization.fp4_config.FP4Config` and assigns it to
`pipeline_config.dit_config.quant_config`. The 411-line module lives only
in `FastVideo-internal/fastvideo/layers/quantization/fp4_config.py` and
hard-imports `flashinfer` at module top — it never made the public
upstream pass. Public has `base_config.py` and `absmax_fp8.py` only.

**Decision (provisional).** Don't upstream `fp4_config.py` in this
session. Reasons:
1. It introduces a new external dependency (`flashinfer`) the public
   package has avoided so far.
2. The class hard-codes LTX-2 layer paths
   (`ltx2.blocks.{i}.attn1.to_q` etc.) — this is "LTX-2-specific FP4",
   not generic FP4. Belongs colocated with `pipelines/basic/ltx2/` if
   it goes anywhere.
3. The FP4 pre-quantize/forward op surface is the kind of thing where
   a careful review pass matters more than a bulk copy.

**What this means for the integration branch.** Dreamverse will boot
fine; only the FP4-quantized path inside `video_generation.py:283`
will fail (lazy import). For workflows that don't enable FP4
quantization, the integration is complete.

### Q-6 (review): how to land FP4Config publicly?

Two reasonable next steps:
1. **Colocate.** Move FP4 code to `fastvideo/pipelines/basic/ltx2/quantization.py`
   with `flashinfer` as an optional extra: `pip install fastvideo[fp4]`.
   Refactor `FP4QuantizeMethod` to take its layer-prefix list from a
   pipeline-config field instead of hardcoding ltx2 paths so the
   approach generalizes.
2. **Keep private.** Treat FP4 as a Dreamverse-side concern — Dreamverse
   imports `fp4_config` from the internal repo via a thin shim. Public
   FastVideo stays focused on generic surfaces. This means the
   "FastVideo-internal removable" goal is partially undone.

Recommendation: option 1 once the API refactor settles — wait until
the LTX-2 colocation step (PR 9 / 10 territory) and land FP4 there.

