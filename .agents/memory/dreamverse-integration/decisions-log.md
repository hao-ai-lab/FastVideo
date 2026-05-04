# Decisions Log — D + Q Resolutions

Cross-doc consolidated decision log. Each entry: ID, source doc,
question/decision, rationale, current status.

For implementation status see [pr-roadmap.md](pr-roadmap.md). For
follow-up actions see [open-threads.md](open-threads.md).

**Last updated:** 2026-05-04 (added D-12 — GpuPool layer separation, Oracle review post-#1257-merge; added D-13 — prompt enhancer / LLMProvider abstraction shape, Oracle review pre-#1258-merge; added D-14 — streaming auxiliaries cohesion + concrete-vs-Protocol scoping, Oracle review during #1284 review cycle).

## Status legend

- ✅ **Resolved** — decision made and implementation complete (or no implementation needed)
- 🟡 **Deferred** — decision made, implementation deferred to a known PR
- 🔴 **Open** — needs decision

## Post-merge architecture decisions

### D-12: `GpuPool` layer separation — keep distinct from `VideoGenerator`

**Status:** ✅ Resolved (interim) + 🟡 Deferred long-term shape to PR 7.10.
**Source:** Oracle review on 2026-05-04, post-PR-#1257 merge.

**Question:** Should `fastvideo.entrypoints.streaming.GpuPool` (PR #1257) be
folded into `fastvideo.entrypoints.video_generator.VideoGenerator`, or kept
separate? Three alternatives were evaluated:

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — `VideoGenerator` (single inference call) and `GpuPool` (multi-session orchestration) stay separate | ✅ Correct as **interim** |
| B | `VideoGenerator` absorbs the pool's role (`from_pretrained_pool`, `acquire/release/run`) | ❌ **Wrong layer.** Conflates execution with serving scheduler. |
| C | `GpuPool` becomes a thin **session-aware async executor** over PR 7.10's `generate_async` | ✅ Correct **long-term destination** |

**Decision:** Alt A as interim; evolve toward Alt C once PR 7.10 lands
`generate_async`. Do NOT pursue Alt B.

**Rationale:**

- `VideoGenerator` is a library handle — "execute one request, possibly
  across ranks via `MultiprocExecutor`/`RayDistributedExecutor`."
- `GpuPool` is serving infrastructure — "schedule N concurrent sessions
  across N independent replicas, with sticky session-to-GPU affinity for
  cache locality."
- These are different layers driven by different consumers (a Python
  script doing `gen.generate(req)` vs. a WebSocket server with sticky
  sessions). Folding them muddies both surfaces.

**Key finding — `MultiprocExecutor` and `SubprocessGpuPool` are orthogonal,
not redundant:**

| Layer | Job | Granularity |
|---|---|---|
| `MultiprocExecutor` (`fastvideo/worker/`) | TP/SP shard ONE inference call across N GPU ranks | per-call |
| `streaming_generator.py` (existing real-time path) | Per-frame streaming via `MultiprocExecutor.submit_step`/`get_result` | per-step within one generator |
| `SubprocessGpuPool` (`entrypoints/streaming/`, PR #1257) | Serve N concurrent sessions on N replicas, sticky-bound | per-session |

Both spawn subprocesses because **CUDA contexts demand process boundaries**,
not because they solve the same problem. Sharing low-level lifecycle
utilities (process spawn, queue plumbing, shutdown) is a future refactor;
unifying the abstractions is wrong.

**Sticky binding stays in the pool, NOT in `VideoGenerator`:** sticky
session-to-GPU affinity is a serving policy driven by LTX-2's per-GPU
continuation cache (last-9-decoded-frames + audio-latents). Different
consumers want different policies — stateless OpenAI HTTP wants
per-request leasing; LTX-2 streaming wants sticky affinity; per-frame
real-time streaming wants a continuous queue. Keeping policy in the pool
keeps `VideoGenerator` policy-free.

**Specific risks flagged in PR #1257 (already merged):**

| Risk | Mitigation (when relevant) |
|---|---|
| `GpuPool.run() -> Any` is sync — fine for whole-segment dispatch, blocks on cancellation | Replace with `run_async() -> AsyncIterator[VideoEvent]` in PR 7.10 cycle (`generate_async` makes this trivial) |
| `PoolAssignment.gpu_id: int` assumes one-GPU-per-worker | Don't lock as public API. Future may need `device_ids: list[int]` for topology-aware pooling (one worker = group of GPUs running internal `MultiprocExecutor`) |
| `GpuPool` could be documented as the canonical FastVideo serving API | Mark as **experimental / server-internal** in docstring until PR 7.10 lands. Don't include in user-facing API docs yet |
| Memory: N processes = N model replicas (~10-50 GB each) | Expected for concurrent serving with crash isolation. CUDA IPC weight sharing loses isolation; CPU-shared-memory loading helps host RAM not device. Real scalable path is topology-aware pooling later. |

**Action items (carried into post-7.10 cycle):**

- [ ] Update `GpuPool` ABC docstring to note "API may change post-PR-7.10"
- [ ] Plan to replace `run()` with `run_async() -> AsyncIterator[VideoEvent]` in PR 7.10 cycle
- [ ] Don't promote `gpu_id: int` to public API; revisit shape post-7.10
- [ ] Consider clarifying field naming (e.g. `worker_id` is the stable identifier; `gpu_id` is current-impl detail)
- [ ] When opening 7.10's PR, have it consume `generate_async` from `GpuPool.run_async` end-to-end

**Open thread it touches:** PR 7.10 (`open-threads.md` item D — generate_async)
unblocks Alt C and is the natural place to land the API shape change.

### D-14: Streaming auxiliaries (PR #1284) — cohesion + concrete-vs-Protocol scoping

**Status:** ✅ Resolved (interim). Two polish items applied during review; one
operational caveat tracked.
**Source:** Oracle review on 2026-05-04, during PR #1284 review cycle.

**Question:** Is PR #1284's bundle of 4 streaming-server auxiliary modules
(`prompt/safety.py`, `prompt/rewrite.py`, `session_logger.py`,
`mock_server.py`) correctly scoped? Should `mock_server` live in production
module path? Should `PromptSafetyFilter` be a Protocol? Should the bundle
have been split into 4 PRs?

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — single PR, 4 modules under `streaming/`, mock_server in production path, concrete safety filter | ✅ **Keep** |
| B | Split into 4 separate PRs | ❌ Process overhead, not architectural improvement |
| C | Move `mock_server.py` into `tests/` | ❌ Would reduce discoverability + install-time usability of `python -m fastvideo.entrypoints.streaming.mock_server` |
| D | Move `session_logger.py` to `streaming/observability/` (or top-level `fastvideo/observability/`) | ❌ Premature — currently session-shaped + streaming-specific; promote when a non-streaming consumer appears |
| E | Convert `PromptSafetyFilter` to Protocol (like `LLMProvider`) | ❌ Premature abstraction — only one classifier exists; small duck-typed surface preserves future Protocol introduction without breaking the concrete |
| F | Convert `MockGenerator` to Protocol | ❌ Same — small duck-typed surface; no second mock generator exists |

**Decision:** Alt A — keep current shape. Apply two polish items from
Oracle's review before merge.

**Rationale:**

- "Streaming-server auxiliaries" is cohesive enough at 730 LOC with
  isolated modules + tests. Each module has independent code path but
  shared deployment context (the streaming server boots them all).
- `mock_server.py` in production path is a strength: reuses
  `build_app()` for protocol parity. Hiding it under `tests/` would lose
  `python -m fastvideo.entrypoints.streaming.mock_server` CLI access for
  FE devs.
- Concrete `PromptSafetyFilter` matches "ship what we have, abstract
  later" pattern. Internal had multi-classifier composition; public
  ships single + leaves chaining as a Dreamverse-side concern (per D-2).
- Same pattern for `MockGenerator`: small duck-typed `_GeneratorLike`
  surface lets a second mock implementation drop in without inheritance.
- `threading.Lock` (not `asyncio.Lock`) in `session_logger.py` is
  correct — writes come from real encoder/control threads via
  `run_in_executor`, not from coroutines directly. `asyncio.Lock` would
  be the wrong primitive for cross-thread concurrency.

**Pre-merge polishes applied (per Oracle):**

| Polish | What | Why |
|---|---|---|
| 1 | Removed `RewriteOptions.user_system_prompt_override` | Inert public field — was declared but never threaded through to `enhancer.rewrite()`. Shipping unused public options is more likely to bite than any structural choice. Re-add when actually wired through. |
| 2 | Sanitized `session_id` filename in `session_logger.SessionLogger._get_file()` | Defense-in-depth: today session_id is server-generated UUID, but a future code path that accepts client-supplied ids would otherwise allow path traversal via `../`. Added `_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")` + sub before `os.path.join`. |

**Operational caveat tracked (not a code change):**

- `SafetyDecision.UNAVAILABLE` is treated as `ALLOW` by callers — a
  policy choice that's correct for an opt-in safety filter, but
  callers should log loudly so operators know the filter is degraded.
  Tracked as open-threads.md item #12.

**Pre-merge review feedback (4 of 4 resolved on the GitHub PR):**

| # | File:Line | Severity | Issue | Fix applied |
|---|---|---|---|---|
| 1 | `session_logger.py:57` | High | `log()` race vs `close()` — `KeyError` on `_locks[session_id]` | Atomic capture in `_get_file()`; master `_registry_lock`; `with lock, contextlib.suppress(ValueError):` |
| 2 | `rewrite.py:71` | Medium | `re.compile()` in hot path | Module-level `_LEADING_MARKER_RE`, top-level `import re` |
| 3 | `safety.py:105` | Medium | `_ensure_loaded()` race on concurrent fastText load | `_load_lock = threading.Lock()` + double-check pattern |
| 4 | `pyproject.toml:145` | Medium | `streaming` extra missing `prompt-safety` | Added to aggregator |

All 4 review threads marked resolved via GraphQL `resolveReviewThread`.

**Action items (deferred):**

- [ ] Track `SafetyDecision.UNAVAILABLE` log loudness in
  open-threads.md item #12 — when streaming server starts using the
  safety filter, ensure operator-visible logging on `UNAVAILABLE`
  results
- [ ] If a second safety classifier appears (Perspective API, Detoxify,
  custom rules), promote `PromptSafetyFilter` to a Protocol — same
  pattern as `LLMProvider` per D-13
- [ ] If a second mock generator appears (different frame patterns,
  different latency models), promote `MockGenerator` to a Protocol

**Open thread it touches:** PR #1284 itself; future safety-classifier
Protocol promotion; future observability module extraction.

### D-13: Prompt enhancer / `LLMProvider` abstraction shape — keep streaming-scoped

**Status:** ✅ Resolved (interim) + 🟡 Three deferred polishes after metrics or 2nd consumer.
**Source:** Oracle review on 2026-05-04, pre-PR-#1258-merge.

**Question:** Is PR #1258's `fastvideo.entrypoints.streaming.prompt.*` module
correctly designed? Should it be (a) Protocol-based vs ABC, (b) under
`streaming/` vs top-level `fastvideo.prompt.*`, (c) closed 3-op enum vs
open `complete()` API?

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — `streaming/prompt/*`, Protocol provider, fixed 3 ops, lazy `httpx`, per-call `AsyncClient` | ✅ **Keep** |
| B | Move to top-level `fastvideo.prompt.*` (decouple from streaming) | ❌ **Premature.** No second consumer exists yet. |
| C | Convert `LLMProvider` Protocol → ABC with default impls + retry classification | ❌ **Wrong direction.** Biases extension toward OpenAI shape; `_openai_compat.py` already factors that as helper not inheritance. |

**Decision:** Alt A as interim. Promote to Alt B only when a second
non-streaming consumer (OpenAI server, batch generation, tooling) actually
needs the prompt enhancer. Don't pursue Alt C.

**Rationale:**

- Public contract is tiny — `name: str` + `async complete(LLMRequest) -> LLMResponse`. ABC adds zero value.
- `_openai_compat.py` is the right place for shared logic — helper, not base class. Anthropic / local / custom providers stay first-class.
- The 3 ops (enhance / auto_extend / rewrite) are LTX-2 streaming concepts. `auto_extend` (continue prompt sequence) and `rewrite` (multi-line alternatives) come directly from session UX. Calling this "the FastVideo prompt API" misrepresents that.

**Specific risks flagged in PR #1258 (already merged-pending review):**

| Risk | Mitigation (when relevant) |
|---|---|
| API publicity — calling this "the FastVideo prompt API" before a second consumer exists | Document module as "streaming-server prompt enhancement" in user-facing docs; keep it nested under `entrypoints/streaming/` |
| `httpx.AsyncClient` per-call (no connection pooling) | Acceptable for ~6-10 calls per LTX-2 session; LLM latency dominates. Add optional `client_factory` parameter LATER if metrics show connect overhead is meaningful. |
| 3 fixed operations could constrain future generic use | Closed enum is right for application-level orchestration. Future generic consumers should either call `provider.complete()` directly, or get a thin separate enhancer that shares the provider/fallback machinery. |
| `register_provider(priority=-1)` semantics rely on Python's negative-index `list.insert` | Cosmetic concern; docstring is clear. Could be tightened to explicit branch later. |
| `runtime_checkable` Protocol with `name: str` instance attribute — static type checkers may miss missing `name` | Acceptable; runtime check via `isinstance(p, LLMProvider)` works for plugin discovery. |

**Action items (deferred):**

- [ ] Document `fastvideo.entrypoints.streaming.prompt.*` as streaming-scoped in user-facing docs (PR 12 docs migration); avoid promoting as framework-level
- [ ] Add optional `client_factory` parameter to providers when metrics justify pooling
- [ ] Plan future move to `fastvideo.prompt.*` (with import shim) when second non-streaming consumer materializes
- [ ] Track Q-2 reactivation: promote LTX-2 prompt orchestration (locked segments, segment-prompts JSON parsing) to public `fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` when a second LTX-2-style consumer appears

**Open thread it touches:** Dreamverse migration (open-threads.md DR-1)
will be the first real test of the public surface. Lessons learned there
inform whether Alt B becomes feasible.

## D-decisions (from `dreamverse_review.md`, Apr 26)

### D-1: Realtime runtime → streaming GpuPool migration shape

**Status:** ✅ Resolved.

Internal `RealtimeRuntimeConfig` had a multi-model registry +
flattened sampling defaults. Public `SubprocessGpuPool` is single-model
+ uses per-request `SamplingConfig`.

**Decision:** Drop multi-model registry on integration branch (not used
in production). Construct `GeneratorConfig` for chosen model and pass to
`SubprocessGpuPool`. Move sampling defaults to a server-side
`default_request: GenerationRequest` template.

**Risk:** Migration branch surfaces missing-model errors if a flow
silently relied on registry to swap models per-session. Integration
tests exercise at least one segment per supported model id before
merging.

### D-2: PR 7.7 prompt enhancer API surface narrower than internal

**Status:** ✅ Resolved.

Public `PromptEnhancer.enhance/auto_extend/rewrite` returns
`LLMResponse(content, provider, model, latency_ms, fallback_used)`.
Internal returns `EnhanceResult(prompt, fallback_used, error, ...)` /
`RewriteResult(prompts, ..., rollout_id, rollout_label, ...)`.

**Decision:** Adapt at the call site via
`Dreamverse/server/prompting/_internal_compat.py` shim. Locked-segment /
next-segment-index plumbing stays Dreamverse-side. Public stays minimal
and provider-agnostic.

**Open question (Q-2):** Promote LTX-2-specific orchestration into
`fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` once a
second consumer appears. Logged for future review.

### D-3: Multi-stage provider race vs. sequential fallback

**Status:** ✅ Resolved (public stays sequential).

Internal enhancer runs all providers in a stage in parallel
(`_run_provider_race`). Public enhancer runs sequentially with
retryable-error fallback.

**Decision:** Public stays sequential for PR 7.7. Race is a
Dreamverse-specific tail-latency optimization that depends on parallel
API budgets.

**Risk / Q-3:** First-segment latency on Dreamverse may regress
slightly when Cerebras has a bad minute (sequential waits 20s before
trying Groq). If real production concern, add public
`concurrency: int = 1` knob behind a race path — but only after measuring.

### D-4: Skip PR 7.9 router for the integration branch

**Status:** ✅ Resolved.

Internal stack ships `router/main.py` for multi-replica load balancing.
Dreamverse deployment uses single replica per region.

**Decision:** Land PR 7.9 publicly (upstream the surface). Skip wiring
into Dreamverse integration branch. Dreamverse's `server/main.py` does
not import from `router/`.

### D-5: Audio re-encode (PR 7.10) needed for streaming, deferred

**Status:** 🟡 Deferred to PR 7.10.

Internal streaming server's per-step path runs `_re_encode_audio` inside
`_stream_av_fmp4_events` so each fMP4 segment ships with
continuation-conditioning audio. Whole-segment `pool.run()` path doesn't
need this.

**Decision:** Land PR 7.10's `generate_async` publicly. Dreamverse
integration branch initially keeps using `pool.run()` (whole segment, no
re-encode). Follow-up branch swaps to `generate_async` + audio re-encode.

**Open question (Q-5):** Acceptable for first switch, or does
Dreamverse audio quality regress vs. internal until 7.10 wires in?

### D-6: `realtime/local_runtime.py` is NOT upstreamed

**Status:** ✅ Resolved.

It was the FastVideo-internal precursor to `streaming.gpu_pool`.
Upstreaming both would create two GPU pool implementations in public.

**Decision:** Don't upstream `realtime/local_runtime.py`. Dreamverse
switches to `streaming.gpu_pool.SubprocessGpuPool` on integration
branch. Internal module can be deleted at follow-up.

### D-7 / Q-6: `FP4Config` is private-only

**Status:** ✅ **Resolved May 2.**

April 26: `Dreamverse/server/video_generation.py:271` imported
`fastvideo.layers.quantization.fp4_config.FP4Config` from
FastVideo-internal only. The 411-line module hard-imported `flashinfer`.

**Two options at the time:**

1. Colocate publicly with `flashinfer` as optional extra
   `pip install fastvideo[fp4]`; refactor `FP4QuantizeMethod` to take
   layer-prefix list from a pipeline-config field instead of hardcoding
   ltx2 paths.
2. Keep private — Dreamverse imports from internal via thin shim.

**Recommendation at the time:** option 1 once API refactor settles.

**Resolution:** May 2 work chose option 1.
- `365a66c7` upstreamed FP4Config with lazy `flashinfer` import in
  loader helper (no public hard-dep)
- `94c983a2` renamed FP4 → NVFP4 to disambiguate from MX-FP4 / OCP-FP4
- `42b30bf9` wired through `fastvideo.layers.quantization`

See [quantization.md](quantization.md) for full details.

### D-8: `ltx2_image_crf` silently dropped by public schema

**Status:** 🔴 **Unverified post-`d80c2a8`.**

April 26: Dreamverse's `server/video_generation.py:406` passed
`ltx2_image_crf=0.0` to `SamplingParam(...)`. Public
`fastvideo.api.sampling_param.SamplingParam` did NOT have this field;
the BE logged ERROR and silently dropped the kwarg.

**Migration target** (per [design.md](design.md) compatibility map):
`request.stage_overrides.refine.image_crf`.

**Resolution status:** `d80c2a8` (May 2) refactored
`server/video_generation.py` to use typed `GeneratorConfig` +
`preset_overrides["refine"]`. Whether this PR routed `image_crf`
through the typed `stage_overrides` path or left it silently dropped is
unverified. See [open-threads.md](open-threads.md).

### D-9: `aarch64-conda-linux-gnu-cc` triton compile failure

**Status:** ✅ Resolved (operational).

Conda env injected an ARM cross-compiler ahead of `gcc` on `$PATH`, so
`torch._inductor`'s triton launcher failed compilation. Setting
`ENABLE_TORCH_COMPILE=0` bypasses it.

**Long-term fix:** clean conda env's compiler shadowing or add
`CC=gcc` override in Dreamverse's worker bootstrap.

### D-10: Warmup OOM on shared GPU

**Status:** ✅ Resolved (operational).

When `CUDA_VISIBLE_DEVICES` lands on a GPU another tenant uses, LTX-2
warmup fails with OOM. Picking an idle GPU (4-7 in test setup) is a
manual step.

**Improvement:** pre-warm probe that checks free memory before booting
the pool would prevent this.

### D-11: ffmpeg fragment write `Broken pipe`

**Status:** ✅ Resolved (cosmetic).

When WS client closes before backend finishes streaming first segment,
ffmpeg hits `[Errno 32] Broken pipe`. Currently propagates to
"User step failed". Cosmetic — swallowing pipe-broken on intentional
disconnect would clean up logs.

## Q-questions (from `streaming-server-upstream-plan.md`, Apr 17)

### Q-1: Router placement (in-repo or separate package)

**Status:** ✅ Resolved (in-tree).

**Recommendation at the time:** separate package `fastvideo-router/` or
`fastvideo/contrib/router/`; defer final call to PR 7.9.

**Resolution:** PR 7.9 implementation places router in-tree at
`fastvideo/entrypoints/streaming/router/`.

### Q-2: Session ID authority

**Status:** ✅ Resolved (server-generated).

**Recommendation:** server-generated UUID; accept externally provided
session ID only for resume flows.

### Q-3: Torch compile kwargs typing (opaque vs full vs hybrid)

**Status:** ✅ Resolved (hybrid).

**Recommendation:** hybrid — type the common four (`backend`,
`fullgraph`, `mode`, `dynamic`) + allow `extras: dict[str, Any]`.

**Resolution:** PR 6 + NVFP4 `221cb20a` shipped exactly this hybrid.

### Q-4: Prompt safety / fasttext dependency

**Status:** ✅ Resolved (optional extra).

**Recommendation:** ship as optional extra `pip install fastvideo[prompt-safety]`.

**Resolution:** PR 7.8 implements as optional extra.

### Q-5: Audio-specific tensor payloads in continuation

**Status:** ✅ Resolved (typed `LTX2ContinuationState`).

`ltx2_audio_clean_latent`, `ltx2_audio_denoise_mask`,
`ltx2_audio_latents` not in pre-refactor public schema.

**Recommendation:** classify as opaque fields inside
`LTX2ContinuationState.payload`, not top-level sampling fields.

**Resolution:** PR 7's typed `LTX2ContinuationState` lifts these into
typed fields (see [cross-repo-surfaces.md](cross-repo-surfaces.md)
field mapping table).

### Q-6: Dynamo subpackage home

**Status:** ✅ Resolved (lives in Dynamo repo).

**Resolution:** No Dynamo code in FastVideo. Full backend package
(handler, adapter, registration, health check) owned by Dynamo repo at
`components/src/dynamo/fastvideo/`, same pattern as vllm/sglang.
FastVideo only guarantees the public API contract.

### Q-7 (was Q-6 in dreamverse_review): How to land FP4Config publicly

**Status:** ✅ Resolved May 2 — option 1 (colocate publicly).

See D-7 above.

### Q-8: Disaggregation readiness contract test

**Status:** 🟡 Recommended; not yet shipped.

PR ai-dynamo/dynamo#7544 is aggregated-only. `ContinuationState` hybrid
already supports future prefill/decode split.

**Recommendation:** PR 7.10 explicitly validate `ContinuationState`
survives round-trip through Dynamo-style RPC (pickle or JSON), even
though Dynamo isn't using it today. Cheap regression guard.

### Q-9: Dynamo progress/status passthrough

**Status:** 🟡 Deferred until Dynamo clarifies.

`NvVideosResponse` has `status` and `progress` fields.

**Recommendation:** PR 7.10 stays aggregated-final-only to match PR
#7544 shape; revisit after Dynamo clarifies their streaming/progress
semantics.

## Cross-doc questions still 🔴 OPEN

These need decisions; tracked also in [open-threads.md](open-threads.md):

| ID | Question | Source | Why it matters |
|---|---|---|---|
| **D-8** | Did `d80c2a8` route `ltx2_image_crf` correctly, or is it still silently dropped? | dreamverse_review | Latent silent-drop bug; FP4-disabled paths may degrade |
| **VPO** | `video_position_offset_sec` — persistent accumulation (a) vs per-segment hint (b) | dreamverse_integration | Needs decision before PR 7.6 emits state |
| **SBS** | `SessionStore` / `BlobStore` lifecycle (TTL/eviction/blob-drop on state replacement) | dreamverse_integration | Needs decision in PR 7.5 design pass |
| **#1** | Migrate `/healthz`+`/readyz`+`/status` into FastVideo `build_app` | streaming-upstream-plan + handoff | Closes BE_FLAVOR=fastvideo FE-compatibility |
| **#3** | Add `cerebras_ifm` to public `PromptEnhancerConfig.provider` Literal | handoff | Internal supports it; public schema doesn't |
| **#4** | Expose `layer_profile` on typed `engine.quantization` | handoff | Removes Dreamverse's `experimental["pipeline_config"]` dodge |
| **#5** | Typed `dit_config.quant_config` carrier (design TBD) | handoff | Eliminates the `experimental["pipeline_config"]` escape hatch entirely |
