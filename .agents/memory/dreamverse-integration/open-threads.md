# Open Threads — Active Follow-Ups

Live work items with priority, effort estimate, dependencies, and
recommended next action.

For why each item is open see [decisions-log.md](decisions-log.md). For
PR-level context see [pr-roadmap.md](pr-roadmap.md).

**Last updated:** 2026-05-14 (added DR-7 follow-up to remove LTX2 debug
logging env vars and replace them with per-pipeline/model state. Earlier: 2026-05-14 added DR-6 follow-up for PR #1335's Gemma
lazy-load/device-placement behavior outside compiled `forward`. Earlier: 2026-05-13 added DR-5 follow-up for PR #1333's LTX2
distilled SSIM reference refresh. Earlier: 2026-05-12 added DR-4 follow-up for PR #1330's skipped
`App websocket integration` suite. Earlier: 2026-05-05 D-20 broken-pipe root cause + fix landed on
`will/dreamverse-monorepo` @ `5eaf0a13`; added new thread D-20-CP for
cherry-picking the public-API audio routing fix to `will/ltx2_sr_port`
so it lands in PR #1288. Earlier: strategy reversal — PR #1287 CLOSED,
replaced by mega-PR #1288 on `will/ltx2_sr_port` @ `b36bdbc9` covering
the full 6-layer stack at once. See [decisions-log.md D-17](decisions-log.md#d-17).
Item D resolution gate is now #1288 merge instead of #1287; same content,
different vehicle.).

## Priority overview

| # | Pri | Item | Effort | Unblocks |
|---|---|---|---|---|
| **D-20-CP** | High | Cherry-pick `[fix] api: route LTX-2 audio kwargs through batch.extra; strict update` (`265ce1a6`) onto `will/ltx2_sr_port` so it lands in PR #1288 | 15 min | Surfaces the public-API audio-conditioning fix in the mega-PR rather than waiting for `will/dreamverse-monorepo` to fold in. Tests already pass (185/185 api). |
| **~~D-22~~** | ~~Med~~ | ~~Per-chunk timing instrumentation in `stream_fmp4` and the controller's AV relay loop~~ | ~~M~~ | ✅ **Resolved 2026-05-06** in `bade2c0a`. `av_streaming.stream_fmp4` now records `av_wav_write_ms`, `av_ffmpeg_spawn_ms`, `av_first_chunk_ms`, `av_chunk_interval_ms_{min,median,p95,max}`, `av_chunk_publish_ms_{median,p95}`, `av_chunk_read_ms_{median,p95}` into the timings dict; `gpu_pool.handle_command` prints a one-line summary per segment so they show up directly in the deploy log. **Controller-side WS-send instrumentation is the remaining unaddressed slice** — see new D-22-CTL. |
| **D-22-CTL** | Low | Controller-side AV relay timing in `apps/dreamverse/server/session/controller.py` (between media event arrival and ws_send_bytes). | S | The worker side is now fully attributed by D-22. The controller-side per-chunk WS-send overhead is the still-unmeasured slice of the 700ms gap between `worker_e2e` and `main_user_step`. Add `t_ws_send_ms` per chunk in the AV relay loop, surface as `controller_ws_send_ms_{median,p95}` in the segment summary log line. |
| **D-23** | Med | On NVENC-capable hosts (RTX 50-series / T4 / A10 / H100 PCIe), benchmark `h264_nvenc` vs `libx264` and decide whether to default `--nvenc=true` for that SKU | S benchmark + S decision | Stutter mitigation depends on hardware; B200 needs different fix path (D-24). |
| **D-24** | Low | For B200-class deploys without NVENC, prototype gen-N+1 // encode-N pipelining or FE buffer pre-fill | L | Architectural change required; benchmark suggests ~700ms can be hidden if we pipeline. Ship-blocker only when realtime stutter becomes user-visible on B200 deploys. |
| **D-8** | High | Verify `ltx2_image_crf` post-`d80c2a8` | 10 min | Confirms typed stage-override path actually flows; closes a latent silent-drop bug |
| **1** | High | Migrate `/healthz`+`/readyz`+`/status` into FastVideo `build_app` | M-L | Closes BE_FLAVOR=fastvideo FE-compatibility; closes streaming-upstream contract debt |
| **2** | High | Fix pre-existing AbsMaxFP8 test failure | S | Self-contained quantization tech debt |
| **VPO** | High | Decide `video_position_offset_sec` semantics (a vs b) | 30 min | Unblocks PR 7.6 state emission |
| **D** | 🟢 in flight | Implement `generate_async` — content shipped in mega-PR **#1288** on `will/ltx2_sr_port` @ `b36bdbc9` (was #1287, CLOSED + re-routed per [D-17](decisions-log.md#d-17)) | L | Closes Q-5/Q-9/PR-7.5 TODOs simultaneously; enables Dynamo backend; unblocks audio re-encode; enables `GpuPool.run_async()` migration (D-12-B). Resolution gate: #1288 merge. |
| **DR-1** | High | Dreamverse: create `prompting/_internal_compat.py` shim + replace local `prompt_enhancer.py` (1933 LOC) — **PR #1258 has merged (`f673423b`); now actionable** | M (~150-200 LOC shim, replace upstream wiring) | Lets Dreamverse stop carrying a 1933-LOC fork |
| **DR-2** | Med | Decide `cerebras_ifm` provider path: (a) public Literal + `CerebrasIFMProvider` shipped, OR (b) Dreamverse-side custom provider via `enhancer.register_provider(...)` | S (decision) + S-M (impl) | Resolves the cerebras_ifm gap left by PR #1258. Same item as legacy #3 below; DR-2 is the Dreamverse-side framing. |
| **DR-3** | Low | Replace Dreamverse `PromptEnhancer._run_blocking_request` manual thread/queue polling with `asyncio.to_thread` after the PR #1327 prompt-enhancer compatibility surface is retired or isolated | S | Review comment #1327 (`prompt_enhancer.py`) is valid, but deferred to avoid patching the local fork in this PR. |
| **DR-4** | Low | Investigate unskipping PR #1330's skipped public `App websocket integration` suite | S-M | Public PR #1330 has `describe.skip(...)` around 27 websocket tests while the internal equivalent suite is active with 25 tests. The 2 public-only tests cover backend unreachable / GPU workers not ready. Not blocking while skipped, but stale assertions may need safe refresh before unskip. |
| **DR-5** | Low | Regenerate LTX2-Distilled latent SSIM references under the intended neutral/distilled defaults, then remove the PR #1333 historical full-guidance pins | S-M | PR #1333 changed public LTX2 distilled defaults to neutral/distilled values, but existing LTX2 latent SSIM references appear to have been generated with historical full-guidance defaults. The current PR pins the SSIM test to old values to keep CI compatible until references are refreshed. |
| **DR-6** | Low | Decide whether `LTX2GemmaTextEncoderModel` needs a non-forward device-placement hook after lazy Gemma load | S | PR #1335 should remove the `model.device` / `model.to(...)` guard from `forward` for Dynamo/fullgraph compatibility. Non-compiled runs probably do not need it because `gemma_model` moves Gemma at first load, but a later wrapper `.to(...)` after lazy load could leave Gemma on the old device unless lifecycle placement handles it. |
| **DR-7** | Low | Remove LTX2 debug logging env-var plumbing and replace it with per-pipeline/model debug state | S-M | PR #1335 review flagged that `initialize_pipeline()` mutates process-global LTX2 debug env vars, which can leak/race across pipeline instances. Deleting only the mutation is low-risk but loses config-driven debug logging; deleting all reads without replacement would remove useful SSIM/latent drift diagnostics. |
| **3** | Med | Add `cerebras_ifm` to `PromptEnhancerConfig.provider` Literal + provider | S-M | Public-side resolution if DR-2 picks (a) |
| **4** | Med | Expose `layer_profile` on typed `engine.quantization` | M | Removes Dreamverse's `experimental["pipeline_config"]` dodge for stage profiles |
| **5** | Med | Design typed `dit_config.quant_config` carrier | L design + L impl | Removes broader `experimental["pipeline_config"]` escape hatch |
| **SBS** | Med | `SessionStore` / `BlobStore` lifecycle policy | M design | Needed in PR 7.5 design pass |
| **D-12-A** | Med | Update `GpuPool` ABC docstring: mark "API may change post-PR-7.10; experimental / server-internal" | trivial | Prevents accidental promotion of streaming-internal API to framework-level |
| **D-12-B** | Med | Replace `GpuPool.run() -> Any` with `run_async() -> AsyncIterator[VideoEvent]` in PR 7.10 cycle | M | Closes the streaming-server cancellation TODO; converges with `generate_async` |
| **D-13-A** | Med | Document `fastvideo.entrypoints.streaming.prompt.*` in user-facing docs as "streaming-server scoped"; avoid framework-level framing | trivial (docs only) | Keeps future move to `fastvideo.prompt.*` cheap |
| **D-13-B** | Low | Add optional `client_factory` parameter to `LLMProvider` for `httpx.AsyncClient` pooling | S | Only if metrics show connect/TLS overhead is meaningful |
| **D-12-C** | Low | Avoid locking `PoolAssignment.gpu_id: int` as public; rename to `worker_id` (already exists) or add `device_ids: list[int]` for topology-aware pooling | S | Future multi-GPU-per-worker refactor stays cheap |
| **6** | Low | Audio attention quantization profile + test update | S | Future audio quant exploration |
| **7** | Low | Schema parity inventory cleanup (env-driven prompt fields) | S-M | Long-term consistency |
| **8** | Low | Stale `apps/web/test-results/` dir cleanup | trivial | Cosmetic |
| **11** | Low | Promote LTX-2 prompt orchestration (locked segments, segment_prompts JSON shape, rollout id/label) to `fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` | M | Resolves Q-2 from decisions-log when a second LTX-2-style consumer appears |
| **12** | Low | When streaming server starts using `PromptSafetyFilter`, ensure operator-visible logging on `SafetyDecision.UNAVAILABLE` results | trivial | Surfaces degraded-safety state to operators (per D-14 Watch-Out item) |
| **13** | Low | When sticky session routing is needed, add `ReplicaRegistry.select(routing_key: str | None = None)` and document where `session_id` lives (WS URL/header preferred over first JSON frame) | M | Forward-compat from D-15 — keeps the door open without buffering/peeking |
| **14** | Low | At higher load, add `_bridge_session()` max-size + timeout limits OR recommend Envoy/HAProxy in front | S-M | The libraries' basic backpressure suffices for MVP; document the limit per D-15 |
| **15** | Low | If active-active multi-primary becomes a requirement, define behavior (round-robin within healthy primaries, weighted, sticky-by-key) | M | Currently `RouterConfig.__post_init__` rejects multi-primary; D-15 deferred until evidence |
| **~~Source-doc disposition~~** | ~~Med~~ | ~~Disposition of 7 untracked source docs~~ | ~~trivial~~ | ✅ **Resolved 2026-05-03** — moved into [source-archive/](source-archive/) |
| **~~9~~** | ~~Low~~ | ~~Commit-message cleanup: PR 8's 3 commits still have `[8/n] Improve API:` prefix~~ | ~~S~~ | ✅ **Resolved 2026-05-04** — bundled into the will/api_7.8 prep rebase. PR 8's 3 commits now read `[type] streaming: ...` |
| **~~10~~** | ~~Low~~ | ~~Commit-message cleanup: PR 7.8/7.9 commits have `streaming: streaming X` duplication~~ | ~~S~~ | ✅ **Resolved 2026-05-04** — bundled into the will/api_7.8 prep rebase. 3 commits dedup'd. |

---

## High priority

### D-20-CP: Cherry-pick public-API audio routing fix to `will/ltx2_sr_port`

**Why:** Commit `265ce1a6` (`[fix] api: route LTX-2 audio kwargs through batch.extra; strict update`) lives only on `will/dreamverse-monorepo` today. It touches public FastVideo surface (`fastvideo/entrypoints/video_generator.py`, `fastvideo/api/sampling_param.py`, plus a new regression test). For PR #1288 to ship a coherent public API — including the strict `SamplingParam.update()` — this commit needs to also land on `will/ltx2_sr_port`.

**Action:**

1. `git checkout will/ltx2_sr_port`
2. `git cherry-pick 265ce1a6` (clean; only touches `fastvideo/` paths that exist on both branches)
3. `pre-commit run --files fastvideo/entrypoints/video_generator.py fastvideo/api/sampling_param.py fastvideo/tests/api/test_extra_overrides_routing.py`
4. `pytest fastvideo/tests/api/ -q` (expect 185 passed)
5. `git push origin will/ltx2_sr_port` (fast-forward, no force)
6. `git checkout will/dreamverse-monorepo` (return to default working branch per runbook)

**Outcome:** PR #1288 picks up the fix automatically (its head IS `will/ltx2_sr_port`). The cherry-pick lives on both branches as separate SHAs; they'll dedupe naturally on any future rebase.

**Effort:** 15 minutes (cherry-pick + lint + test + push).

**Dependencies:** None. Tests already pass; no rebase conflicts expected.

**Files touched (same on both branches):**

- `fastvideo/entrypoints/video_generator.py`
- `fastvideo/api/sampling_param.py`
- `fastvideo/tests/api/test_extra_overrides_routing.py` (new file)

### D-8: Verify `ltx2_image_crf` typed flow post-`d80c2a8`

**Why:** Apr 26 dreamverse_review documented this field getting silently
dropped by the public `SamplingParam`. May 2 `d80c2a8` (Dreamverse)
refactored to typed `GeneratorConfig` + `preset_overrides`. Whether
`image_crf` now flows through `request.stage_overrides.refine.image_crf`
(per [design.md](design.md) mapping) or is still dropped is unverified.

**Action:**
1. Read [`Dreamverse/server/video_generation.py`](file:///home/william5lin/Dreamverse/server/video_generation.py)
   post-`d80c2a8` for `image_crf` handling
2. Trace through to FastVideo's `request.stage_overrides.refine.image_crf`
3. Confirm runtime consumption in [`fastvideo/pipelines/basic/ltx2/`](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/)

**Effort:** 10 min, no code changes.

**Outcome:** Either confirms working OR identifies bug → opens fix item.

### Item #1: Migrate `/healthz`+`/readyz`+`/status` into `build_app`

**Why:** Today
[`fastvideo.entrypoints.streaming.server.build_app`](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/server.py)
exposes only `/health` + `/v1/stream`. Dreamverse FE expects all of
`/healthz`, `/readyz`, `/status`, `/curated-presets`,
`/prompt-system-config`, devtools.

The streaming-server-upstream plan (line 84) explicitly lists
`/healthz`+`/readyz`+`/status` as part of the contract that the upstream
of `realtime/` → `streaming/` must preserve. They were deferred from
PR 7.5's MVP. `/curated-presets` and `/prompt-system-config` are
operator-side and stay in Dreamverse (FE feature-detects).

**Action:**
1. Read PR 7.5 (#1251) `build_app` to scope what's there
2. Read [`Dreamverse/server/routes/health.py`](file:///home/william5lin/Dreamverse/server/routes/health.py)
   for the route shapes Dreamverse already consumes
3. Propose route migration as commit on top of `will/api_7.5` or as
   part of PR 7.10 cycle
4. Land

**Effort:** Medium-Large (route shapes need preservation; tests).

**Dependencies:** None blocking; can land anytime.

**Files likely to touch:**
- `fastvideo/entrypoints/streaming/server.py::build_app`
- New `fastvideo/entrypoints/streaming/health.py`
- Tests in `fastvideo/tests/entrypoints/streaming/`

### Item #2: AbsMaxFP8 pre-existing test failure

**Why:** [`fastvideo/tests/ops/quantization/test_absmax_fp8.py::test_create_weights_rejects_invalid_dtype`](file:///home/william5lin/FastVideo/fastvideo/tests/ops/quantization/test_absmax_fp8.py)
fails with `AssertionError not raised`. Pre-existing on `main`; verified
NOT introduced by NVFP4 work via `git stash`.

**Action:**
1. `git log --oneline fastvideo/tests/ops/quantization/test_absmax_fp8.py`
   to find when it last passed
2. Either:
   - Restore the assert in `AbsMaxFP8LinearMethod.create_weights` if
     intentional behavior was lost
   - Drop the test if assert is no longer correct
3. Verify

**Effort:** Small.

**Dependencies:** None.

### Item VPO: `video_position_offset_sec` semantics

**Why:** Per
[`fastvideo/pipelines/basic/ltx2/continuation.py`](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/continuation.py),
`LTX2ContinuationState.video_position_offset_sec` exists as a state
field. Two valid interpretations:

- **(a) Persistent across segments** — accumulating time offset for long
  sessions; useful for time-coherent audio chaining.
- **(b) Per-segment hint that rides on the carrier** — runtime
  overwrites every time; field is harmless redundancy.

Dreamverse computes `prefix_sec = float(audio_extra) / 24.0` per segment
in `apply_audio` and currently does NOT persist it on
`ContinuationState`. Field's docstring leans toward (b).

**Decision deadline:** before PR 7.6 starts emitting/consuming the
field (PR 7.6 branch is ready, not yet PR'd).

**Action:**
1. Confirm field's intended semantics with audio team
2. If (a): document the accumulation rule explicitly + add tests
3. If (b): leave docstring as-is + add test confirming overwrite

**Effort:** 30 min discussion + small implementation.

### Item D: Implement `generate_async` (PR 7.10)

**Why:** Highest leverage. Closes:

- D-5 / Q-5: audio re-encode for cross-segment continuity
- Q-9: Dynamo progress passthrough (deferred)
- PR 7.5's mid-segment cancellation TODO
- Unblocks Dynamo native backend integration
- **D-12-B**: enables `GpuPool.run() -> run_async() -> AsyncIterator[VideoEvent]` migration

**Action:** See [streaming-server.md](streaming-server.md) "PR 7.10 — the
unlock PR" section for scoping.

**Effort:** Large.

**Dependencies:** Best after PR 7.6 lands (gpu_pool upstream).

**Files:**
- `fastvideo/entrypoints/video_generator.py` — add `generate_async`,
  refactor `generate_video` as wrapper
- `fastvideo/api/results.py` — add `VideoEvent`/`VideoProgressEvent`/
  `VideoPartialEvent`/`VideoFinalEvent`
- `fastvideo/entrypoints/streaming/server.py` — consume `generate_async`,
  remove TODO markers
- `fastvideo/entrypoints/streaming/gpu_pool.py` — add `run_async()`
  forwarding events from worker to caller
- New `fastvideo/tests/entrypoints/test_generate_async.py`
- New `fastvideo/tests/contract/test_dynamo_shape.py` (already in PR 8)

### Item DR-1: Dreamverse — replace local `prompt_enhancer.py` with public + compat shim

**Why:** Today Dreamverse carries `Dreamverse/server/prompt_enhancer.py`
(1933 LOC) — a local copy/derivative of the FastVideo-internal version.
After PR #1258 merges, Dreamverse should switch to the public
`fastvideo.entrypoints.streaming.prompt.PromptEnhancer` and delete most
of the local module.

**Migration shape:**

1. **Create** `Dreamverse/server/prompting/_internal_compat.py` (~150-200 LOC):
   - Wraps public `PromptEnhancer.enhance()` → returns `EnhanceResult` shape Dreamverse expects
   - Wraps public `PromptEnhancer.auto_extend()` — JSON-parses `LLMResponse.content` into `{"next_prompt": "..."}`
   - Wraps public `PromptEnhancer.rewrite()` — JSON-parses into `{"segment_prompts": [...]}` with lenient fallback for malformed JSON
   - Layers locked-segment + rollout_id + rollout_label metadata back on top
2. **Update** `Dreamverse/server/runtime.py + main.py` — replace `from prompt_enhancer import PromptEnhancer` with `from prompting._internal_compat import PromptEnhancer`
3. **Delete most of** `Dreamverse/server/prompt_enhancer.py` (1933 LOC). Keep only the bits that don't have a public equivalent:
   - Race-based parallel fallback (`_run_provider_race`) — Dreamverse-specific tail-latency optimization
   - `cerebras_ifm` provider — pending DR-2 decision
   - Multi-classifier prompt safety (NSFW + hate-speech chained) — public ships single classifier
4. **Tests** — verify Dreamverse session controllers still see the expected response shapes through the shim

**Effort:** Medium (~150-200 LOC shim + replace upstream wiring + delete 1700+ LOC local module + test fixture updates).

**Dependencies:**
- PR #1258 must merge first (publishes `fastvideo.entrypoints.streaming.prompt.*`)
- DR-2 informs the cerebras_ifm path

**Files:**
- New: `Dreamverse/server/prompting/_internal_compat.py`
- Modified: `Dreamverse/server/runtime.py`, `Dreamverse/server/main.py`
- Mostly deleted: `Dreamverse/server/prompt_enhancer.py`

---

## Medium priority

### Item DR-2: Decide `cerebras_ifm` provider path

**Why:** Public PR #1258's `PromptEnhancerConfig.provider` is
`Literal["cerebras", "groq"]`. Internal supports `"cerebras_ifm"` (the
Cerebras IFM API endpoint with different auth). Dreamverse needs
`cerebras_ifm` working post-migration.

Two options:

| Option | Approach | Pros | Cons |
|---|---|---|---|
| **(a) Public** | Add `"cerebras_ifm"` to public Literal + ship `CerebrasIFMProvider` in `fastvideo/entrypoints/streaming/prompt/providers/cerebras_ifm.py` | Discoverable; users with IFM access can use typed config | Adds ~50 LOC + Literal extension to public surface |
| **(b) Dreamverse-side** | Implement `CerebrasIFMProvider` Dreamverse-side as a custom `LLMProvider`, register via `enhancer.register_provider(CerebrasIFMProvider())` | Zero public surface change; private endpoint stays private | Slightly more boilerplate Dreamverse-side; not surfaced to non-Dreamverse users |

**Recommendation:** Option (b) is more contained. Option (a) is more
discoverable. Default to (b) unless there's a third-party user who needs
IFM access. The Dreamverse-side PR carrying DR-1 is the natural place to
make this decision.

**Effort:** Small (decision) + Small-Medium (implementation).

**Dependencies:** DR-1 (compat shim creation).

### Item #3: `cerebras_ifm` provider in public Literal

**Why:** Same item as DR-2 from the public-side framing. If DR-2 picks
option (a), this is the implementation. If DR-2 picks option (b), this
item is closed without implementation.

**Action:** See DR-2.

**Effort:** S-M.

### Item #4: Expose `layer_profile` on typed `engine.quantization`

**Why:** Today `transformer_quant: "NVFP4"` always constructs
`NVFP4Config()` with default `layer_profile="refine"`. Dreamverse
dodges via `experimental["pipeline_config"]`.

**Action:**
1. Add `transformer_quant_layer_profile: str | None = None` to
   `QuantizationConfig` in [`schema.py`](file:///home/william5lin/FastVideo/fastvideo/api/schema.py)
2. Thread through [`compat.py`](file:///home/william5lin/FastVideo/fastvideo/api/compat.py)
3. Update `_apply_transformer_quant` in
   [`fastvideo_args.py`](file:///home/william5lin/FastVideo/fastvideo/fastvideo_args.py)
   to pass profile
4. Update Dreamverse to drop the `experimental["pipeline_config"]`
   dodge in favor of typed knob
5. Tests in [`test_typed_quant_flow.py`](file:///home/william5lin/FastVideo/fastvideo/tests/api/test_typed_quant_flow.py)

**Effort:** Medium.

**Files:** schema.py, compat.py, fastvideo_args.py, test_typed_quant_flow.py,
+ Dreamverse/server/video_generation.py.

### Item #5: Typed `dit_config.quant_config` carrier

**Why:** The `experimental["pipeline_config"]` escape hatch in
Dreamverse should eventually become a typed field. Design TBD.

**Action:** Heaviest design work. Should consult Oracle.

**Effort:** Large design + Large implementation.

**Dependencies:** #4 should land first; this is the "final form" of #4.

### Item SBS: `SessionStore` / `BlobStore` lifecycle policy

**Why:** PR 7's in-memory implementations have no eviction, no TTL, no
automatic blob cleanup on state replacement. Documented as per-deployment
policy decision.

When PR 7.5/7.6 land the live consumer, who owns:

- bounded session capacity (LRU? TTL? hard max?)
- blob `drop()` chained when state is replaced
- session expiry on websocket disconnect

**Recommendation:** streaming server's session manager. Worth stating
explicitly in PR 7.5's design.

**Effort:** Medium design + small implementation.

### Item D-12-A: Update `GpuPool` ABC docstring — mark experimental

**Why:** Per D-12 in [decisions-log.md](decisions-log.md), `GpuPool`
should be documented as "API may change post-PR-7.10; experimental /
server-internal" to prevent accidental promotion of streaming-internal
API to framework-level. PR #1257 merged without this caveat.

**Action:** Edit
[`fastvideo/entrypoints/streaming/gpu_pool.py`](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/gpu_pool.py)
class docstring on `GpuPool` ABC. Add a note: "API may change post-PR-7.10
when run_async() lands; treat as server-internal for now."

**Effort:** Trivial.

**Dependencies:** None.

### Item D-12-B: Replace `GpuPool.run() -> Any` with `run_async() -> AsyncIterator[VideoEvent]`

**Why:** Per D-12, this is the canonical evolution post-PR-7.10. Closes
the streaming server's cancellation TODO and converges the streaming +
OpenAI + Dynamo consumers on a single async API.

**Action:** As part of PR 7.10 cycle:
1. Add `GpuPool.run_async(session_id, request) -> AsyncIterator[VideoEvent]`
2. Worker forwards events through `result_queue` with type discriminator
3. Streaming server replaces `await pool.run(...)` with `async for event in pool.run_async(...)`
4. Sync `run()` becomes a thin compat wrapper that collects events and returns the final
5. Cancellation propagates: client disconnect → `asyncio.CancelledError` → worker stops mid-step

**Effort:** Medium. Adds ~50-100 LOC + tests.

**Dependencies:** Item D (PR 7.10 — `generate_async` on `VideoGenerator`).

### Item D-13-A: Document `streaming/prompt/*` as streaming-scoped

**Why:** Per D-13 in [decisions-log.md](decisions-log.md), the prompt
enhancer is currently scoped to streaming-server use even though the
abstraction is general. Phrase user-facing docs as "streaming-server
prompt enhancement" to keep future move to `fastvideo.prompt.*` cheap.

**Action:** When PR 12 (docs migration) is written, the prompt enhancer
section should:
- Be titled "Streaming Server Prompt Enhancement", not "Prompt API"
- Note the 3 fixed operations (`enhance` / `auto_extend` / `rewrite`) are
  shaped by LTX-2 streaming session needs
- Note that consumers wanting custom prompt operations can use
  `provider.complete()` directly with their own LLMRequest
- Avoid `from fastvideo import LLMProvider` exports until a second
  consumer exists

**Effort:** Trivial (docs only).

**Dependencies:** PR 12 (docs migration).

---

## Low priority

### Item D-13-B: Optional `client_factory` parameter for `httpx.AsyncClient` pooling

**Why:** Today `_openai_compat.py` instantiates `httpx.AsyncClient` per
call (no connection pooling). Reviewer flagged inefficient. Team chose
simplicity for the expected scale (~6-10 enhancer calls per LTX-2
session). If real-world metrics show connect/TLS overhead is meaningful,
add an optional `client_factory: Callable[[], httpx.AsyncClient] | None`
parameter to providers so they can share a pool.

**Action:** Only when metrics justify. Add `client_factory=None` parameter
to `CerebrasProvider` / `GroqProvider` constructors and pass through to
`complete_openai_compatible()`. Default to current per-call behavior.

**Effort:** Small.

**Dependencies:** None blocking; only act on real perf data.

### Item DR-4: Unskip PR #1330 `App websocket integration` suite safely

**Why:** Public PR #1330 currently wraps `App websocket integration` in
`describe.skip(...)`, so the 27-test websocket integration suite does not
run. The internal repo has the equivalent suite active via `describe(...)`
with 25 tests. The two public-only cases cover backend unreachable and GPU
workers not ready readiness/reachability behavior.

**Action:** Later, investigate whether the public suite can be unskipped and
refresh any stale assertions without changing the intended websocket contract.
Because the suite is skipped today, this is not blocking the current stack or
current PR #1330 review.

**Effort:** Small-Medium.

**Dependencies:** Best handled when someone can run the frontend integration
stack end-to-end and compare the public assertions against the internal active
suite.

### Item DR-6: Gemma lazy-load device placement outside `forward`

**Why:** PR #1335 review flagged this pattern in
`fastvideo/models/encoders/gemma.py::LTX2GemmaTextEncoderModel.forward`:

```py
if model.device != target_device:
    model.to(device=target_device)
```

The immediate concern is the compiled/Dynamo path: `model.device` and
`model.to(...)` inside `forward` can introduce non-tensor/device parsing work
that fullgraph tracing should not see. Removing the guard from `forward` is the
right PR #1335 review fix.

For non-compiled execution, the guard is mostly defensive rather than required:
`gemma_model` already moves the lazily loaded HF Gemma model to the wrapper's
current parameter device on first load. The remaining edge case is a lifecycle
sequence where Gemma is loaded, then the parent wrapper is later moved to a
different device; in that case Gemma could stay behind unless placement is
handled outside `forward`.

**Action:** After PR #1335 review is unblocked, decide whether FastVideo needs a
small lifecycle hook/helper for this class so lazy Gemma is moved whenever the
wrapper/device placement changes. If yes, implement it outside `forward`; if no,
document that Gemma must be loaded after final device placement.

**Effort:** Small.

**Dependencies:** Not blocking PR #1335 if the forward-path guard is removed and
the normal load path keeps placing Gemma on the wrapper's current device.

### Item DR-7: Remove LTX2 debug logging env-var plumbing

**Why:** PR #1335 review flagged that
`fastvideo/pipelines/basic/ltx2/ltx2_pipeline.py::initialize_pipeline()` sets
and pops LTX2 debug env vars such as `LTX2_PIPELINE_DEBUG_LOG`,
`LTX2_PIPELINE_DEBUG_PATH`, `LTX2_DEBUG_DETAIL`, and
`LTX2_PIPELINE_DEBUG_DETAIL_PATH`. Those env vars are process-global, so one
pipeline instance can enable, overwrite, or clear debug behavior for another
pipeline instance running in the same process.

Deleting only the `initialize_pipeline()` env mutation is low risk for normal
generation, but it would stop config-driven debug logging unless replaced.
Deleting all env-var reads without replacement is riskier because these logs are
useful for SSIM/latent drift diagnosis and may be used by local debug scripts.

**Action:** Replace LTX2 debug env-var plumbing with per-pipeline/model debug
state. Use pipeline/model config for construction-time hooks and a
`ForwardContext`/`ForwardBatch`-style carrier for forward-time logging. Keep
external env-var compatibility only if there is a documented operator workflow
that still needs it.

**Effort:** Small-Medium.

**Dependencies:** Not blocking PR #1335 if the immediate fix is limited to
removing process-global mutation from pipeline initialization while preserving
existing externally supplied env-var reads.

### Item D-12-C: Avoid locking `PoolAssignment.gpu_id: int` as public

**Why:** Today `PoolAssignment` exposes `gpu_id: int`, assuming
one-GPU-per-worker. Future topology-aware pooling may need
`device_ids: list[int]` (one worker = group of GPUs running internal
`MultiprocExecutor`). Don't freeze the int field as public API.

**Action:**
- Treat `gpu_id` as a current-impl detail; prefer `worker_id` (already
  exists, is stable identifier)
- When a worker actually spans multiple GPUs, add
  `PoolAssignment.device_ids: list[int]` and let `gpu_id` be `device_ids[0]`
  for backward compat
- Or rename to `gpu_id` → `device_id` with deprecation alias

**Effort:** Small (1 field rename + alias).

**Dependencies:** Driven by an actual future "one worker = many GPUs" use case. Don't act preemptively.

### Item #6: Audio attention quantization profile

**Why:** Today audio attn and FFN are bf16. If an audio-quant profile
is added to `NVFP4Config.fp4_layers`, update
[`test_basic_av_block_propagates_quant_config_to_all_children`](file:///home/william5lin/FastVideo/fastvideo/tests/ops/quantization/test_nvfp4_ltx2_wiring.py).

**Effort:** Small (one test + one config field).

### Item #7: Schema parity inventory cleanup

**Why:** A few internal-only fields are not exposed publicly:

- `PROMPT_HTTP_TIMEOUT_MS`
- `PROMPT_INITIAL_STAGE_TIMEOUT_MS`
- `PROMPT_TEMPERATURE`
- `PROMPT_MAX_COMPLETION_TOKENS`
- `PROMPT_AUTO_SLEEP_MS`
- `PROMPT_AUTO_TIMEOUT_MS`
- curated-presets file paths

These flow via env vars on `dreamverse-server` today. If
`fastvideo serve --config` becomes the canonical entrypoint, they need
typed homes.

**Effort:** Small-Medium.

### Item #8: Stale `apps/web/test-results/` directory

**Why:** Cosmetic. `.gitignore` entry hides it from `git status`, but
the dir has stale `.last-run.json` (45 bytes) from a prior Playwright
run.

**Action:** `rm -rf apps/web/test-results` whenever convenient.

**Effort:** Trivial.

### ~~Item #9~~ + ~~#10~~: Commit-message cleanups — ✅ Resolved 2026-05-04

Both items resolved during the `will/api_7.8` prep rebase. A targeted
conditional script (`/tmp/opencode/cleanup_subjects_v2.sh` — only amends
when text actually changes) ran across 33 commits, modified 6:

- **#9 fix**: extended the regex from `\[\d+\.\d+/n\]` to
  `\[\d+(\.\d+)?/n\]` so single-digit prefixes match. PR 8's 3 commits
  now read `[type] streaming: ...` instead of `[type] [8/n] Improve API: ...`.
- **#10 fix**: added second substitution `streaming: streaming X` →
  `streaming: X`. PR 7.8 / 7.9 commits no longer have the duplication.

The conditional check skipped pre-commit-hook flakiness on no-op amends
(unlike the earlier first attempt). All affected commits verified clean
post-rebase.

### Item #11: Promote LTX-2 prompt orchestration to public (when 2nd consumer exists)

**Why:** Per Q-2 in [decisions-log.md](decisions-log.md) and D-13's
"missing alternative", the LTX-2-specific orchestration (locked
segments, segment_prompts JSON shape, rollout id/label, lenient JSON
parsing) currently stays Dreamverse-side per DR-1. If a second
LTX-2-style consumer appears (e.g. another video model with multi-segment
continuation needing the same prompt orchestration), promote this layer
to `fastvideo.entrypoints.streaming.prompt.ltx2_orchestration`.

**Action:** Wait for a second consumer to materialize. Until then, the
orchestration stays in Dreamverse's `_internal_compat.py` shim (DR-1).

**Effort:** Medium when triggered.

**Dependencies:** A second consumer.

---

## Recommended pull order

If you have unbounded time and want to maximize forward progress:

1. **D-8 verify** (10 min) — eliminates uncertainty
2. **D-12-A docstring** (trivial) — caveat the GpuPool API publicly
3. **Item #2 AbsMaxFP8** (S) — clears tech debt
4. **Item VPO video_position_offset_sec** (30 min) — unblocks PR 7.10 (since 7.6 has merged, this is now scoped to whatever consumer first reads the field)
5. **DR-1 + DR-2 Dreamverse migration** (M) — **now unblocked since PR #1258 merged**; replaces 1700+ LOC of local fork
6. **Item #4 layer_profile** (M) — closes Dreamverse quant escape hatch
7. **Item #1 build_app routes** (M-L) — closes FE-compat
8. **Item D generate_async** (L) — unlock PR; brings along D-12-B (run_async) + closes Q-5/Q-9/PR-7.5 TODOs
9. **Item #5 typed quant_config carrier** (L+L) — final form
10. **Items #6/#7/#8 + D-12-C/D-13-A/D-13-B + #11** — cleanup polish (#9, #10 resolved 2026-05-04)

If you have a specific user goal (e.g. "ship `BE_FLAVOR=fastvideo`
flavor end-to-end"), that goal dictates the order — read this list as a
menu, not a prescription.

---

## Verification gates per item

When implementing any item above, evidence required:

| Phase | Check |
|---|---|
| Build | `lsp_diagnostics` clean on changed files |
| Test | new + relevant existing tests pass; output captured |
| Manual QA | actually run the affected feature end-to-end (per AGENTS.md MANUAL_QA_MANDATE) |
| Regression | full `fastvideo/tests/api/` + `contract/` + relevant SSIM (if NVFP4 touch) |

For NVFP4 touches: re-run `test_nvfp4_ltx2_wiring.py` +
`test_typed_quant_flow.py` (CPU) + ideally a flashinfer-enabled path
test (manual, not in CI).

For Dreamverse-side items (DR-1, DR-2): re-run
`Dreamverse/apps/web/npx playwright test e2e/preset-prompt-generation.spec.ts`
end-to-end against the live BE+FE — this is the contract test that
exercises the prompt enhancer through a real session.
