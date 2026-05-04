# Open Threads — Active Follow-Ups

Live work items with priority, effort estimate, dependencies, and
recommended next action.

For why each item is open see [decisions-log.md](decisions-log.md). For
PR-level context see [pr-roadmap.md](pr-roadmap.md).

**Last updated:** 2026-05-03.

## Priority overview

| # | Pri | Item | Effort | Unblocks |
|---|---|---|---|---|
| **D-8** | High | Verify `ltx2_image_crf` post-`d80c2a8` | 10 min | Confirms typed stage-override path actually flows; closes a latent silent-drop bug |
| **1** | High | Migrate `/healthz`+`/readyz`+`/status` into FastVideo `build_app` | M-L | Closes BE_FLAVOR=fastvideo FE-compatibility; closes streaming-upstream contract debt |
| **2** | High | Fix pre-existing AbsMaxFP8 test failure | S | Self-contained quantization tech debt |
| **VPO** | High | Decide `video_position_offset_sec` semantics (a vs b) | 30 min | Unblocks PR 7.6 state emission |
| **D** | High | Implement `generate_async` (PR 7.10) | L | Closes Q-5/Q-9/PR-7.5 TODOs simultaneously; enables Dynamo backend; unblocks audio re-encode |
| **3** | Med | Add `cerebras_ifm` to PromptEnhancerConfig provider Literal + provider | S-M | Closes provider-API gap PR 7.7 left for Dreamverse |
| **4** | Med | Expose `layer_profile` on typed `engine.quantization` | M | Removes Dreamverse's `experimental["pipeline_config"]` dodge for stage profiles |
| **5** | Med | Design typed `dit_config.quant_config` carrier | L design + L impl | Removes broader `experimental["pipeline_config"]` escape hatch |
| **SBS** | Med | `SessionStore` / `BlobStore` lifecycle policy | M design | Needed in PR 7.5 design pass |
| **6** | Low | Audio attention quantization profile + test update | S | Future audio quant exploration |
| **7** | Low | Schema parity inventory cleanup (env-driven prompt fields) | S-M | Long-term consistency |
| **8** | Low | Stale `apps/web/test-results/` dir cleanup | trivial | Cosmetic |
| **~~Source-doc disposition~~** | ~~Med~~ | ~~Disposition of 7 untracked source docs~~ | ~~trivial~~ | ✅ **Resolved 2026-05-03** — moved into [source-archive/](source-archive/) |

---

## High priority

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
- New `fastvideo/tests/entrypoints/test_generate_async.py`
- New `fastvideo/tests/contract/test_dynamo_shape.py` (already in PR 8)

---

## Medium priority

### Item #3: `cerebras_ifm` provider in public Literal

**Why:** Public typed `PromptEnhancerConfig.provider` is
`Literal["cerebras", "groq"]`. Internal supports `"cerebras_ifm"`
(`config.py:143`). The `streaming_demo.yaml` defaults to `"cerebras"`.

For agents that need `cerebras_ifm`, the `dreamverse-server` flavor
respects `FASTVIDEO_PROMPT_PROVIDER` env var (legacy path);
`fastvideo serve --config` does not currently expose it.

**Action:**
1. Extend Literal in [`fastvideo/api/schema.py`](file:///home/william5lin/FastVideo/fastvideo/api/schema.py)
2. Implement provider in
   `fastvideo/entrypoints/streaming/prompt/providers/cerebras_ifm.py`
3. Tests + register

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

---

## Low priority

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

---

## Recommended pull order

If you have unbounded time and want to maximize forward progress:

1. **D-8 verify** (10 min) — eliminates uncertainty
2. **Item #2 AbsMaxFP8** (S) — clears tech debt
3. **Item VPO video_position_offset_sec** (30 min) — unblocks PR 7.6
4. **Item #3 cerebras_ifm** (S-M) — closes Dreamverse provider gap
5. **Item #4 layer_profile** (M) — closes Dreamverse quant escape hatch
6. **Item #1 build_app routes** (M-L) — closes FE-compat
7. **Item D generate_async** (L) — unlock PR
8. **Item #5 typed quant_config carrier** (L+L) — final form
9. **Items #6/#7/#8** — cleanup

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
