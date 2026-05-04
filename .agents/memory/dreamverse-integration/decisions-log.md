# Decisions Log — D + Q Resolutions

Cross-doc consolidated decision log. Each entry: ID, source doc,
question/decision, rationale, current status.

For implementation status see [pr-roadmap.md](pr-roadmap.md). For
follow-up actions see [open-threads.md](open-threads.md).

**Last updated:** 2026-05-03.

## Status legend

- ✅ **Resolved** — decision made and implementation complete (or no implementation needed)
- 🟡 **Deferred** — decision made, implementation deferred to a known PR
- 🔴 **Open** — needs decision

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
