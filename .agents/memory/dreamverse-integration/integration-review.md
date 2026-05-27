# Integration Review — Drift Audit + Path Forward

> # ⚠️ DEPRECATED — superseded by [integration-plan.md](integration-plan.md)
>
> This document recommended **Option D** (Dreamverse stays a separate repo;
> generic backend merges into FastVideo). On 2026-05-05 the team chose
> **Option B+** instead (Dreamverse FE + product server move into FastVideo
> as `apps/dreamverse/`; generic backend stays at
> `fastvideo.entrypoints.streaming.*` per Option D's principle).
> See [decisions-log.md D-18](decisions-log.md#d-18) for the strategy
> reversal rationale and [integration-plan.md](integration-plan.md) for the
> executable migration plan.
>
> **What's still authoritative in this file:**
> - **Part 1 — Drift audit** (the 17-row drift summary table). The drift
>   findings remain valid; the migration plan in `integration-plan.md`
>   folds them into specific phases.
> - **OSS precedent citations** (vLLM, BentoML, Ray Serve, TGI+ChatUI,
>   Transformers.js, ComfyUI, AUTOMATIC1111). Reused in `integration-plan.md`.
>
> **What's superseded:**
> - **Part 2 — Recommendation (Option D)**. Replaced by Option B+ in the
>   new plan. Read `integration-plan.md` for the current decision.
> - **Part 3 — Action items**. Replaced by the phased migration plan.
>
> Kept in tree for historical reference and audit trail. Do not delete.

**Last updated:** 2026-05-05 (deprecated header added).

**Scope:** FastVideo public `will/ltx2_sr_port` at the requested audit
anchor `b36bdbc9`; Dreamverse `will/integrate-public-fastvideo` at
`ec8ef92`; FastVideo-internal `will/rebase-nbv` as read-only comparison.

**Memory-dir context:** the current integration memory snapshot tracks the
same public mega-PR lineage as `will/ltx2_sr_port`, with PRs #1257,
#1258, #1284, and #1286 already merged, #1287 closed, and #1288 open as
the consolidated landing vehicle for LTX-2 SR runtime, NVFP4,
`generate_async`, Dynamo contract, and memory-dir cleanup. Source:
[memory index](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/README.md#L8-L19)
and [D-17](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L19-L45).

**Bottom line:** Zero core typed API drift — typed construction, typed
continuation state, NVFP4 wiring, and Dynamo-facing async events are either
already public or in #1288. **Real drift remains on the realtime-runtime
contract surface (`/healthz` / `/readyz` / `/status` routes per
[cross-repo-surfaces.md](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L74-L88))
and on operational/product edges**: stale Dreamverse docs/scripts, a
1933-LOC Dreamverse prompt-enhancer fork, two unresolved per-session
fields (`ltx2_image_crf` D-8, `video_position_offset_sec` VPO), one
missing example config, and two internal-only utilities whose product
relevance is not yet proven.

---

## Part 1 — Drift audit

### Methodology

1. **Compared three repositories and branches.**
   - FastVideo public: `/home/william5lin/FastVideo`, branch
     `will/ltx2_sr_port`.
   - Dreamverse: `/home/william5lin/Dreamverse`, branch
     `will/integrate-public-fastvideo`.
   - FastVideo-internal: `/home/william5lin/FastVideo-internal`, branch
     `will/rebase-nbv`.
   - Canonical repo paths are listed in the integration memory index:
     [repo paths](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/README.md#L72-L79).

2. **Scoped the audit to the ultimate integration goal.**
   - Dreamverse should depend on public `fastvideo`, not
     `FastVideo-internal`.
   - FastVideo should own the reusable backend subset that Dreamverse
     currently needs from internal: streaming runtime, GPU pool, router,
     prompt enhancer, NVFP4, continuation state, and typed generation.
   - Dynamo should consume FastVideo through typed public Python APIs, not
     through private modules.
   - The three Dreamverse surfaces are documented as pipeline construction,
     realtime runtime, and continuation state:
     [cross-repo surfaces](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L13-L20).

3. **Separated intentional refactor from drift.**
   - A path rename is not drift if the public branch contains the same
     responsibility under the typed design.
   - A deleted file is not drift if the public design intentionally
     consolidated it.
   - A private alias is not drift if the public schema exposes a typed
     replacement with contract tests.
   - This matches the typed-public-boundary rule in
     [design.md](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L24-L43).

4. **Used memory docs for rationale and worktree files for concrete proof.**
   - API schema and public exports:
     [schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L68-L85),
     [api exports](file:///home/william5lin/FastVideo/fastvideo/api/__init__.py#L49-L109).
   - Streaming server current routes:
     [build_app](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/server.py#L88-L160).
   - Dreamverse dependency state:
     [pyproject server extra](file:///home/william5lin/Dreamverse/pyproject.toml#L17-L22),
     [uv lock editable source](file:///home/william5lin/Dreamverse/uv.lock#L716-L722).
   - Contract tests:
     [Dreamverse shape](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L1-L26),
     [Dynamo shape](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dynamo_shape.py#L1-L19),
     [generate_async](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_generate_async.py#L1-L7).

5. **Did not treat product-only Dreamverse behavior as FastVideo drift.**
   - Dreamverse keeps a local product server and Next.js UI today:
     [README baseline](file:///home/william5lin/Dreamverse/README.md#L5-L16).
   - Product-only routes, curated presets, devtools, and frontend-specific
     behavior belong in Dreamverse unless a second non-Dreamverse consumer
     needs them.

6. **Risk scale used below.**
   - **P0:** blocks Dreamverse from running without FastVideo-internal.
   - **P1:** blocks clean `BE_FLAVOR=fastvideo` or Dynamo/public API use.
   - **P2:** reproducibility or maintenance drag.
   - **P3:** optional parity or future memory/perf improvement.

### Findings: zero core typed API drift

The public branch is aligned with the goal on the **core typed API
surface** (construction, request, continuation state, async events).
The table below lists items that look like drift only if compared by
path name or legacy field name. They are intentional public refactors
or already guarded by tests. **Note:** the realtime-runtime _contract_
surface (FE-required health routes) is a separate matter — see "real
drift items" §4 below.

| Investigated item | Drift? | Evidence | Conclusion |
|---|---:|---|---|
| Dreamverse surface 1: pipeline construction | No | Dreamverse migrated from flat kwargs to typed `GeneratorConfig` at `d80c2a8`; mapping documented in [cross-repo surfaces](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L22-L47). | Stable public surface exists. |
| Dreamverse surface 2: realtime runtime | No on architecture; some route work remains | Runtime migration target is public `streaming/`, not internal `realtime/`: [streaming upstream](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L14-L31). | Rename/refactor is intentional. |
| Dreamverse surface 3: continuation state | No | Public typed `ContinuationState` plus LTX-2 state mapping are documented in [cross-repo surfaces](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L108-L146). | Public state is a superset of Dreamverse's data carrier. |
| Internal `fastvideo/entrypoints/realtime/` | No | Public design chooses parallel `fastvideo/entrypoints/streaming/`: [layout decision](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L55-L60), [current build_app](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/server.py#L88-L160). | Intentional rename plus typed-config rewrite. |
| Internal `configs/sample/` presets | No | Public PR 4 intentionally deleted `configs/sample/` and moved defaults to per-family presets plus `fastvideo/api/sampling_param.py`: [design](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L194-L205), [PR roadmap](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L21-L29). | Intentional consolidation. |
| LTX-2 pipeline presets | No | Public target is model-owned named presets and per-family colocation: [design](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L175-L205). | Public layout matches design. |
| Internal `use_fp4_linear` flag | No | Public typed quant carrier is `engine.quantization.transformer_quant`; schema field exists in [schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L68-L85), compat resolves it in [compat.py](file:///home/william5lin/FastVideo/fastvideo/api/compat.py#L267-L279). | Replaced by typed NVFP4 surface. |
| Public-only `transformer_quant` field | No | Public `FastVideoArgs` pins typed quant to `dit_config.quant_config`: [fastvideo_args](file:///home/william5lin/FastVideo/fastvideo/fastvideo_args.py#L220-L228), [apply logic](file:///home/william5lin/FastVideo/fastvideo/fastvideo_args.py#L260-L279). | Public superset, not drift. |
| Internal `config_model_path` | No | Public typed home is `generator.pipeline.components.config_root`: [design mapping](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L261-L269), [compat mapping](file:///home/william5lin/FastVideo/fastvideo/api/compat.py#L295-L299). | Alias is covered. |
| Internal flat video request fields | No | Public `GenerationRequest` nests `inputs`, `sampling`, `runtime`, `output`, `state`, `extensions`: [schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L193-L204). Internal legacy fields live in internal protocol at [protocol.py](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/openai/protocol.py#L64-L82). | Intentional request refactor. |
| Dreamverse typed init kwargs | No | Contract test asserts current Dreamverse load kwargs all land on typed fields, not `experimental`: [test_dreamverse_shape](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L44-L135). | Guard in place. |
| Dreamverse request path | No | Contract test asserts request fields round-trip through typed `GenerationRequest`: [test_dreamverse_shape](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L153-L197). | Guard in place. |
| Dynamo native backend shape | No | FastVideo's only obligation is stable typed Python API: [cross-repo contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L188-L210). | Dynamo should stay out of FastVideo. |
| `generate_async` event API | No | API exists in [video_generator](file:///home/william5lin/FastVideo/fastvideo/entrypoints/video_generator.py#L264-L332), event types exist in [results.py](file:///home/william5lin/FastVideo/fastvideo/api/results.py#L109-L164). | #1288 covers the async contract. |
| Dynamo request mapping | No | Authoritative source is the contract test [test_dynamo_shape](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dynamo_shape.py#L90-L175) which asserts `req.prompt`, `req.sampling.{height,width,num_frames,fps,num_inference_steps,guidance_scale,seed,negative_prompt}`, and `req.inputs.{image_path,video_path}` against the actual nested [`GenerationRequest` schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L193-L204). The `streaming-server.md` Dynamo mapping table mis-cites a `prompt -> sampling.prompt` path that no longer exists; the test is correct, the doc is stale and tracked for refresh. | Guard in place; companion doc needs minor refresh. |
| Public API exports | No | `VideoEvent`, `VideoResult`, and typed schema classes are exported from [fastvideo.api](file:///home/william5lin/FastVideo/fastvideo/api/__init__.py#L49-L109). | Integration imports resolve. |
| FastVideo-internal FP4/NVFP4 paths | No | Public NVFP4 files and roles are documented in [quantization](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/quantization.md#L24-L35); actual `NVFP4Config` documents lazy FlashInfer and public naming in [nvfp4_config.py](file:///home/william5lin/FastVideo/fastvideo/layers/quantization/nvfp4_config.py#L1-L19). | Public is typed superset. |
| AbsMaxFP8 refactor | No for Dreamverse | Public quant registry includes `AbsMaxFP8` and `NVFP4`: [quantization init](file:///home/william5lin/FastVideo/fastvideo/layers/quantization/__init__.py#L1-L8). AbsMaxFP8 failure is tracked as separate tech debt: [open threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L100-L117). | Not Dreamverse blocker. |
| Internal realtime API regression test | No | Public contract tests replace it: [Dreamverse contract](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L1-L26), [Dynamo contract](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dynamo_shape.py#L1-L19), [generate_async tests](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_generate_async.py#L91-L230). | Better scoped guards exist. |
| Dreamverse dependency declaration | No | `server` extra declares `fastvideo>=0.1.7`: [pyproject](file:///home/william5lin/Dreamverse/pyproject.toml#L17-L22). Dev lock resolves editable public `../FastVideo`: [uv.lock](file:///home/william5lin/Dreamverse/uv.lock#L716-L722), [package source](file:///home/william5lin/Dreamverse/uv.lock#L777-L780). | Dependency is already switched in metadata/lock. |

#### Core conclusion for the zero-typed-drift section

The public typed API no longer needs to mirror `FastVideo-internal` file
paths. The correct test is whether Dreamverse and Dynamo can express their
needs through public typed objects and public entrypoints. On that test,
the **typed core** is covered (construction, request, continuation,
async events). The **runtime contract** still has health-route gaps —
see real drift §4. On the **typed core**:

- `GeneratorConfig` and `GenerationRequest` cover construction and calls:
  [schema surface](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L45-L72).
- `ServeConfig.streaming` covers the server envelope:
  [schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L244-L279).
- `generate_async` covers streaming, OpenAI, and Dynamo on one substrate:
  [video_generator](file:///home/william5lin/FastVideo/fastvideo/entrypoints/video_generator.py#L264-L332).
- Contract tests now encode the cross-repo shapes:
  [Dreamverse](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L70-L214),
  [Dynamo](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dynamo_shape.py#L170-L331),
  [async events](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_generate_async.py#L91-L273).

### Findings: real drift items requiring action

#### 1. Dreamverse README and bootstrap script still point at FastVideo-internal

- **Priority:** P0 for a clean public-dependency story.
- **Effort:** Small.
- **Owner:** Dreamverse repo.
- **Evidence:** Dreamverse metadata already points at public FastVideo:
  [pyproject](file:///home/william5lin/Dreamverse/pyproject.toml#L17-L22),
  [uv source](file:///home/william5lin/Dreamverse/pyproject.toml#L54-L61),
  [uv.lock](file:///home/william5lin/Dreamverse/uv.lock#L716-L722).
- **Drift:** README still tells users that `uv` resolves from
  `../FastVideo-internal` and that bootstrap expects `../FastVideo-internal`:
  [README](file:///home/william5lin/Dreamverse/README.md#L76-L109).
- **Drift:** bootstrap script still defaults to cloning the private repo and
  verifying imports from that clone:
  [script defaults](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L7-L11),
  [script clone flow](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L33-L63),
  [script import assertion](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L66-L88).
- **Action:** Replace private-fork bootstrap with public FastVideo bootstrap
  or delete the bootstrap once PyPI publication is the default path.
- **Do not overreach:** no FastVideo code change required.

#### 2. Dreamverse carries a 1933-line prompt-enhancer fork

- **Priority:** P1.
- **Effort:** Medium.
- **Owner:** Dreamverse repo, after public prompt enhancer is available.
- **Evidence:** Dreamverse local fork starts at
  [server/prompt_enhancer.py](file:///home/william5lin/Dreamverse/server/prompt_enhancer.py#L1-L80).
- **Public replacement:** FastVideo now has provider-agnostic
  `PromptEnhancer` with `enhance`, `auto_extend`, `rewrite`, and
  `register_provider`:
  [public enhancer](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/prompt/enhancer.py#L66-L142).
- **Provider extension point:** custom providers implement `LLMProvider`:
  [provider protocol](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/prompt/providers/base.py#L63-L75).
- **Tracking:** DR-1 in open threads already defines the compat-shim shape:
  [DR-1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L174-L206).
- **Action:** Replace the fork with a small Dreamverse shim that adapts
  public `LLMResponse` to Dreamverse's product response objects and keeps
  only product-only extras.
- **Do not overreach:** do not merge Dreamverse's full prompt product layer
  into FastVideo unless a second consumer needs the same semantics.

#### 3. `cerebras_ifm` provider is unresolved

- **Priority:** P1 if Dreamverse needs IFM in production; P2 otherwise.
- **Effort:** Small decision plus small/medium implementation.
- **Owner:** Team decision; implementation either Dreamverse-side or public.
- **Public state:** `PromptEnhancerConfig.provider` is currently
  `Literal["cerebras", "groq"]`:
  [schema](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L229-L235).
- **Design note:** public Literal excludes `cerebras_ifm` today:
  [streaming-server D-3](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L61-L100).
- **Tracking:** DR-2 already frames the public-vs-Dreamverse decision:
  [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229).
- **Recommended default:** implement IFM as a Dreamverse-side custom provider
  registered through `enhancer.register_provider(...)` unless there is a
  non-Dreamverse public user.

#### 4. `/healthz`, `/readyz`, and `/status` are not in public `build_app`

- **Priority:** P1 for `BE_FLAVOR=fastvideo` frontend compatibility.
- **Effort:** Medium/Large because route shapes need tests.
- **Owner:** FastVideo public.
- **Public current state:** `build_app` exposes `GET /health` and
  `WS /v1/stream`:
  [server.py](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/server.py#L126-L160).
- **Dreamverse expected state:** Dreamverse exposes `GET /healthz`,
  `GET /readyz`, and `GET /status`:
  [routes/health.py](file:///home/william5lin/Dreamverse/server/routes/health.py#L34-L79).
- **Tracking:** open item #1 documents route ownership and files likely to
  touch:
  [open threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99).
- **Design note:** `/curated-presets`, `/prompt-system-config`, and devtools
  stay Dreamverse-side, with feature detection:
  [streaming route contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L232-L259).

#### 5. `fastvideo/models/layerwise_offload.py` exists only internally

- **Priority:** P3 unless memory-tight Dreamverse deployments require it.
- **Effort:** Medium if adopted; low if documented as deferred.
- **Owner:** FastVideo public only if a concrete deployment needs it.
- **Internal evidence:** internal file defines async layerwise CPU offload
  manager with pinned CPU memory and prefetch stream:
  [layerwise_offload.py](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L1-L20),
  [prefetch path](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L127-L180).
- **Public state:** no equivalent public file was identified in this audit.
- **Action:** defer unless Dreamverse or another public deployment hits a
  memory ceiling that cannot be handled by existing offload knobs.
- **Decision rule:** if adopted, port as a generic offload utility with
  tests; do not make it Dreamverse-specific.

#### 6. Standalone LTX-2 upsampler CLI exists only internally

- **Priority:** P2 for reproducibility; P3 for product runtime.
- **Effort:** Small/Medium after scope decision.
- **Owner:** FastVideo public if standalone upsampling is a supported user
  workflow.
- **Internal utility:** `upscale_video_file(...)` reads an existing video,
  prepares frame count/resolution, loads VAE + upsampler, and writes an mp4:
  [upsample.py](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/upsample.py#L120-L180),
  [write tail](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/upsample.py#L181-L202).
- **Internal CLI:** `fastvideo upsample` wrapper exists internally:
  [cli/upsample.py](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L15-L35),
  [CLI args](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L48-L130).
- **Public related functionality:** LTX-2 SR refine stage covers the
  in-pipeline latent upsample/refine path:
  [ltx2_refine.py](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L1-L22),
  [upsample stage](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L116-L180).
- **Action:** decide whether standalone file-to-file upsampling is a public
  CLI promise or whether the SR refine stage is sufficient.

#### 7. Reproducible streaming demo config lives only in Dreamverse

- **Priority:** P2.
- **Effort:** Small.
- **Owner:** FastVideo public.
- **Evidence:** canonical demo config currently lives at
  [Dreamverse/serve_configs/streaming_demo.yaml](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L1-L12).
- **Config content:** it documents LTX-2 distilled model, one GPU,
  no offload, compile settings, NVFP4, refine overrides, default request,
  and streaming settings:
  [generator block](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L31-L87),
  [streaming block](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L108-L149).
- **Memory pointer:** design.md already treats this as the canonical
  example:
  [design YAML example](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L235-L239).
- **Action:** copy/adapt it into
  `examples/serving/streaming_demo.yaml` with public-safe comments.

#### 8. LTX-2 stage equivalence is a verification gap, not proven drift

- **Priority:** P2.
- **Effort:** Medium if parity checks are added; small if only manual audit.
- **Owner:** FastVideo public.
- **Public state:** model-specific LTX-2 stages are colocated under
  `fastvideo/pipelines/basic/ltx2/stages/`, consistent with the target
  layout in [design.md](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/design.md#L175-L205).
- **Example public stage:** `ltx2_refine.py` explicitly says it is a
  public-side port of the internal stage and describes the three-stage SR
  flow:
  [ltx2_refine.py](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L1-L22).
- **Action:** verify behavior for the six internal `ltx2_*` stage files
  against public colocated stages. If a mismatch is found, file it as a
  real drift item with a failing parity test.

### Findings: deferred / accepted residual

These items should not block the public-dependency transition.

1. **StepVideo residual.**
   - Dreamverse's model registry is LTX-2/LTX-2.3 only:
     [Dreamverse config](file:///home/william5lin/Dreamverse/server/config.py#L28-L45).
   - Internal local tests even stub StepVideo modules to keep LTX registry
     tests focused:
     [test_ltx2_registry.py](file:///home/william5lin/FastVideo-internal/tests/local_tests/test_ltx2_registry.py#L38-L61).
   - Conclusion: accepted low-priority deferral unless Dreamverse adds a
     StepVideo model.

2. **Internal debug-only `FastVideoArgs` fields.**
   - Internal debug fields exist around `FastVideoArgs` and stage/model sums:
     [internal grep source](file:///home/william5lin/FastVideo-internal/fastvideo/fastvideo_args.py#L200-L203).
   - They are debug-only and not a public user-facing integration surface.
   - Conclusion: low-priority; do not add to public schema unless a debug
     workflow requires them.

3. **Private request aliases.**
   - Public request schema is nested and strict:
     [GenerationRequest](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L193-L204).
   - Legacy OpenAI flat fields are compatibility input, not the canonical
     public API:
     [internal protocol](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/openai/protocol.py#L64-L82).
   - Conclusion: no action beyond current compat tests.

4. **`experimental["pipeline_config"]` escape hatch.**
   - Dreamverse currently uses an explicit in-memory quant config because
     typed `transformer_quant: "NVFP4"` does not expose `layer_profile`:
     [quantization](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/quantization.md#L86-L97).
   - Open thread #4 tracks `layer_profile`:
     [open threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L244-L260).
   - Conclusion: defer broader typed carrier design; add `layer_profile`
     first if Dreamverse needs base/refine profile selection.

5. **Router sticky routing and active-active semantics.**
   - Public router intentionally ships active-passive first and defers
     sticky/weighted routing:
     [D-15](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L118-L155).
   - Follow-ups are tracked:
     [D-15 action items](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L175-L201).
   - Conclusion: not drift; defer until load-balancing needs are real.

6. **AbsMaxFP8 failure.**
   - Pre-existing and not introduced by NVFP4:
     [state](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/state.md#L154-L159),
     [quantization](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/quantization.md#L202-L214).
   - Conclusion: fix separately; not a Dreamverse public-dependency blocker.

### Drift summary table

| # | Item | Priority | Effort | Status | Tracked where | Next action |
|---:|---|---|---|---|---|---|
| 1 | Dreamverse README still names `../FastVideo-internal` | P0 | S | Real drift | [README lines](file:///home/william5lin/Dreamverse/README.md#L76-L109) | Update docs to public FastVideo / PyPI path. |
| 2 | Dreamverse private bootstrap clones internal repo | P0 | S | Real drift | [bootstrap script](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L7-L11) | Replace or delete private bootstrap. |
| 3 | Dreamverse `prompt_enhancer.py` fork | P1 | M | Real drift | [DR-1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L174-L206) | Build compat shim over public enhancer. |
| 4 | `cerebras_ifm` provider path | P1/P2 | S-M | Real drift / decision | [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229) | Choose public provider vs Dreamverse custom provider. |
| 5 | Health route mismatch | P1 | M-L | Real drift | [open item #1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99) | Add `/healthz`, `/readyz`, `/status` to public build_app. |
| 6 | Missing public streaming demo config | P2 | S | Real drift | [Dreamverse config](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L1-L12) | Add `examples/serving/streaming_demo.yaml`. |
| 7 | Standalone upsampler CLI | P2/P3 | S-M | Real drift if standalone CLI is desired | [internal CLI](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L15-L35) | Decide CLI promise; port or defer. |
| 8 | Layerwise offload utility | P3 | M | Optional internal-only residual (no Dreamverse deployment requires it today) | [internal manager](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L15-L20) | Defer until memory-tight deployment needs it. |
| 9 | LTX-2 stage equivalence | P2 | S-M | Verification gap | [public refine stage](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L1-L22) | Add targeted parity audit/test if needed. |
| 10 | StepVideo | P3 | M | Accepted residual | [Dreamverse model registry](file:///home/william5lin/Dreamverse/server/config.py#L28-L45) | No action unless Dreamverse adds StepVideo. |
| 11 | Debug-only fields | P3 | S | Accepted residual | [internal args](file:///home/william5lin/FastVideo-internal/fastvideo/fastvideo_args.py#L200-L203) | Do not publicize unless needed. |
| 12 | `layer_profile` typed quant knob | P2 | M | Tracked gap | [open item #4](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L244-L260) | Add typed layer profile if Dreamverse drops escape hatch. |
| 13 | `ltx2_image_crf` per-segment field flow (D-8) | P1 | S | Open verification gap — Dreamverse still passes `ltx2_image_crf=0.0` per [Dreamverse video_generation.py](file:///home/william5lin/Dreamverse/server/video_generation.py#L420-L435); needs trace-through to confirm it lands on `request.stage_overrides.refine.image_crf` rather than being silently dropped | [D-8 in open-threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L52-L68) | 10-min trace + add a Dreamverse-shape contract test pinning the field. |
| 14 | `video_position_offset_sec` semantics (VPO) | P1 | S | Open decision — persistent-vs-per-segment ambiguity unresolved | [VPO in open-threads](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L120-L144) | Confirm semantics with audio team; document + add test. Decision deadline was "before PR 7.6 emits state" — that PR (7.6 / #1257) is now MERGED, so the decision is overdue. |
| 15 | `GpuPool` ABC docstring missing experimental caveat (D-12-A) | P3 | trivial | Tracked gap — `GpuPool` ABC at [gpu_pool.py:74-83](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/gpu_pool.py#L74-L83) lacks the "API may change post-PR-7.10; experimental / server-internal" caveat | [D-12-A](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L301-L313) | Edit docstring; trivial. |
| 16 | `GpuPool.run_async()` migration (D-12-B) | P2 | M | Tracked gap — `GpuPool.run() -> Any` should become `run_async() -> AsyncIterator[VideoEvent]` per D-12 / D-12-B | [D-12-B](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L317-L327) | Land alongside #1288 merge or in immediate follow-up. |
| 17 | `SessionStore` / `BlobStore` lifecycle policy (SBS) | P2 | M | Tracked gap — in-memory defaults have no eviction/TTL/blob-cleanup policy | [SBS](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L278-L294) | Streaming server design pass needed before high-traffic deployment. |
| 13 | Router sticky / active-active | P3 | M | Deferred | [D-15](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L175-L201) | Defer until reconnect/load evidence. |
| 14 | AbsMaxFP8 test failure | P2 | S | Separate tech debt | [state](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/state.md#L154-L159) | Fix outside Dreamverse migration. |
| 15 | Dynamo backend package | P1 | External | Not FastVideo drift | [Dynamo contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L188-L210) | Reopen Dynamo-side PR after public API lands. |

---

## Part 2 — Integration path tradeoffs

### The four options

#### Option A — Status quo: Dreamverse stays separate and depends on `fastvideo`

**Shape**

- FastVideo remains the Python library and reusable backend runtime.
- Dreamverse remains the product repo with FastAPI product glue and Next.js
  frontend.
- Dreamverse `server` extra depends on `fastvideo>=0.1.7`:
  [pyproject](file:///home/william5lin/Dreamverse/pyproject.toml#L17-L22).
- Local development can keep using editable `../FastVideo` until PyPI
  publication catches up:
  [uv.lock](file:///home/william5lin/Dreamverse/uv.lock#L716-L722).

**What it solves**

- Directly satisfies "Dreamverse depends on public FastVideo".
- Keeps frontend release cadence independent.
- Keeps product-specific prompts, routes, and UI in the product repo.
- Minimizes FastVideo packaging and CI growth.

**What it does not solve by itself**

- Does not remove Dreamverse prompt-enhancer fork unless DR-1 is executed.
- Does not give Dreamverse FE compatibility with public `build_app` until
  health routes migrate.
- Does not make Dreamverse server itself reusable as a public entrypoint.

**Best fit**

- Default for the next release if the goal is to stop using
  FastVideo-internal quickly and safely.

#### Option B — Dreamverse as a subfolder under FastVideo

**Shape**

- One repository: FastVideo contains `dreamverse/server/` and
  `dreamverse/apps/web/`.
- Dreamverse can remain a separate package in the same repo, or FastVideo's
  build can ignore Dreamverse by default.
- CI must understand Python library tests plus Next.js install/build/test.

**What it solves**

- Eliminates sibling-checkout drift.
- Makes cross-repo integration changes atomic.
- Easier for a single reviewer to see library and product changes together.

**Costs**

- Adds frontend dependency management to a Python ML library repo.
- Couples clone size, CI setup, issue tracking, and review load.
- Forces maintainers to decide whether product assets are included in source
  distributions, wheels, docs, and release notes.

**Best fit**

- Only if Dreamverse becomes the primary FastVideo product surface and the
  team accepts a product monorepo.

#### Option C — Full merge into `fastvideo.entrypoints.dreamverse.*`

**Shape**

- Dreamverse backend becomes FastVideo code.
- Public import becomes something like
  `from fastvideo.entrypoints.dreamverse import build_app`.
- CLI becomes `fastvideo dreamverse-serve --config dreamverse.yaml`.
- Frontend either ships as static assets in the package or as a frontend
  extra.

**What it solves**

- One namespace and one release train for library plus product backend.
- No dependency boundary between Dreamverse server and FastVideo internals.
- Product route contract can be tested entirely inside FastVideo CI.

**Costs**

- Maximally expands FastVideo's public/security surface.
- Locks product experiments to FastVideo release cadence.
- Makes private prompt/provider/product assumptions look like framework API.
- Has weak precedent for a Python ML library plus Next.js product being merged
  into the library namespace.

**Best fit**

- Only if Dreamverse is no longer a separate product and becomes the
  canonical FastVideo UI/serving mode.

#### Option D — Hybrid: backend merges, frontend stays separate

**Shape**

- Reusable backend components merge into public FastVideo.
- Frontend stays in a separate Dreamverse UI repo or Dreamverse product repo.
- The backend should be generic where possible: `fastvideo.entrypoints.streaming`,
  not product-only names, unless product-only routes are intentionally
  accepted as public API.
- This matches the current trajectory: streaming server, GPU pool, prompt
  enhancer, safety/rewrite/session logging, router, NVFP4, and
  `generate_async` are public-side work already tracked in the PR roadmap:
  [pr-roadmap](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L19-L42).

**What it solves**

- Removes FastVideo-internal dependency for reusable backend pieces.
- Keeps product frontend cadence independent.
- Gives non-Dreamverse users a streaming backend and typed API without
  carrying the Dreamverse app.
- Gives Dynamo a stable library API while leaving Dynamo package code in
  Dynamo:
  [Dynamo contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L188-L210).

**Costs**

- Requires careful boundary discipline: generic streaming/server code in
  FastVideo; product routes/prompts/presets in Dreamverse.
- Requires contract tests to prevent drift.
- Some Dreamverse compatibility routes may become public and need support.

**Best fit**

- Best long-term target if the team wants FastVideo to own serving/runtime
  infrastructure while keeping Dreamverse as a separately evolving product.

### Comparison matrix

| Criterion | A. Separate dep | B. Subfolder monorepo | C. Full namespace merge | D. Hybrid backend merge |
|---|---|---|---|---|
| Alignment with stated goal | High: Dreamverse depends on public package | Medium: no external dep, but product becomes repo-local | Medium: dependency disappears by absorption | High: reusable backend in public, product separate |
| Time to remove `FastVideo-internal` | Fastest | Medium | Slowest | Medium-fast |
| Build complexity | Low | High: Python + Next.js in one repo | High: Python package plus static/frontend extras | Medium: Python backend only in FastVideo |
| Release cadence | Independent | Coupled clone; releases can still be separate but more friction | Fully coupled | Backend coupled to FastVideo, frontend independent |
| Security surface in FastVideo | Low | Medium/High | Highest | Medium |
| Contributor friction | Low for both repos | Higher for library contributors | Highest; product assumptions in library | Medium; clear backend boundary needed |
| Dependency management | Normal package pin | Workspace/monorepo tooling needed | FastVideo extras/static asset decisions needed | FastVideo extras for backend; FE out-of-tree |
| CI cost | Low/medium | High | High | Medium |
| Contract-test value | High; cross-repo contract tests are essential | Medium; same repo but still useful | Medium; less boundary pressure | High; generic backend vs product boundary |
| Precedent strength | Strong: library/server plus external UI patterns exist | Mixed | Weak for Python ML library + Next.js inside namespace | Strongest match: in-tree server/backend, external UI |
| Packaging risk | Low | Medium/high | High | Medium |
| Future Dynamo fit | Strong | Strong if API remains clean | Risky if product API bleeds in | Strong |
| Frontend iteration speed | Highest | Lower | Lowest | Highest |
| Risk of product-specific API leakage | Low | Medium | High | Medium; controllable with naming discipline |
| Reversibility | High | Medium | Low | Medium/high |

### OSS precedents (with citations)

| Pattern | Project | What it supports | Citation |
|---|---|---|---|
| Library plus in-tree server | vLLM | A Python ML library can ship an in-tree OpenAI-compatible server while clients remain external. | https://github.com/vllm-project/vllm/blob/bcf5cac9fb956788f649d1f5297b74c886a9d6d3/README.md#L64-L74 |
| Service packaging | BentoML | Packaging model + service + dependencies is supported, but CWD packaging creates discipline needs. | https://github.com/bentoml/BentoML/blob/32230a5276a8da8b23c4a06a9ec6272c1993451a/docs/source/build-with-bentoml/asgi.rst#L5-L18 |
| YAML-driven production serving | Ray Serve | Production updates should avoid in-place mutation; use new deployment/traffic switch. | https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html |
| Library/server plus external UI | TGI + ChatUI | Server can live with backend project while UI is separate. | https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/docs/source/basic_tutorials/consuming_tgi.md#L182-L186 |
| Lean library plus examples elsewhere | Transformers.js | Library stays lean; demos/examples can live outside core. | https://github.com/huggingface/transformers.js/blob/f7487c737aa8cafbc106c9adf69dc9578c8f3fe0/README.md#L26-L34 |
| Product monorepo that later split frontend | ComfyUI | Product UI/server monorepo can hit release-cadence mismatch and split FE later. | https://github.com/comfyanonymous/ComfyUI/blob/fed8d5efa6b70d5b24c4c33cb643bfccc39d45b5/README.md#L131-L149 and https://github.com/Comfy-Org/ComfyUI_frontend/blob/60f789d58070a9d1d789b260f83c36d7293a39f0/README.md#L31-L60 |
| Tightly coupled UI/server product | AUTOMATIC1111 SD WebUI | Product repos can couple UI/server tightly, but security surface becomes product-sized. | https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/webui.py#L48-L104 |

#### Precedent synthesis

- Strong precedents exist for a Python ML library shipping a server entrypoint.
- Strong precedents exist for keeping frontend/product UI out of the backend
  library repo.
- The cited set does not contain a clean precedent for merging a Next.js
  product into a Python ML library namespace.
- The most applicable pattern is **backend/server in the ML project,
  product UI outside**.

### Recommendation

#### Recommend Option D, constrained: backend merges as generic FastVideo streaming; frontend stays separate

Recommendation: follow **Option D** as the long-term architecture, but keep
the backend merge generic. In practice, this means continuing the current
public FastVideo path:

- `fastvideo.entrypoints.streaming.*` owns reusable streaming runtime.
- `fastvideo.entrypoints.streaming.gpu_pool` owns generic GPU worker pools.
- `fastvideo.entrypoints.streaming.prompt.*` owns provider-agnostic prompt
  operations.
- `fastvideo.entrypoints.streaming.router.*` owns FastVideo-aware routing.
- `fastvideo.api` owns typed construction, requests, results, events, and
  continuation state.
- Dreamverse keeps product-only FE, curated presets, prompt UX, product
  routes, and launch scripts.

This is effectively the path already underway in PRs #1257, #1258, #1284,
#1286, and #1288:
[PR roadmap](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L21-L42).

#### Why not Option A as the final answer?

Option A is the fastest near-term release posture and should be used as the
immediate migration posture. However, plain status quo is not enough for
the ultimate goal because reusable backend pieces still need to live in
public FastVideo so Dreamverse can stop reaching into internal code. That
work is already partly complete:

- GPU pool: [D-12](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L47-L116).
- Prompt enhancer: [PR roadmap 7.7](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L32-L35).
- Streaming auxiliaries: [PR roadmap 7.8](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/pr-roadmap.md#L35-L36).
- Router: [D-15](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L118-L155).
- `generate_async`: [streaming-server unlock PR](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L314-L345).

So the practical answer is:

- **Near term:** Option A operationally, after docs/scripts are fixed.
- **Architecture target:** Option D, with generic backend ownership in
  FastVideo and product ownership in Dreamverse.

#### Why not Option B?

Option B makes cross-repo coordination easier but imports frontend build,
package, and CI complexity into FastVideo. That is unnecessary while a
normal package dependency plus contract tests can guard the integration.
FastVideo's current repo structure is a Python package with examples and
docs, not a product monorepo:
[codebase map](file:///home/william5lin/FastVideo/.agents/memory/codebase-map/README.md#L5-L75).

#### Why not Option C?

Option C makes the product backend a public FastVideo namespace. That is
only appropriate if the team wants to support Dreamverse as a first-class
FastVideo product surface. Today the known public obligations are generic:
typed requests, streaming server, GPU pool, prompt provider protocol,
router, NVFP4, and Dynamo event APIs. Product-only Dreamverse behavior does
not need to become framework API.

#### Conditions that would change the recommendation

Move from constrained D toward **C** only if all of these become true:

1. Dreamverse is declared the canonical FastVideo serving product.
2. Product routes such as curated presets and prompt-system config are
   accepted as public FastVideo API.
3. FastVideo maintainers accept the security and support surface.
4. Release cadence for product UX and FastVideo core is intentionally
   coupled.
5. Frontend packaging/static asset strategy is explicitly owned by
   FastVideo.

Move from constrained D back toward **A** if any of these become true:

1. Prompt enhancement, router, or GPU pool turn out to be Dreamverse-only.
2. No second user appears for the streaming backend outside Dreamverse.
3. FastVideo maintainers want to minimize serving surface and publish only
   Python library APIs.
4. Dreamverse needs product changes faster than FastVideo can release.
5. Security review rejects in-tree serving/router responsibilities.

### Migration sketch for the recommended path

#### Phase 0 — Land the public backend stack

- **Effort:** Large, already in flight.
- **Owner:** FastVideo public.
- **Files:** #1288 scope, especially `fastvideo/api/`,
  `fastvideo/entrypoints/video_generator.py`,
  `fastvideo/entrypoints/streaming/`, LTX-2 pipeline stages, NVFP4 files,
  and contract tests.
- **Exit criteria:** #1288 merges; public `fastvideo.api.VideoEvent` and
  `VideoGenerator.generate_async` are available:
  [results.py](file:///home/william5lin/FastVideo/fastvideo/api/results.py#L109-L164),
  [video_generator.py](file:///home/william5lin/FastVideo/fastvideo/entrypoints/video_generator.py#L264-L332).

#### Phase 1 — Fix Dreamverse dependency docs and bootstrap

- **Effort:** Small.
- **Owner:** Dreamverse.
- **Files:**
  - [README.md](file:///home/william5lin/Dreamverse/README.md#L76-L109)
  - [bootstrap script](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L7-L11)
  - [pyproject.toml](file:///home/william5lin/Dreamverse/pyproject.toml#L17-L22)
  - [uv.lock](file:///home/william5lin/Dreamverse/uv.lock#L716-L722)
- **Exit criteria:** no user-facing docs or scripts mention
  `FastVideo-internal` as the expected dependency path.

#### Phase 2 — Add public health/readiness/status route compatibility

- **Effort:** Medium/Large.
- **Owner:** FastVideo public.
- **Files likely to touch:**
  - `fastvideo/entrypoints/streaming/server.py::build_app`
  - new `fastvideo/entrypoints/streaming/health.py`
  - tests under `fastvideo/tests/entrypoints/streaming/`
- **Source route shapes:**
  [Dreamverse health routes](file:///home/william5lin/Dreamverse/server/routes/health.py#L34-L79).
- **Tracking:** [open item #1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99).
- **Exit criteria:** Dreamverse FE can target public `build_app` for
  `/healthz`, `/readyz`, `/status`, and `/v1/stream`; product-only routes
  remain feature-detected.

#### Phase 3 — Replace Dreamverse prompt enhancer fork

- **Effort:** Medium.
- **Owner:** Dreamverse.
- **Files likely to touch:**
  - new `Dreamverse/server/prompting/_internal_compat.py`
  - `Dreamverse/server/runtime.py`
  - `Dreamverse/server/main.py`
  - `Dreamverse/server/prompt_enhancer.py`
- **Public API:**
  [PromptEnhancer](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/prompt/enhancer.py#L66-L142),
  [LLMProvider](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/prompt/providers/base.py#L63-L75).
- **Tracking:** [DR-1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L174-L206).
- **Exit criteria:** Dreamverse no longer carries a full local fork for the
  generic prompt operations public FastVideo already owns.

#### Phase 4 — Decide and implement `cerebras_ifm`

- **Effort:** Small decision plus small/medium implementation.
- **Owner:** Team decision, then Dreamverse or FastVideo.
- **Default recommendation:** Dreamverse-side custom provider.
- **Public schema source:**
  [PromptEnhancerConfig](file:///home/william5lin/FastVideo/fastvideo/api/schema.py#L229-L235).
- **Tracking:** [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229).
- **Exit criteria:** Dreamverse IFM provider works after prompt fork removal.

#### Phase 5 — Move streaming demo config into FastVideo examples

- **Effort:** Small.
- **Owner:** FastVideo public.
- **Source:**
  [Dreamverse streaming_demo.yaml](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L1-L149).
- **Target:** `examples/serving/streaming_demo.yaml`.
- **Exit criteria:** users can reproduce the typed streaming path from the
  FastVideo repo without checking out Dreamverse.

#### Phase 6 — Remove `experimental["pipeline_config"]` where practical

- **Effort:** Medium for `layer_profile`; Large for a full typed
  `dit_config.quant_config` carrier.
- **Owner:** FastVideo public, then Dreamverse cleanup.
- **Tracking:**
  [open item #4](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L244-L260),
  [quantization follow-up](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/quantization.md#L175-L200).
- **Exit criteria:** Dreamverse can express its quant layer profile through
  typed config instead of in-memory mutation.

#### Phase 7 — Decide standalone upsampler CLI

- **Effort:** Small/Medium.
- **Owner:** FastVideo public.
- **Input:** internal standalone utility
  [upsample.py](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/upsample.py#L120-L180)
  and internal CLI
  [cli/upsample.py](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L48-L130).
- **Public alternative:** SR refine stage already covers in-pipeline latent
  upsampling:
  [ltx2_refine.py](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L116-L180).
- **Exit criteria:** explicit decision: port CLI, document refine-stage-only
  support, or defer.

#### Phase 8 — Validate LTX-2 stage parity and offload residuals

- **Effort:** Small/Medium for stage parity; Medium for layerwise offload.
- **Owner:** FastVideo public.
- **Stage source:**
  [public LTX-2 stages](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/).
- **Offload source:**
  [internal layerwise offload](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L15-L20).
- **Exit criteria:** no known behavior gap between internal and public LTX-2
  stages; offload is either deliberately deferred or ported with tests.

### Open questions

1. **Which provider path for `cerebras_ifm`?**
   - Public provider or Dreamverse-side custom provider?
   - Default recommendation: Dreamverse-side unless there is another user.
   - Source: [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229).

2. **Should public FastVideo support standalone LTX-2 file upsampling?**
   - If yes, port internal CLI.
   - If no, document that SR support is pipeline-refine only.
   - Sources: [internal CLI](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L15-L35),
     [public refine stage](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L1-L22).

3. **Does Dreamverse need layerwise CPU offload?**
   - If memory-tight deployments require it, port as generic FastVideo.
   - Otherwise defer.
   - Source: [internal offload manager](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L15-L20).

4. **How much Dreamverse route surface should FastVideo own?**
   - Health/readiness/status should migrate because they are part of
     streaming-server compatibility.
   - Curated presets and prompt-system config should stay Dreamverse-side
     unless product policy changes.
   - Source: [route contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/streaming-server.md#L232-L259).

5. **Should `layer_profile` be the only near-term quant typed addition?**
   - Adding `layer_profile` is bounded.
   - A typed carrier for arbitrary mutated `PipelineConfig` is larger design
     work.
   - Source: [quantization follow-ups](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/quantization.md#L175-L200).

6. **When does Option D become Option C?**
   - Only if Dreamverse backend routes become public FastVideo product API.
   - Until then, keep generic streaming code in FastVideo and product code in
     Dreamverse.

---

## Part 3 — Action items

1. **P0 / S — Update Dreamverse README dependency notes.**
   - Replace `../FastVideo-internal` with public FastVideo instructions.
   - Preserve local editable `../FastVideo` dev flow where useful.
   - Source: [README stale lines](file:///home/william5lin/Dreamverse/README.md#L76-L109).

2. **P0 / S — Replace or remove private FastVideo bootstrap script.**
   - Current script clones `FastVideo-internal` and verifies imports from it.
   - Source: [script](file:///home/william5lin/Dreamverse/.agents/skills/bootstrap-fastvideo-private-fork/scripts/bootstrap_fastvideo_private.sh#L7-L11).

3. **P1 / M-L — Add `/healthz`, `/readyz`, and `/status` to public `build_app`.**
   - Keep `/curated-presets` and prompt-system config in Dreamverse.
   - Source: [open item #1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L69-L99).

4. **P1 / M — Replace Dreamverse prompt-enhancer fork with compat shim.**
   - Wrap public `PromptEnhancer`.
   - Keep only Dreamverse-specific metadata and product fallback behavior.
   - Source: [DR-1](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L174-L206).

5. **P1 / S-M — Decide `cerebras_ifm` provider path.**
   - Default: Dreamverse custom provider via `register_provider`.
   - Source: [DR-2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L211-L229).

6. **P2 / S — Add `examples/serving/streaming_demo.yaml` to FastVideo.**
   - Start from Dreamverse config and remove Dreamverse-private comments.
   - Source: [streaming_demo.yaml](file:///home/william5lin/Dreamverse/serve_configs/streaming_demo.yaml#L1-L149).

7. **P2 / M — Verify each public LTX-2 colocated stage against internal behavior.**
   - Start with refine, denoising, latent prep, image conditioning, text
     encoding, and audio decoding.
   - Source: [public refine stage](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py#L1-L22).

8. **P2 / M — Add typed `transformer_quant_layer_profile` if Dreamverse needs it.**
   - Thread schema → compat → `FastVideoArgs._apply_transformer_quant`.
   - Source: [open item #4](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L244-L260).

9. **P2 / S-M — Decide standalone LTX-2 upsampler CLI support.**
   - Port internal CLI only if file-to-file upsampling is a public workflow.
   - Source: [internal upsample CLI](file:///home/william5lin/FastVideo-internal/fastvideo/entrypoints/cli/upsample.py#L15-L35).

10. **P2 / S — Fix pre-existing AbsMaxFP8 test failure separately.**
    - Do not block Dreamverse migration on it.
    - Source: [open item #2](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/open-threads.md#L100-L117).

11. **P2 / S-M — Document public streaming install extras and dependencies.**
    - Include router `websockets`, prompt enhancer provider SDKs, and optional
      safety classifier extras.
    - Source: [D-16 dependency note](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L224-L242).

12. **P2 / S — Keep contract tests in the FastVideo CI path.**
    - Guard Dreamverse shape, Dynamo shape, and async events.
    - Sources: [Dreamverse test](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dreamverse_shape.py#L1-L26),
      [Dynamo test](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_dynamo_shape.py#L1-L19),
      [async test](file:///home/william5lin/FastVideo/fastvideo/tests/contract/test_generate_async.py#L1-L7).

13. **P3 / M — Defer layerwise offload until a deployment needs it.**
    - Port only as generic FastVideo utility with tests.
    - Source: [internal offload](file:///home/william5lin/FastVideo-internal/fastvideo/models/layerwise_offload.py#L15-L20).

14. **P3 / M — Defer StepVideo public parity for this integration.**
    - Dreamverse model registry is LTX-2/LTX-2.3 only.
    - Source: [Dreamverse config](file:///home/william5lin/Dreamverse/server/config.py#L28-L45).

15. **P3 / S — Do not add debug-only fields to public schema by default.**
    - Keep them private unless there is a user-facing debugging workflow.
    - Source: [internal debug args](file:///home/william5lin/FastVideo-internal/fastvideo/fastvideo_args.py#L200-L203).

16. **P3 / M — Keep router active-active and sticky routing deferred.**
    - Add only when session-routing evidence justifies it.
    - Source: [D-15 action items](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L175-L201).

17. **P3 / S — Preserve Dynamo as an external backend package.**
    - FastVideo should expose typed API; Dynamo code lives in Dynamo.
    - Source: [Dynamo contract](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/cross-repo-surfaces.md#L188-L210).

18. **P3 / S — After #1288 merges, update memory-dir state.**
    - Mark item D resolved and update branch tips.
    - Source: [runbook post-merge steps](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/runbook.md#L51-L70).

19. **P3 / S — Remove stale split-PR mental model from follow-up docs.**
    - #1288 is the current vehicle; split bookmarks are historical.
    - Source: [D-17 implications](file:///home/william5lin/FastVideo/.agents/memory/dreamverse-integration/decisions-log.md#L34-L45).

20. **P3 / S — Keep product-only Dreamverse frontend out of FastVideo unless explicitly re-scoped.**
    - This preserves release cadence and avoids packaging bloat.
    - Source: [Dreamverse baseline](file:///home/william5lin/Dreamverse/README.md#L5-L16).
