# PR Roadmap

Status of all 17 PRs in the FastVideo public API refactor + streaming
server upstream + Dynamo backend contract + post-deprecation cleanup.

For design rationale see [design.md](design.md). For streaming-specific
PRs (7.5-7.10) see [streaming-server.md](streaming-server.md). For NVFP4
work that runs parallel to this sequence see [quantization.md](quantization.md).

**Last updated:** 2026-05-05 (PRs 7.5 / 7.6 / 7.7 / 7.8 / 7.9 merged; PR 7.10 opened as #1287).

## Status legend

- ✅ **Landed on `origin/main`**
- 🟢 **Open / in flight** — branch exists, may have open PR
- 🟡 **Planned** — designed, not started
- 🔵 **Future** — deferred to post-PR-13 cleanup

## Landed PRs (0 → 7.7)

| # | PR | Status | Merge commit | Scope |
|---|---|---|---|---|
| 0 | #1218 [1/n] | ✅ | merged | Parity inventory + typed inference schema |
| 1 | #1218 [1/n] | ✅ | merged | Strict parser/validation/overrides + API tests |
| 2 | #1220 [2/n] | ✅ | merged | Typed `VideoGenerator` constructors + request path + compat |
| 3 | #1226 [3/n] | ✅ | merged | CLI/YAML-first typed config loading for `generate` and `serve` |
| 4 | #1234 [4/n] | ✅ | merged | Preset registry + presets for all 13 model families; `SamplingParam` moved to `fastvideo/api/`; `configs/sample/` deleted entirely |
| 5 | #1237 [5/n] | ✅ | merged | `ServeConfig.default_request` wired into stateless OpenAI server |
| 5.5 | (`5d1d71fc`) | ✅ | merged | Streaming server package skeleton, typed `StreamingConfig`/`GpuPoolConfig`/`PromptEnhancerConfig`/`PromptSafetyConfig`/`WarmupConfig`, `streaming-serve` CLI stub |
| 6 | #1239 [6/n] | ✅ | merged | LTX2 public preset + asset wiring + `gpu_pool.py` typed-kwarg translation |
| 7 | #1250 [7/n] | ✅ | merged | Typed LTX2 continuation state + streaming session store + blob store |
| **7.5** | **#1251** | ✅ | `95fd29e0` (merged 2026-04-26) | Streaming server skeleton (WebSocket + fMP4 + single generator). 8 commits. Deferred TODOs (per-step progress, mid-segment cancellation) carried forward to PR 7.10. |
| **7.6** | **#1257** | ✅ | `eb0a4152` (merged 2026-05-04) | GPU pool upstream + worker subprocess + two-segment warmup. 7 commits squashed. APPROVED by Eigensystem. See [decisions-log.md D-12](decisions-log.md#d-12) for the architectural review. |
| **7.7** | **#1258** | ✅ | `f673423b` (merged 2026-05-04) | Prompt enhancer with `LLMProvider` abstraction. Built-in providers: cerebras, groq. 3 commits squashed. **Public Literal does NOT include `cerebras_ifm`** — open-threads.md item DR-2 covers the gap. See [decisions-log.md D-13](decisions-log.md#d-13) for the architectural review. |
| **7.8** | **#1284** | ✅ | `eb3a3942` (merged 2026-05-04) | Streaming auxiliaries — `prompt/safety.py` (optional fasttext, lazy import), `prompt/rewrite.py`, `session_logger.py` (thread-safe JSONL), `mock_server.py` (build_mock_app + MockGenerator for FE dev). 730 LOC, 2 commits. See [decisions-log.md D-14](decisions-log.md#d-14). |
| **7.9** | **#1286** | ✅ | `2aaeee2a` (merged 2026-05-05) | Streaming router (multi-replica load balancer + WS proxy + `fastvideo router-serve` CLI). Squashed `cd76cf51 + 1ac1e732 + b0b7f59c + a152cb77` (router-polish second-pass; cherry-pick of `40e265b8` from `will/ltx2_sr_port`). See [decisions-log.md D-15](decisions-log.md#d-15) (structural review) + [D-16](decisions-log.md#d-16) (second-pass polish). |

## In flight (7.10)

| # | PR | Status | Branch | Scope |
|---|---|---|---|---|
| **7.10** | **#1287** | 🟢 OPEN, MERGEABLE | `will/api_7.10` (head `6ae7a99f`) | **Dynamo backend contract** — `VideoGenerator.generate_async`, `default_health_check_request()`, `VideoEvent` hierarchy. 3 commits, 468/4 LOC across 4 files. The "unlock PR" — enables PR 7.5's mid-segment cancellation, audio re-encode (Q-5), Dynamo progress passthrough (Q-9), and `GpuPool.run_async()` migration (D-12-B). Stacked on `main`; first slice of post-#1286 rebased `will/ltx2_sr_port`. |

## Planned (8 → 13)

| # | Status | Branch | Scope |
|---|---|---|---|
| 8 | 🟡 ready (slice 4-6 of `will/ltx2_sr_port`) | `will/api_8` (`f32e31ec`) | Server contract docs (OpenAI HTTP + Dynamo native-backend integration reference) + Dreamverse/Dynamo shape contract tests. 3 commits. |
| LTX-2 SR | 🟡 ready (slice 7-15) | `will/ltx2_sr_runtime` (`e7297519`) | LTX-2 SR runtime port + i2v conditioning + alignment harness against FastVideo-internal. 9 commits. NOT in canonical PR plan — separate stream. |
| NVFP4 | 🟡 ready (slice 16-21) | `will/ltx2_nvfp4` (`6793166b`) | NVFP4 wire-up + per-component compile + typed `transformer_quant` flow. 6 commits. NOT in canonical PR plan. |
| LTX-2 fixes | 🟡 ready (slice 22-23) | `will/ltx2_post_fixes` (`25897b67`) | Post-handoff parity fixes (Gemma `to()`, list-of-generators). 2 commits. |
| agents cleanup | 🟡 ready (slice 24-33, == `will/ltx2_sr_port` HEAD) | `will/agents_cleanup` (`b34d9704`) | `.agents/` Phase 1 cleanup + dreamverse-integration memory + STACK/CO-AUTHORS docs + authors.md + runbook.md + D-12...D-16 + open-threads tracking. 10 commits. **Independent of API stack** — could fast-track. |
| 9 | 🟡 | — | LongCat preset migration + colocation (9 model-specific stage files) |
| 10 | 🟡 | — | Hunyuan15 SR preset migration + colocation + SR field migration POC |
| 11 | 🟡 | — | SSIM/performance test migration off legacy `generate_video(..., **kwargs)` |
| 12 | 🟡 | — | Docs + examples migration (includes streaming server + Dynamo) |
| 13 | 🟡 | — | Deprecation cleanup (includes flat LTX2 kwargs the internal `gpu_pool.py` used to consume) |

## Future (compat.py death sequence)

After PR 13 lands deprecation warnings, `fastvideo/api/compat.py` (~370
lines) is the last translation shim between typed public API and legacy
internals (`FastVideoArgs`, `SamplingParam`).

| # | Status | Scope | Lines removed |
|---|---|---|---|
| 14 | 🔵 reachable | Strip forward translation: `legacy_from_pretrained_to_config`, `legacy_generate_call_to_request`, `_sampling_param_to_request_raw`, `_LEGACY_REQUEST_ALIASES`, `_LTX2_REFINE_FLAT_KEYS`. Depends on PRs 11/12/7.6 callers being migrated. | ~100 |
| 15 | 🔵 | `FastVideoArgs` becomes a `@dataclass` view over `GeneratorConfig` with `@property` accessors backing legacy field names. ~600-line god-object refactor. Depends on PR 14. | reverse-translation half (~150) trivial |
| 16 | 🔵 | `ForwardBatch` reads `GenerationRequest` by reference; kills `request_to_sampling_param` and the `ForwardBatch(**shallow_asdict(sampling_param), …)` spread. `SamplingParam` demoted or deleted. Depends on PR 15. | rest |
| 17 | 🔵 | Move `normalize_generator_config`, `normalize_generation_request`, `load_generator_config_from_file` to `parser.py`. Delete `compat.py`. | file gone |

PRs 15-17 touch training, distributed, and worker code in addition to
inference path; realistically 1-2 quarters beyond the current plan.

## Dependency chain

```
PR 13 (deprecation)
  ↓
PRs 11, 12, 7.6 (migrate callers)
  ↓
PR 14 (forward translation gone)      ─── ~100 lines out of compat.py
  ↓
PR 15 (FastVideoArgs as view)         ─── reverse-translation trivial
  ↓
PR 16 (ForwardBatch reads request)    ─── SamplingParam demoted
  ↓
PR 17 (move normalizers, delete file)
```

## NVFP4 work (out-of-band, parallel to PR 7.5+)

NOT in the canonical PR sequence. Lives on `will/ltx2_sr_port`
(currently @ `156103b9`) — a separate stack alongside the public-API
upstreaming. See [quantization.md](quantization.md) for what each commit
locks in.

| Commit range | Topic |
|---|---|
| `cfccd292..b6ac7630` | LTX-2 i2v + SR runtime port + alignment harness |
| `a4760bae..c6c14c55` | NVFP4 LTX-2 wire-up + per-component compile + parity fixes (May 2 handoff) |
| `a5fcd19c..156103b9` | Post-handoff parity/perf fixes |

## Key landed artifacts (reference points)

- Parity inventory: [`docs/design/inference_schema_parity_inventory.yaml`](file:///home/william5lin/FastVideo/docs/design/inference_schema_parity_inventory.yaml) + guard [`fastvideo/tests/api/test_schema_parity_inventory.py`](file:///home/william5lin/FastVideo/fastvideo/tests/api/test_schema_parity_inventory.py)
- Typed schema: [`fastvideo/api/schema.py`](file:///home/william5lin/FastVideo/fastvideo/api/schema.py)
- Compat layer: [`fastvideo/api/compat.py`](file:///home/william5lin/FastVideo/fastvideo/api/compat.py)
- Preset system: [`fastvideo/api/presets.py`](file:///home/william5lin/FastVideo/fastvideo/api/presets.py) + per-family `pipelines/basic/<family>/presets.py`
- Streaming package skeleton (PR 5.5): [`fastvideo/entrypoints/streaming/`](file:///home/william5lin/FastVideo/fastvideo/entrypoints/streaming/)
- LTX2 typed continuation state (PR 7): [`fastvideo/pipelines/basic/ltx2/continuation.py`](file:///home/william5lin/FastVideo/fastvideo/pipelines/basic/ltx2/continuation.py)

## Known notable decisions carried forward

- **Public inference boundary stays plain dataclasses + plain dict/YAML/JSON**
  — not OmegaConf, not runtime config wrappers.
- **Every public entrypoint normalizes into typed config objects** before
  touching legacy `FastVideoArgs` or `SamplingParam`.
- **Legacy `generate_video(..., **kwargs)` stays on direct legacy execution
  path until PR 11**'s SSIM/performance migration. Prevents golden
  baselines from drifting during compat period.
- **Typed requests use schema defaults**; legacy `generate_video(...)`
  continues to inherit model-specific `SamplingParam` defaults during
  compat period.
- **Preset registry uses explicit `_register_presets()` pattern** matching
  `_register_configs()`; lookup keyed by `model_family`.
- **Stateless OpenAI server clones `ServeConfig.default_request`** and
  merges user overrides; preset validation runs before legacy generation.
- **Streaming server added as sibling `fastvideo/entrypoints/streaming/`**
  rather than extending `fastvideo/entrypoints/openai/` (PR 5.5).

## Per-PR commit-level detail

For per-PR commit lists, test plans, and merge criteria, the archived
source [`source-archive/PR-plan.md`](source-archive/PR-plan.md) (1145 lines)
remains the deepest reference. This file is the navigable summary.
