# Handoff: LTX-2 NVFP4 wire-up + Dreamverse launch-demo skill

This document hands off in-flight work to the next coding agent. It covers
two related streams that landed across two repos:

1. **FastVideo** (`will/ltx2_sr_port`): wire NVFP4 (NVIDIA's block-scaled
   FP4) inference + per-component torch.compile + supporting parity fixes
   so the public package matches `FastVideo-internal` for the LTX-2
   distilled streaming path used by Dreamverse.
2. **Dreamverse** (`will/integrate-public-fastvideo`): switch the GPU
   worker to the typed `GeneratorConfig` API, rename `FP4Config` →
   `NVFP4Config`, add a `launch-demo` skill + canonical
   `serve_configs/streaming_demo.yaml` for `fastvideo serve --config`.

Stack remains green: 222/222 FastVideo unit/contract/api tests pass; 8/8
Playwright e2e tests pass against the live `dreamverse-server` + Next.js
stack.

---

## Repo + branch state

| Repo | Path | Branch | Tip |
| --- | --- | --- | --- |
| FastVideo | `/home/william5lin/FastVideo` | `will/ltx2_sr_port` | `c6c14c55` |
| Dreamverse | `/home/william5lin/Dreamverse` | `will/integrate-public-fastvideo` | `3d7fd89` |
| Reference (read-only) | `/home/william5lin/FastVideo-internal` | (their) `main` | source of truth for parity |

> **Working branch on FastVideo is `will/ltx2_sr_port`, not the default checkout.**
> The shell may report `will/uv-pip-install-everywhere` because that was
> the earlier checkout. Run `git checkout will/ltx2_sr_port` before
> picking up FastVideo work.

### Live processes (do not duplicate)

```
:8009  dreamverse-server    pid 2453227   (warmed, /readyz returns 200)
:5274  next-server (dev)    pid 2399103   (devtools build)
```

### Stashes

* FastVideo: `stash@{0}: WIP on main: …HunyuanVideo plugin…` — pre-existing,
  unrelated to this work, do not pop.
* Dreamverse: `stash@{0}: wip: server modular refactor (split
  config/prompting/runtime/session)` — 3867 lines of orphan modular split
  off this branch. Do not pop on this branch; recover on a separate
  feature branch if anyone wants to resurrect it.

---

## What landed (FastVideo: `cfccd292..c6c14c55`)

Six commits on top of the i2v / continuation latent port:

```
c6c14c55 test(nvfp4): lock LTX-2 wiring + typed transformer_quant flow
94c983a2 refactor(quant): rename FP4 → NVFP4 to disambiguate from other FP4 variants
42b30bf9 feat(ltx2): wire FP4 inference through fastvideo.layers.quantization
6da342ba feat(compile): per-component compile + transformer_refine + prepare hook
221cb20a feat(api): typed per-component CompileConfig + FastVideoArgs carriers
a4760bae fix(api): propagate generic refine_* args + match internal randn
```

Each commit message has the rationale. Highlights below.

### `a4760bae` — three small parity fixes

* `FastVideoArgs.__post_init__` now calls `_resolve_refine_args()` which
  copies the public-facing generic `refine_*` knobs onto their
  `ltx2_refine_*` runtime carriers. Was missing → callers that set
  `refine_lora_path=...` saw "applied to 0 layers" warnings as the value
  was silently dropped.
* `_randn_ltx2_video_latents` patch path reverted from `randn_tensor` →
  `torch.randn` to bit-match internal under single-generator inference.
  Identical for a single `torch.Generator` but diverges for
  `list[Generator]` (per-sample seeds).
* Classified 19 `refine_*` / `ltx2_refine_*` / i2v / `ltx2_audio_*` /
  `ltx2_conditioning_latent_*` / `ltx2_video_conditions` fields in the
  schema-parity inventory yaml.

### `221cb20a` — typed CompileConfig + FastVideoArgs carriers

`CompileConfig` (in `fastvideo/api/schema.py`) gained per-component knobs:

```python
@dataclass
class CompileConfig:
    enabled: bool = False                 # master DiT switch
    backend / fullgraph / mode / dynamic / extras  # master kwargs

    # Per-component overlays, None = inherit master `enabled`
    text_encoder_enabled: bool | None = None
    vae_enabled: bool | None = None
    audio_vae_enabled: bool | None = None

    # Per-component kwargs override master when non-empty
    dit_kwargs: dict = ...
    text_encoder_kwargs: dict = ...
    vae_kwargs: dict = ...
    audio_vae_kwargs: dict = ...
```

Matching carrier fields on `FastVideoArgs`:
`enable_torch_compile_text_encoder/vae/audio_vae` and
`torch_compile_kwargs_dit/text_encoder/vae/audio_vae`. Compat layer
round-trips them through `legacy_from_pretrained_to_config` and
`generator_config_to_fastvideo_args`. **No behavior change yet** — these
are surface ports only; consumed in the next commit.

### `6da342ba` — refine + per-component compile + prepare_for_compile

`composed_pipeline_base.post_init` now:

* compiles `transformer_refine` alongside `transformer` and
  `transformer_2` whenever the DiT compile flag is on (closes the LTX-2
  stage-2 silent-eager bug);
* dispatches per-component compile loops (text encoder, VAE, audio VAE)
  with per-component kwargs falling back to master when empty;
* calls `module.prepare_for_compile()` on each compiled submodule
  before invoking `torch.compile` (hook protocol — model-specific).
  Implemented on `Gemma3` to materialize HF weights outside Dynamo's
  tracer.

### `42b30bf9` — NVFP4 LTX-2 inference wire-up *(largest)*

End-to-end:

1. `models/dits/ltx2.py` — swap `nn.Linear` → `ReplicatedLinear` for the
   FP4-eligible subset (`LTXSelfAttention`, `LTXDistributedSelfAttention`,
   `FeedForward`/`GELUApprox`); plumb `quant_config` and `prefix=` from
   `BasicAVTransformerBlock` → `_init_transformer_blocks` → `LTXModel`
   → `LTX2Transformer3DModel`. Other linears
   (`TimestepEmbedding`, `PixArtAlphaTextProjection`, `patchify_proj`,
   `proj_out`, `AdaLayerNormSingle.linear`) stay `nn.Linear` —
   matches internal exactly.
2. Port `_supports_prequantized_input` and
   `_linear_project_with_optional_prequant` helpers. Attention forward
   pre-quantizes input once (`quantize_input`), reuses the
   `(x_fp4, x_scale, x_global_sf)` tuple for k/v projections when
   `context is x` — bit-matches internal's fused path.
3. `models/loader/fsdp_load.py` — new `_maybe_convert_model_to_nvfp4`
   helper detects via `isinstance(quant_method, NVFP4QuantizeMethod)`
   (no flag); calls `convert_model_to_nvfp4` to materialize
   `_nvfp4_weight*` / `_nvfp4_alpha` / `_weight_global_sf` buffers.
   `flashinfer` import is lazy (inside the helper), so the loader is a
   no-op on hosts without flashinfer.
4. `layers/quantization/__init__.py` — registered `"NVFP4"` in
   `QuantizationMethods` literal + `get_quantization_config`.
5. `api/compat.py` + `fastvideo_args.py` — typed
   `engine.quantization.transformer_quant: "NVFP4"` resolves to a
   concrete `NVFP4Config()` instance, carried on `FastVideoArgs.transformer_quant`,
   pinned onto `pipeline_config.dit_config.quant_config` in
   `__post_init__._apply_transformer_quant`. **The explicit setter
   (legacy mutation pattern) wins** if `dit_config.quant_config` is
   already non-None.
6. `layers/linear.py` — `LinearBase.__init__` now falls back to
   `UnquantizedLinearMethod` when `quant_config.get_quant_method` returns
   `None`. `NVFP4Config` only tags a curated subset of LTX-2 layers, and
   the previous `assert quant_method is not None` would crash any
   non-tagged layer that received a quant_config.

### `94c983a2` — FP4 → NVFP4 rename

NVIDIA's specific block-scaled fp4 format (e2m1 mantissa, fp32 alpha,
`layout_128x4` scale layout, group size 16) — distinct from MX-FP4 /
OCP-FP4 / generic e3m0. Mechanical rename, no behavior change:

* `fp4_config.py` → `nvfp4_config.py`
* `FP4Config` → `NVFP4Config`; `get_name()` returns `"nvfp4"`
* `FP4QuantizeMethod` → `NVFP4QuantizeMethod`
* `convert_model_to_fp4` → `convert_model_to_nvfp4`
* `QuantizationMethods` literal: `"FP4"` → `"NVFP4"`
* registered buffer names: `_fp4_weight`/`_fp4_alpha` →
  `_nvfp4_weight`/`_nvfp4_alpha`
* loader helper renamed
* test file rename + symbol updates

Internal-scope torch op namespace `fastvideo_fp4::*` and
`_get_ltx2_fp4_stage_profile` deliberately left as-is — purely
internal naming that mirrors FastVideo-internal.

### `c6c14c55` — contract + numerical lock-in tests

* `fastvideo/tests/ops/quantization/test_nvfp4_ltx2_wiring.py` (6 tests):
  asserts that `LTXSelfAttention.to_q/to_k/to_v/to_out` are
  `ReplicatedLinear`; `NVFP4Config()` attaches `NVFP4QuantizeMethod`
  on the quantized subset with the correct `layer_prefix`; non-tagged
  projections (cross-attn K/V, audio attn, audio FFN) fall back to
  `UnquantizedLinearMethod`; `BasicAVTransformerBlock` propagates
  `quant_config` and `prefix` correctly to all 4 attention modules +
  FFN at once.
* `fastvideo/tests/api/test_typed_quant_flow.py` (4 tests): asserts
  typed `engine.quantization.transformer_quant: "NVFP4"` →
  `NVFP4Config()` instance flow; default leaves `transformer_quant`
  None; explicit `dit_config.quant_config = …` wins over typed carrier.

---

## What landed (Dreamverse: `248060b..3d7fd89`)

Three commits on top of the e2e tier:

```
3d7fd89 feat(skill): launch-demo orchestrator + fastvideo serve YAML
d80c2a8 refactor(server): drive FP4 + per-component compile via typed GeneratorConfig
4cc6b30 chore: gitignore Playwright + Next.js build artifacts under apps/web
```

### `d80c2a8` — server/video_generation.py refactor

Three coordinated changes in the GPU worker:

* Replace legacy `load_kwargs` dict + `VideoGenerator.from_pretrained(model_root, **kwargs)`
  call with the typed `GeneratorConfig` (`EngineConfig` /
  `OffloadConfig` / `CompileConfig` / `PipelineSelection` /
  `ComponentConfig`). Refine knobs move from `ltx2_refine_*` flat
  kwargs into `preset_overrides["refine"]`. **The in-memory
  `pipeline_config` pin** (`dit_config.quant_config = NVFP4Config()`)
  keeps using the legacy `experimental["pipeline_config"]` carrier
  because typed `transformer_quant: "NVFP4"` doesn't yet support
  setting `layer_profile`.
* Rename FP4 → NVFP4.
* Re-enable `"mode": "max-autotune-no-cudagraphs"` (was commented out).
  Closes the last known divergence vs FastVideo-internal in the
  worker-level path trace.

### `4cc6b30` — gitignore Playwright/Next.js artifacts

Added `apps/web/{node_modules,.next,test-results,playwright-report}` to
`.gitignore`. Mirror of the existing `prod-ui/` ignore set.

### `3d7fd89` — launch-demo skill

```
.agents/skills/launch-demo/
├── SKILL.md
└── scripts/
    ├── launch_demo.sh                 # orchestrator: BE + FE + health probes + Ctrl-C trap
    ├── launch_backend_dreamverse.sh   # uv run dreamverse-server (default)
    ├── launch_backend_fastvideo.sh    # uv run fastvideo serve --config (typed path)
    └── launch_frontend.sh             # next dev (devtools/dev/single5s)
serve_configs/
└── streaming_demo.yaml                # canonical ServeConfig matching internal/ui
```

YAML has every field annotated with the internal source line it mirrors:
LTX-2 distilled, NVFP4, 121 frames @ 1088×1920 24fps, 5 inference steps,
2-step refine gs=1.0 add_noise=true, max-autotune-no-cudagraphs compile,
121-frame default request, 300s session timeout, 6 segment cap, av_fmp4
streaming, cinematic-drone warmup prompt, 2400s warmup timeout, 9
conditioning frames + 0 end-offset, prompt enhancer on with cerebras /
gpt-oss-120b / 20s timeout.

**Two BE flavors documented in SKILL.md:**

| `BE_FLAVOR=` | Boots | Routes served | FE compatible |
| --- | --- | --- | --- |
| `dreamverse` (default) | `dreamverse-server` | `/healthz`, `/readyz`, `/curated-presets`, `/v1/stream`, devtools, session monitor | ✓ full |
| `fastvideo` | `fastvideo serve --config <yaml>` | `/health`, `/v1/stream` | ⚠ FE will surface fetch errors for `/curated-presets`, `/readyz` until those routes migrate into FastVideo's `build_app` |

The fastvideo flavor exists today as the verifiable typed-config path
(YAML parses, streaming worker boots, dotted overrides work). It is not
yet a drop-in for the FE — see "Open follow-ups" below.

---

## Verified

* `222 passed, 1 skipped` across `fastvideo/tests/api/`,
  `fastvideo/tests/contract/`,
  `fastvideo/tests/ops/quantization/test_nvfp4_*`,
  `tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py`.
* `8 passed` Playwright e2e (backend-health 5, frontend-shell 2,
  preset-prompt-generation 1) against the live `dreamverse-server`
  + Next.js stack.
* `streaming_demo.yaml` parses cleanly against `ServeConfig`; the
  validation path of `fastvideo serve --config <yaml>` runs without
  error and accepts dotted overrides like `--server.port 8010`.
* FastVideo `bash -n` clean across all four launch scripts.

---

## Critical context (gotchas a successor should know)

### NVFP4 layer set is asymmetric — by design

`NVFP4Config.fp4_layers` covers:

* `attn1.{to_q,to_k,to_v,to_out}` — full self-attention
* `attn2.{to_q,to_out}` — cross-attn Q + out only (text context not quantized)
* `audio_to_video_attn.{to_q,to_out}` — AV cross Q + out
* `video_to_audio_attn.{to_k,to_v}` — VA cross K + V
* `ffn.{fc_in,fc_out}` — video FFN
* `adaln_single.linear` — but this is `nn.Linear` (not `LinearBase`),
  so it never actually gets FP4'd. List entry has no effect; matches
  internal.

**NOT in the set:** audio self-attention (`audio_attn1.*`), audio
cross-attention (`audio_attn2.*`), audio FFN (`audio.ffn.*`). Audio
path is cheap enough that quant overhead isn't worth it. Test
`test_basic_av_block_propagates_quant_config_to_all_children` locks
this in — if you add audio quantization later, update the test.

### `LinearBase` fallback is load-bearing

`fastvideo/layers/linear.py:191-202`: when `quant_config.get_quant_method`
returns `None` (layer not in the quant config's set), we fall back to
`UnquantizedLinearMethod`. **Do not remove this fallback** — it would
break every non-tagged `ReplicatedLinear` constructed with a
`NVFP4Config`, and `assert quant_method is not None` in
`ReplicatedLinear.__init__` would fire on unmatched layers.

### Typed `transformer_quant` precedence

`FastVideoArgs._apply_transformer_quant` only writes
`dit_config.quant_config` when it's currently `None`. If a caller has
explicitly set `pipeline_config.dit_config.quant_config = NVFP4Config(...)`,
the explicit setter wins. Dreamverse's `video_generation.py` relies on
this — it sets `NVFP4Config()` directly because the typed
`transformer_quant: "NVFP4"` doesn't expose `layer_profile`.

### Pre-existing AbsMaxFP8 test failure is NOT mine

`fastvideo/tests/ops/quantization/test_absmax_fp8.py::test_create_weights_rejects_invalid_dtype`
fails on `main` and on this branch with the same error
("AssertionError not raised"). I confirmed via `git stash` that the
failure pre-dates my changes. Not blocking; tracked as separate tech
debt.

### `transformer_refine` is auto-compiled with the master DiT flag

Set `enable_torch_compile=True` and `transformer_refine` compiles
along with `transformer` and `transformer_2`. There is **no separate
`enable_torch_compile_refine` flag** — by design, refine inherits the
DiT compile state to keep the typed surface small. If you need them
decoupled, add a new field; don't repurpose existing ones.

### `prepare_for_compile` is a duck-type protocol, not a base class method

Defined nowhere; called via `getattr(module, "prepare_for_compile", None)`
in `composed_pipeline_base._maybe_compile_pipeline_module`. Currently
only Gemma implements it (to materialize HF weights outside Dynamo).
Add to other models that have lazy external state if you observe
compile-time graph breaks.

### Public typed `PromptEnhancerConfig.provider` is `Literal["cerebras", "groq"]`

Internal supports `"cerebras_ifm"` (config.py:143). The public typed
schema does not. The `streaming_demo.yaml` defaults to `"cerebras"`.
For agents that need `cerebras_ifm`, the `dreamverse-server` flavor
respects the `FASTVIDEO_PROMPT_PROVIDER` env var (legacy path);
`fastvideo serve --config` does not currently expose it.

### Dreamverse `pipeline_config` is still a Python object passed via `experimental`

The typed `GeneratorConfig` doesn't have a clean home for an
in-memory `PipelineConfig` instance with mutated `dit_config`. We
pass it via `pipeline.experimental["pipeline_config"]` — the
`compat.py` legacy adapter recognizes that key and threads it through
to `FastVideoArgs.from_kwargs`. This is fine but not pretty; if
someone designs a typed `dit_config` carrier later, this becomes
obsolete.

### `fastvideo serve --config` is not yet a drop-in for the FE

`fastvideo.entrypoints.streaming.server.build_app` exposes only
`/health` and `/v1/stream`. The Dreamverse Next.js shell expects
`/healthz`, `/readyz`, `/status`, `/curated-presets`,
`/curated-presets/append`, `/prompt-system-config`, and the devtools
routes. These all live in `Dreamverse/server/main.py` +
`Dreamverse/server/routes/`. Until they migrate into FastVideo's
`build_app` (or are exposed via a Dreamverse-side proxy), the
`BE_FLAVOR=fastvideo` flavor is for verifying the typed serve config
path only — not for full FE compatibility.

---

## Open follow-ups (prioritized)

### High

1. **Migrate FE-required routes into FastVideo's `build_app`.**
   `/healthz`, `/readyz`, `/status` look obviously fastvideo-side
   (they're streaming-server health). `/curated-presets` and
   `/prompt-system-config` are operator-side surfaces and should
   probably stay in Dreamverse (or migrate as opt-in routes that the
   FE feature-detects). Without this, `BE_FLAVOR=fastvideo` is
   permanently a "diagnostic" flavor. Closes the
   `launch-demo` skill TODO.

2. **AbsMaxFP8 test failure cleanup.** Pre-existing. Either fix the
   test (`AbsMaxFP8LinearMethod.create_weights` no longer asserts on
   invalid dtype — restore the assert if intentional, otherwise drop
   the test).

### Medium

3. **Add `cerebras_ifm` to public `PromptEnhancerConfig.provider`
   Literal.** Trivial schema change; needs paired enhancer-side
   provider implementation in
   `fastvideo/entrypoints/streaming/prompt/providers/`.

4. **Expose `layer_profile` on typed `engine.quantization`.** Today
   `transformer_quant: "NVFP4"` always constructs `NVFP4Config()`
   with the default `layer_profile="refine"`. To support stage-1
   profiles (no `attn2.to_out`, no cross-modal AV) via typed config,
   add `transformer_quant_layer_profile: str | None = None` and
   thread it through `compat.py`. Dreamverse currently dodges this
   by setting `NVFP4Config()` directly via `experimental`.

5. **Typed `dit_config.quant_config` carrier.** The
   `experimental["pipeline_config"]` escape hatch in Dreamverse
   should eventually become a typed field. Design TBD.

### Low

6. **Audio attention quantization profile.** If an audio-quant
   profile is added to `NVFP4Config.fp4_layers` (currently audio attn
   and FFN are bf16), update
   `test_basic_av_block_propagates_quant_config_to_all_children`.

7. **Schema parity inventory.** A few internal-only fields are not
   exposed publicly (`PROMPT_HTTP_TIMEOUT_MS`,
   `PROMPT_INITIAL_STAGE_TIMEOUT_MS`, `PROMPT_TEMPERATURE`,
   `PROMPT_MAX_COMPLETION_TOKENS`, `PROMPT_AUTO_SLEEP_MS`,
   `PROMPT_AUTO_TIMEOUT_MS`, the curated-presets file paths).
   These all flow via env vars on `dreamverse-server` today; if
   `fastvideo serve --config` becomes the canonical entrypoint,
   they'll need typed homes.

8. **Empty `apps/web/test-results/` directory locally.** The
   `.gitignore` entry I added makes it invisible to `git status`,
   but the dir itself still has a stale `.last-run.json` (45 bytes)
   from a prior Playwright run. Harness blocked auto-cleanup
   ("pre-existing files"); the user can `rm -rf
   apps/web/test-results` whenever convenient.

---

## How to pick up work

### Quick orientation (run these first)

```bash
# FastVideo state
cd /home/william5lin/FastVideo
git checkout will/ltx2_sr_port
git log --oneline cfccd292..HEAD     # six commits added this round
.venv/bin/python -m pytest fastvideo/tests/api/ \
    fastvideo/tests/contract/ \
    fastvideo/tests/ops/quantization/test_nvfp4_*.py \
    tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py \
    -q --no-header   # expect 222 passed, 1 skipped

# Dreamverse state
cd /home/william5lin/Dreamverse
git log --oneline 248060b..HEAD      # three commits added this round
cat serve_configs/streaming_demo.yaml | head -40
ls .agents/skills/launch-demo/

# Live stack health (already running on this host)
curl -s http://localhost:8009/readyz | head -c 200
curl -s http://localhost:5274/ | head -c 100
( cd apps/web && npx playwright test --reporter=line )   # expect 8 passed
```

### Reference docs

* **FastVideo internal/ui parity source:** `../FastVideo-internal/ui/ltx2-streaming/server/config.py`
* **NVFP4 source on internal:** `../FastVideo-internal/fastvideo/layers/quantization/fp4_config.py`
* **Worker-trace audit:** `../FastVideo/dreamverse_review.md` (D-1
  multi-model, D-5 audio re-encode, prior gap inventory)
* **Schema parity inventory:** `docs/design/inference_schema_parity_inventory.yaml`
* **PR-plan for the broader migration:** `../FastVideo/PR plan.md`

### Files most likely to need touches in follow-ups

* `fastvideo/api/schema.py` — `CompileConfig`, `QuantizationConfig`,
  `PromptEnhancerConfig` Literal extension.
* `fastvideo/api/compat.py` — typed → flat translation.
* `fastvideo/fastvideo_args.py` — carrier fields and
  `_apply_transformer_quant`.
* `fastvideo/entrypoints/streaming/server.py::build_app` — add
  `/healthz`, `/readyz`, `/status` routes for FE compatibility (high
  priority follow-up #1).
* `Dreamverse/server/video_generation.py` — typed `GeneratorConfig`
  builder (current).
* `Dreamverse/serve_configs/streaming_demo.yaml` — every parity
  knob; edit here, not in shell scripts.

---

## Don't / Cautions

* **Don't pop the Dreamverse stash on this branch.** It's 3867 lines
  of orphan modular refactor (server/{config,prompting,runtime,session}/)
  with broken absolute imports. If anyone wants to resurrect it, do so
  on a separate feature branch.
* **Don't remove the `LinearBase` `UnquantizedLinearMethod` fallback.**
  See "Critical context" above.
* **Don't repurpose `enable_torch_compile` to mean DiT-only.** It also
  drives `transformer_refine` and `transformer_2` compile. Add a new
  flag if decoupling is needed.
* **Don't change `NVFP4Config` buffer names back to `_fp4_*`.** The
  rename is intentional to disambiguate from MX-FP4 / OCP-FP4.
* **Don't bypass the typed surface for new options.** New compile /
  quant / refine knobs should land on the dataclass + compat.py +
  parity inventory together. The existing test suite locks this in.
* **Don't merge to main without a CI run that covers FP4.** Current
  CI doesn't run flashinfer-dependent paths; the wiring tests in
  `test_nvfp4_ltx2_wiring.py` are CPU-only by design and don't
  exercise the actual FP4 kernels.

---

*Last updated: end of session that landed `c6c14c55` on FastVideo and
`3d7fd89` on Dreamverse. Stack remains green; no dirty state.*
