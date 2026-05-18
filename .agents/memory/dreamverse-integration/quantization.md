# Quantization — NVFP4, LinearBase Fallback, Layer Profiles

What landed in the May 2 NVFP4 stack, why it's load-bearing, and what's
still owed (`layer_profile`, typed quant carrier, AbsMaxFP8 cleanup).

For overall API design see [design.md](design.md). For the open
follow-ups see [open-threads.md](open-threads.md).

**Last updated:** 2026-05-03.

## NVFP4 — what it is

NVIDIA's specific block-scaled FP4 format:

- e2m1 mantissa
- fp32 alpha
- `layout_128x4` scale layout
- group size 16

Distinct from MX-FP4 / OCP-FP4 / generic e3m0. The May 2 rename
(`94c983a2`) disambiguated the naming throughout FastVideo's public
surface.

## Files (current)

| File | Role |
|---|---|
| [`fastvideo/layers/quantization/nvfp4_config.py`](file:///home/william5lin/FastVideo/fastvideo/layers/quantization/nvfp4_config.py) | `NVFP4Config`, `NVFP4QuantizeMethod`, `convert_model_to_nvfp4` |
| [`fastvideo/layers/quantization/__init__.py`](file:///home/william5lin/FastVideo/fastvideo/layers/quantization/__init__.py) | `QuantizationMethods` literal includes `"NVFP4"`; `get_quantization_config` resolves it |
| [`fastvideo/layers/linear.py`](file:///home/william5lin/FastVideo/fastvideo/layers/linear.py) | `LinearBase.__init__` falls back to `UnquantizedLinearMethod` when `quant_config.get_quant_method` returns None — **load-bearing** |
| [`fastvideo/models/loader/fsdp_load.py`](file:///home/william5lin/FastVideo/fastvideo/models/loader/fsdp_load.py) | `_maybe_convert_model_to_nvfp4` helper detects via `isinstance(quant_method, NVFP4QuantizeMethod)`; calls `convert_model_to_nvfp4` to materialize buffers |
| [`fastvideo/models/dits/ltx2.py`](file:///home/william5lin/FastVideo/fastvideo/models/dits/ltx2.py) | `nn.Linear` → `ReplicatedLinear` for FP4-eligible subset; `_supports_prequantized_input` + `_linear_project_with_optional_prequant` helpers; quant_config + prefix= plumbing |
| [`fastvideo/api/compat.py`](file:///home/william5lin/FastVideo/fastvideo/api/compat.py) | Typed `engine.quantization.transformer_quant: "NVFP4"` resolves to `NVFP4Config()` instance |
| [`fastvideo/fastvideo_args.py`](file:///home/william5lin/FastVideo/fastvideo/fastvideo_args.py) | `__post_init__._apply_transformer_quant` pins `pipeline_config.dit_config.quant_config = NVFP4Config()` |

## Buffer naming (post-rename)

| Old | New |
|---|---|
| `_fp4_weight` / `_fp4_alpha` | `_nvfp4_weight` / `_nvfp4_alpha` |
| `_weight_global_sf` | unchanged |
| `convert_model_to_fp4` | `convert_model_to_nvfp4` |
| `FP4QuantizeMethod` | `NVFP4QuantizeMethod` |
| `QuantizationMethods` literal `"FP4"` | `"NVFP4"` |

Internal-scope torch op namespace `fastvideo_fp4::*` and
`_get_ltx2_fp4_stage_profile` deliberately left as-is — purely internal
naming that mirrors FastVideo-internal.

## Layer set asymmetry — by design

`NVFP4Config.fp4_layers` (default `layer_profile="refine"`) covers:

- `attn1.{to_q,to_k,to_v,to_out}` — full self-attention
- `attn2.{to_q,to_out}` — cross-attn Q + out only (text context not quantized)
- `audio_to_video_attn.{to_q,to_out}` — AV cross Q + out
- `video_to_audio_attn.{to_k,to_v}` — VA cross K + V
- `ffn.{fc_in,fc_out}` — video FFN
- `adaln_single.linear` — but this is `nn.Linear` (not `LinearBase`),
  so it never actually gets FP4'd. List entry has no effect; matches
  internal.

**NOT in the set:**

- audio self-attention (`audio_attn1.*`)
- audio cross-attention (`audio_attn2.*`)
- audio FFN (`audio.ffn.*`)

Audio path is cheap enough that quant overhead isn't worth it. Test
[`test_basic_av_block_propagates_quant_config_to_all_children`](file:///home/william5lin/FastVideo/fastvideo/tests/ops/quantization/test_nvfp4_ltx2_wiring.py)
locks this in — if you add audio quantization later, update the test.

## `LinearBase` fallback — DO NOT REMOVE

[`fastvideo/layers/linear.py:191-202`](file:///home/william5lin/FastVideo/fastvideo/layers/linear.py#L191-L202): when `quant_config.get_quant_method` returns
`None` (layer not in the quant config's set), we fall back to
`UnquantizedLinearMethod`.

**Removing this fallback would break every non-tagged
`ReplicatedLinear` constructed with an `NVFP4Config`** — the previous
`assert quant_method is not None` would crash on unmatched layers (e.g.
text-encoder K/V projections, audio attention, etc.).

This is one of the load-bearing changes from `42b30bf9`.

## `transformer_quant` precedence rules

`FastVideoArgs._apply_transformer_quant` only writes
`dit_config.quant_config` when it's currently `None`. **If a caller has
explicitly set** `pipeline_config.dit_config.quant_config = NVFP4Config(...)`,
the explicit setter wins.

Dreamverse's `video_generation.py` relies on this precedence — it sets
`NVFP4Config()` directly via `experimental["pipeline_config"]` because
typed `transformer_quant: "NVFP4"` doesn't yet expose `layer_profile`.
See "Open follow-ups" below.

## Attention forward optimization

[`models/dits/ltx2.py`](file:///home/william5lin/FastVideo/fastvideo/models/dits/ltx2.py)
ports `_supports_prequantized_input` and
`_linear_project_with_optional_prequant`. Attention forward
pre-quantizes input once (`quantize_input`), reuses the
`(x_fp4, x_scale, x_global_sf)` tuple for k/v projections when
`context is x` — bit-matches internal's fused path.

## `prepare_for_compile` protocol

[`composed_pipeline_base._maybe_compile_pipeline_module`](file:///home/william5lin/FastVideo/fastvideo/pipelines/composed_pipeline_base.py)
calls `getattr(module, "prepare_for_compile", None)` before invoking
`torch.compile`. Defined as a duck-type protocol — no base class method.

Currently only **Gemma3** implements it (to materialize HF weights
outside Dynamo's tracer). Add to other models that have lazy external
state if you observe compile-time graph breaks.

## Per-component compile flags

`CompileConfig` (in
[`fastvideo/api/schema.py`](file:///home/william5lin/FastVideo/fastvideo/api/schema.py))
gained per-component knobs in `221cb20a`:

```python
@dataclass
class CompileConfig:
    enabled: bool = False                 # master DiT switch
    backend: str = "inductor"
    fullgraph: bool = False
    mode: str | None = None
    dynamic: bool | None = None
    extras: dict = field(default_factory=dict)

    # Per-component overlays, None = inherit master `enabled`
    text_encoder_enabled: bool | None = None
    vae_enabled: bool | None = None
    audio_vae_enabled: bool | None = None

    # Per-component kwargs override master when non-empty
    dit_kwargs: dict = field(default_factory=dict)
    text_encoder_kwargs: dict = field(default_factory=dict)
    vae_kwargs: dict = field(default_factory=dict)
    audio_vae_kwargs: dict = field(default_factory=dict)
```

**`transformer_refine` is auto-compiled with the master DiT flag.** No
separate `enable_torch_compile_refine` flag — by design, refine inherits
DiT compile state to keep typed surface small. Decoupling would add a
new flag, not repurpose existing ones.

## Quantization commit chain (`will/ltx2_sr_port`)

| Commit | Locks in |
|---|---|
| `365a66c7  feat(quantization): upstream LTX-2 FP4Config with lazy flashinfer` | Public colocation of FP4Config (resolves dreamverse_review Q-6 option 1); flashinfer lazy-imported in loader helper, no public hard-dep |
| `a4760bae  fix(api): propagate generic refine_*` | `_resolve_refine_args()` copies generic `refine_*` knobs onto `ltx2_refine_*` runtime carriers; `_randn_ltx2_video_latents` reverts to `torch.randn` to bit-match internal under single-generator inference |
| `221cb20a  feat(api): typed per-component CompileConfig` | `CompileConfig` per-component knobs; matching `FastVideoArgs` carriers; compat layer round-trip |
| `6da342ba  feat(compile): per-component compile + transformer_refine + prepare hook` | `composed_pipeline_base.post_init` compiles `transformer_refine` alongside `transformer`/`transformer_2`; per-component compile loops; `prepare_for_compile` hook on Gemma3 |
| `42b30bf9  feat(ltx2): wire FP4 inference` (largest) | `nn.Linear` → `ReplicatedLinear` for FP4-eligible LTX2 subset; `quant_config` + `prefix=` plumbing; `_maybe_convert_model_to_nvfp4` helper; `LinearBase` fallback to `UnquantizedLinearMethod`; typed `transformer_quant` resolution |
| `94c983a2  refactor(quant): rename FP4 → NVFP4` | Mechanical rename across config, methods, buffers, tests |
| `c6c14c55  test(nvfp4): lock LTX-2 wiring + typed transformer_quant flow` | 6+4 tests in `test_nvfp4_ltx2_wiring.py` + `test_typed_quant_flow.py` |
| `a5fcd19c  [fix]: lazy-import flash_attn 2 fallback in attention backend` | post-handoff: lazy import to avoid hard flash_attn 2 dep |
| `d4ee5be2  [fix]: avoid model.to() round-trip in Gemma encoder forward` | post-handoff: parity / perf fix |
| `156103b9  [fix]: unwrap list-of-generator before torch.randn in LTX-2 latent prep` | post-handoff: parity fix for list-of-generators (was bit-matching only single-generator path) |

## Tests

| Test | Asserts |
|---|---|
| [`fastvideo/tests/ops/quantization/test_nvfp4_ltx2_wiring.py`](file:///home/william5lin/FastVideo/fastvideo/tests/ops/quantization/test_nvfp4_ltx2_wiring.py) (6 tests) | `LTXSelfAttention.to_q/to_k/to_v/to_out` are `ReplicatedLinear`; `NVFP4Config()` attaches `NVFP4QuantizeMethod` on the quantized subset with correct `layer_prefix`; non-tagged projections (cross-attn K/V, audio attn, audio FFN) fall back to `UnquantizedLinearMethod`; `BasicAVTransformerBlock` propagates `quant_config`+`prefix` correctly to all 4 attention modules + FFN |
| [`fastvideo/tests/api/test_typed_quant_flow.py`](file:///home/william5lin/FastVideo/fastvideo/tests/api/test_typed_quant_flow.py) (4 tests) | typed `engine.quantization.transformer_quant: "NVFP4"` → `NVFP4Config()` instance flow; default leaves `transformer_quant` None; explicit `dit_config.quant_config = ...` wins over typed carrier |

CPU-only by design; do NOT exercise actual FP4 kernels (no flashinfer in
CI). Real kernel coverage requires a CI run with flashinfer installed.

## Open follow-ups (quantization-specific)

### #4: Expose `layer_profile` on typed `engine.quantization`

Today `transformer_quant: "NVFP4"` always constructs `NVFP4Config()`
with default `layer_profile="refine"`. To support stage-1 profiles (no
`attn2.to_out`, no cross-modal AV) via typed config, add
`transformer_quant_layer_profile: str | None = None` and thread it
through:

- `fastvideo/api/schema.py` — `QuantizationConfig` field
- `fastvideo/api/compat.py` — typed → flat translation
- `fastvideo/fastvideo_args.py` — `_apply_transformer_quant` consumes it

Dreamverse currently dodges this by setting `NVFP4Config()` directly via
`experimental["pipeline_config"]`. Exposing `layer_profile` removes the
dodge. See [open-threads.md](open-threads.md) #4.

### #5: Typed `dit_config.quant_config` carrier (replace `experimental["pipeline_config"]`)

Long-term: design a typed home for an in-memory `PipelineConfig`
instance with mutated `dit_config`. Today `compat.py` recognizes the
`pipeline_config` key in `experimental` and threads it through to
`FastVideoArgs.from_kwargs`. This is fine for short-term but not pretty.

Heaviest design work in the open queue. May need Oracle consult.

### #2: AbsMaxFP8 pre-existing test failure

`fastvideo/tests/ops/quantization/test_absmax_fp8.py::test_create_weights_rejects_invalid_dtype`
fails on `main` and on `will/ltx2_sr_port` with the same error
(`AssertionError not raised`). Verified via `git stash` that the
failure pre-dates NVFP4 work.

Either:
- Fix the test (`AbsMaxFP8LinearMethod.create_weights` no longer
  asserts on invalid dtype — restore the assert if intentional, or drop
  the test).

Self-contained tech debt; small fix.

## Don't / Cautions

- **Don't change `NVFP4Config` buffer names back to `_fp4_*`.** Rename
  is intentional to disambiguate from MX-FP4 / OCP-FP4.
- **Don't remove the `LinearBase` `UnquantizedLinearMethod` fallback.**
  Load-bearing for non-tagged layers when a `quant_config` is set.
- **Don't repurpose `enable_torch_compile` to mean DiT-only.** It also
  drives `transformer_refine` and `transformer_2` compile.
- **Don't bypass the typed surface for new options.** New compile /
  quant / refine knobs should land on the dataclass + compat.py +
  parity inventory together. The existing test suite locks this in.
- **Don't merge to main without a CI run that covers FP4.** Current CI
  doesn't run flashinfer-dependent paths; the wiring tests are CPU-only
  by design.
