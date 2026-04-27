---
name: review-add-model-pr
description: Use when reviewing a PR that adds a new model / pipeline (or a major variant like I2V/V2V/DMD) to FastVideo under `fastvideo/pipelines/basic/<family>/`. Walks a reviewer through the canonical surface the porter should have touched, the parity bar, and the 36 documented failure modes from prior ports. Returns a structured review verdict (block / nit / follow-up).
---

# Review an add-model PR

## Purpose

Catch the failure modes that have actually shipped in past add-model PRs
(documented in `.agents/skills/add-model/REVIEW.md`) **before merge**,
without re-litigating design choices that the `add-model` skill has
already settled. The reviewer's job is *not* to redesign the port — it
is to verify the porter did the things the skill prescribes, exercised
the parity gate honestly, and surfaced the right knobs to end users.

This skill assumes the PR claims to add a model. For PRs adding only
sampling-preset tweaks, a single pipeline kwarg, or a new test against
an existing pipeline, this skill is overkill — comment scoped to those
specific changes instead.

## When to use

- A new pipeline directory under `fastvideo/pipelines/basic/<family>/`.
- A new variant pipeline (e.g. `<family>_i2v_pipeline.py`,
  `<family>_dmd_pipeline.py`) sibling-added to an existing family.
- A "first-class component" PR that adds a new DiT, VAE, or encoder
  port without (yet) wiring a pipeline (REVIEW item 23).
- Re-review of a port that previously skipped the post-parity hot-path
  pass (REVIEW item 33) or shipped with placeholder diffusers imports
  (REVIEW item 30).

## When not to use

- PR only edits sampling defaults in an existing
  `<family>/presets.py` → review the values against the model card
  inline, no skill needed.
- PR only adds a new SSIM reference → use the
  `seed-ssim-references` skill instead.
- PR refactors shared infra (`fastvideo/layers/`, `fastvideo/attention/`,
  `fastvideo/pipelines/stages/` base classes) without touching a
  family directory → that's not an add-model PR.

## Required reading before starting the review

1. **The PR description.** What family is being added, what variants
   (T2V / I2V / V2V / DMD / T2A / …), what the porter claims is parity-
   verified, and which of the four prereqs (official repo URL, HF
   weights path, HF token, target `model_family`) they collected.
2. **`.agents/skills/add-model/SKILL.md` Files-table** (rows 1–17) — the
   canonical surface a model port touches. You'll cross-reference this
   against the diff in step 2 below.
3. **`.agents/skills/add-model/REVIEW.md` summary table** — the 36
   failure modes. The "Pitfall map" section below indexes them by where
   they typically show up in a diff.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `pr` | Yes | GitHub PR number, URL, or local branch name. |
| `model_family` | Recommended | The snake_case family slug (e.g. `stable_audio`, `magi_human`). Lets you grep-scope the review. |
| `official_repo` | Recommended | URL of the upstream reference, so you can sanity-check the parity test references it correctly. |
| `parity_run_log` | Optional | If the porter attached a parity run log, read it before reading code. |

## Steps

### 1. Verify the PR scope is shaped like an add-model PR

Run `gh pr view <pr> --json files | jq -r '.files[].path'` (or
`git diff --name-only origin/main...HEAD`). Confirm the diff touches
**at least 6 of the 17 Files-table rows** in `add-model/SKILL.md`.
Typical shape:

| If you see... | Expect... | If missing... |
|---|---|---|
| `fastvideo/models/dits/<family>.py` | `fastvideo/configs/models/dits/<family>.py` + export in `__init__.py` | Block: DiT config + export missing (Files-table rows 2 + 3). |
| `fastvideo/pipelines/basic/<family>/<family>_pipeline.py` | `presets.py`, `registry.py` edit, smoke test, parity test, example | Block on whichever is missing — the porter will be told the same by the skill. |
| `<family>_i2v_pipeline.py` (sibling-added) | A new preset for the I2V workload + the registry entry pointing at the I2V HF repo | Block: I2V variants get their own preset + workload tag (REVIEW item 8). |
| Standalone DiT/VAE/encoder under `fastvideo/models/<bucket>/` with no pipeline | A "first-class component" PR (REVIEW item 23) | Confirm with the author that the consuming pipeline PR is in flight or planned. Don't block. |
| Edits under `fastvideo/configs/pipelines/base.py`, `fastvideo/api/sampling_param.py`, `fastvideo/pipelines/pipeline_batch_info.py` | New pipeline-call kwargs the family needs | Sanity-check the field's default doesn't change behavior for other families. Often a footgun (e.g. `0 or default` Python truthiness). |

### 2. Walk the diff in dependency order, not file order

Components have a strict dependency order — review them in this order
so you can spot mismatches as they cascade:

1. **DiT** (`fastvideo/models/dits/<family>.py` + config) — foundational.
2. **VAE** — only one usually exists; if not, this is the second-biggest
   review surface.
3. **Encoder(s) / conditioner** — text encoder is usually shared; new
   conditioners (e.g. `MultiConditioner`-style) need careful look.
4. **`PipelineConfig`** subclass.
5. **Pipeline class** — wires modules + stages.
6. **Stages** — most should be standard; mod-specific subclasses
   warrant careful review.
7. **Presets + registry** — usually mechanical; check workload type
   selection against REVIEW item 28.
8. **Tests** — both component parity and pipeline parity.
9. **Example script** — last, but the bar is high (REVIEW items 31, 35).

For each component, run the Per-component checklist (next section).

### 3. Apply the Per-component checklist

For each new file in `fastvideo/models/<bucket>/<family>.py`:

#### DiT review (`fastvideo/models/dits/<family>.py`)

- [ ] **No raw `nn.Linear`** in QKV/MLP/proj/embedder paths. All
  projections should be `ReplicatedLinear` from
  `fastvideo.layers.linear`. Exceptions are the legacy MoE-packed
  case (REVIEW item 11) — must be flagged in a comment with the
  weight-layout justification.
- [ ] **No raw SDPA / `flash_attn_func`** calls. Self-attention should
  be `DistributedAttention` (or `LocalAttention` for cross-attention).
  See `add-model/SKILL.md` "Attention layers" table.
- [ ] **No raw `nn.LayerNorm` in modulation paths.** Should be
  `FP32LayerNorm` / `RMSNorm` / `LayerNormScaleShift` from
  `fastvideo.layers.layernorm`.
- [ ] **No `from diffusers import` or `from transformers import
  <ModelClass>`** anywhere outside of test files (REVIEW item 30, the
  hard ban). Tokenizers are the *only* allowed `from transformers`
  runtime import in production code.
- [ ] **`from_official_state_dict()` (or equivalent)** is provided so
  the consuming pipeline can load the published checkpoint without
  going through `diffusers.from_pretrained`.
- [ ] **Param-mapping deviations from upstream are localized** — e.g.
  if the model uses `gamma`/`beta` and FastVideo's `FP32LayerNorm`
  uses `weight`/`bias`, the remap happens *in the loader*, not by
  renaming the layers.
- [ ] **Partial / non-standard rotary** (e.g. halves-swap vs
  interleaved-pair) is kept local with a one-line WHY comment, not a
  copy of FastVideo's `_apply_rotary_emb` with edits.

#### VAE review (`fastvideo/models/vaes/<family or arch>.py`)

- [ ] **File name matches the convention** (REVIEW item 29): name
  after the *arch* if the VAE is shared across families
  (`oobleck.py`, `autoencoder_kl.py`); name after the *family* if it's
  specific (`wanvae.py`).
- [ ] **`from_pretrained()` (or `from_official_state_dict`)** loads
  weights from the published HF path *without* a Diffusers
  intermediary (REVIEW item 30 again).
- [ ] **Per-channel `latents_mean`/`latents_std` are reshaped with
  explicit `.view(1, -1, 1, 1, 1)`** when applied (REVIEW item 22) —
  silent broadcasting along the wrong dim is a stealth bug.
- [ ] **Normalization-convention mismatches with the wrapper**
  (REVIEW item 20) — if upstream `decode()` does
  `z = z*std + mean` *internally* but the FastVideo wrapper expects
  pre-denormalized input, the parity test must compensate or the
  decode parity will look broken.
- [ ] **Pipeline-glue wrapper** (e.g. `fastvideo/models/vaes/<family>_audio.py`
  for lazy-load semantics) exists if the pipeline needs lazy-load /
  hide-from-named_parameters semantics (REVIEW item 26).

#### Conditioner / encoder review

- [ ] If the conditioner mixes text + numeric (e.g. duration), it
  produces the **DiT-ready (cross_attn_cond, cross_attn_mask,
  global_embed) triple** in a single helper, not via inline cat-ing
  in the conditioning stage.
- [ ] **T5 / Llama / SigLIP TP wiring** uses the encoder bucket's TP
  primitives (`QKVParallelLinear`, `MergedColumnParallelLinear`,
  `RowParallelLinear`) — not `ReplicatedLinear`.
- [ ] If the conditioner intentionally hides its T5 from
  `named_parameters()` (so the SA-style checkpoint loader doesn't try
  to match upstream-absent T5 keys), that exclusion is documented in
  a comment.

#### `PipelineConfig` review (`fastvideo/configs/pipelines/<family>.py`
or `fastvideo/pipelines/basic/<family>/pipeline_configs.py`)

- [ ] **Subclasses `PipelineConfig`** from
  `fastvideo.configs.pipelines.base`.
- [ ] **`vae_config`, `dit_config`, `text_encoder_configs`** are
  defaulted via `field(default_factory=...)` (mutable-default-arg
  Python rule).
- [ ] **The text-encoder slot** matches what the pipeline actually
  uses. If the pipeline owns its own conditioner (no FastVideo-loaded
  text encoder), the text-encoder tuples should be `tuple()` and the
  parent's length-equality validator should still pass — the porter
  must zero out *all four* of `text_encoder_configs`,
  `text_encoder_precisions`, `preprocess_text_funcs`,
  `postprocess_text_funcs` together (we've seen this break before).
- [ ] **Component-bucket inheritance** (REVIEW item 24) — if the new
  config goes under `vaes/`, it must subclass `VAEConfig` not
  `EncoderConfig`. The bucket directory determines the base.
- [ ] **`__post_init__`** is used only to flip `load_encoder` /
  `load_decoder` / etc. on the child configs, not to do any heavy
  build.

#### Pipeline class review (`fastvideo/pipelines/basic/<family>/<family>_pipeline.py`)

- [ ] **`EntryClass` is a single class**, not a list.
- [ ] **`_required_config_modules`** lists exactly what the loader
  reads from `model_index.json`. Missing keys cause silent loading
  degradation.
- [ ] **`load_modules()`** does not have any `from diffusers import`
  / `from transformers import <ModelClass>` for production
  components. If the porter explicitly opted into a temporary
  diffusers shim, REJECT — the right move is to ship the native port
  or hold the pipeline back (REVIEW item 30).
- [ ] **`torch.backends.*` flags** (TF32, cuDNN benchmark) — if set,
  they're set **once in `load_modules`**, not per-call inside a stage
  (REVIEW item 33). Mid-run flips invalidate the cuDNN algorithm
  cache and amplify A2A SDE drift.
- [ ] **`create_pipeline_stages()`** uses standard stages from
  `fastvideo.pipelines.stages` where possible, only subclassing when
  the math diverges. New stage classes live in
  `fastvideo/pipelines/basic/<family>/stages/` and are re-exported in
  `fastvideo/pipelines/stages/__init__.py`.
- [ ] **One pipeline class for kwargs-driven variants** (REVIEW item
  34) — T2A/A2A/inpaint that share weights/components shouldn't be
  split into three classes. Triggers to split: separate
  `_required_config_modules`, separate HF repo, divergent forward
  signatures, separate `WorkloadType`. Reject the split unless one
  of those applies.

#### Stages review (`fastvideo/pipelines/basic/<family>/stages/*.py`)

- [ ] **No `init_audio_strength = 0` / "0 or default" footguns** —
  Python truthy-or fallbacks (`x or default`) silently swallow `0`,
  `0.0`, `""`. Use `x if x is not None else default`.
- [ ] **Loud-fail on malformed kwarg combos** — e.g. inpaint without
  mask should raise `ValueError`, not silently fall through to T2A.
- [ ] **Hot-path discipline** (REVIEW item 33): no per-step
  `torch.zeros_like(...)` or `torch.randn_like(...)` inside the
  sampler loop; pre-allocate buffers + reuse via `.normal_()` /
  `.copy_()`.
- [ ] **No dead `batch.extra` writes** — every key written should be
  read by a downstream stage.
- [ ] **Stage docstring** is one or two lines. No "Mirrors upstream X"
  / "Vendored from Y" provenance (REVIEW item 36); upstream
  comparison belongs in the parity test, not the production docstring.

#### Presets + registry review

- [ ] **Sampling defaults** match the published model card example
  block. Track the *model* defaults, not the upstream library's
  *generic* defaults (this distinction shipped a 100% drift before
  for Stable Audio).
- [ ] **`workload_type`** matches what the model actually does. If
  the family is audio (T2A/A2A/AV) and `WorkloadType` doesn't yet
  have those values (REVIEW item 28), accept `"t2v"` as a placeholder
  but require a TODO comment + the porter to file a follow-up.
- [ ] **Registry detector** matches both the HF path and the
  pipeline class name (`_class_name` from `model_index.json`).
- [ ] **`ALL_PRESETS`** export exists and is added to
  `_register_presets()`'s group tuple in `fastvideo/registry.py`.

#### Tests review — **the most load-bearing review surface**

REVIEW item 16 is the single most expensive failure mode. A skipped
test reads as green in CI. Confirm explicitly:

- [ ] **Component parity tests exist** for every non-reused
  component: DiT, VAE (if new), encoder (if new). Find them under
  `tests/local_tests/<bucket>/test_<family>_*.py`.
- [ ] **Each component parity test produces a non-skip pass on the
  reviewer's machine** if at all possible. If the porter says "I ran
  it locally", ask for the diff numbers in the PR description.
- [ ] **Pipeline parity test exists** at
  `tests/local_tests/pipelines/test_<family>_pipeline_parity.py` AND
  has a non-skip pass — *not* a "skipped because the official clone
  is missing" pass.
- [ ] **Smoke test exists** at `test_<family>_pipeline_smoke.py` — no
  GPU, just import + registry + preset wiring. CI can run this even
  if local-only parity tests skip.
- [ ] **Parity reference is the official upstream**, not diffusers
  (REVIEW item 30). Diffusers parity is acceptable as a *secondary*
  test only when (a) the published weights load through both and (b)
  the official repo is also imported and compared.
- [ ] **Tolerances are scope-appropriate** (REVIEW item 21): single-
  block + single-kernel = `atol=1e-4`; full-DiT cross-kernel = `0.1`
  with a complementary `abs_mean drift < 5%` check; bare `assert_close`
  alone with a loose tolerance is a smell.
- [ ] **Stub helpers for upstream private DSL deps** (REVIEW items
  17, 18, 32) — if the upstream has `magi_compiler`-style imports
  that aren't on PyPI, a small `tests/local_tests/helpers/<family>_upstream.py`
  shim is fine. But if the deps it bypasses become real installs,
  the shim must be deleted (item 32 — no zombie no-op shims).
- [ ] **GQA-aware kernel routing in stub paths** (REVIEW item 19) —
  if the parity test routes upstream's flash_attn through SDPA, KV
  heads must be `repeat_interleave`'d explicitly.
- [ ] **VAE normalization symmetry** (REVIEW item 20) — if upstream
  bundles `z = z*std + mean` in `decode()` and FastVideo expects
  pre-denormalized, the test must compensate explicitly.

#### Example script review (`examples/inference/basic/basic_<family>*.py`)

The bar here is high — the example is the user's entry point, not the
porter's debugging script.

- [ ] **User-story-shaped docstring** (REVIEW item 31) — at least one
  `User story (<persona>):` block, then a "How it works" / "Picking
  the dial" / "Tunable knobs" section. Not a code-narration docstring.
- [ ] **5-15 LOC of constants + one `generate_video()` call**
  (REVIEW item 35). If the example carries a 25-line `_load_reference`
  / shape-norm / resample helper, that glue belongs *in the
  pipeline*, not the example.
- [ ] **Pipeline accepts file paths** for any media-input kwarg
  (`init_audio`, `inpaint_audio`, `image_path`). The example just
  passes the path; pipeline does decode + resample internally.
- [ ] **No `torchaudio.load` on container formats** (mp4, m4a) —
  routes through `torchcodec` → CUDA NVRTC. PyAV (already a
  FastVideo dep, used by `_mux_audio`) handles all formats.
- [ ] **Tunable-knob defaults** match the model's published sweet
  spot, not the upstream library's generic defaults.

#### Local repro doc review (optional but recommended)

If the family ships a `tests/local_tests/<family>.md`:

- [ ] **Setup section** covers HF gated access, optional inference
  deps, upstream clone instructions, model cache pre-warm.
- [ ] **Per-test table** explains what each test compares against
  with expected drift numbers.
- [ ] **Troubleshooting section** covers gated-skip behavior, batch-
  vs-single-run flag interactions (cuDNN benchmark!), first-call
  cache download blowup.

### 4. Cross-check the "first-class component" rule

Run `grep -rn "from diffusers import\|from transformers import"
fastvideo/pipelines/basic/<family>/ fastvideo/models/dits/<family>*
fastvideo/models/vaes/<family>* fastvideo/models/encoders/<family>*`.

The **only** acceptable hits are `from transformers import
<TokenizerFast>` (data-utility, no weights) and `from transformers
import T5EncoderModel` *only if* (a) it's loaded via HF and (b) the
T5 weights are absent from the model's checkpoint by design (e.g.
SA conditioner).

Anything else — `StableAudioDiTModel`, `AutoencoderKL`,
`UnetXxx`, `T2VPipeline` — is a REVIEW item 30 violation. Block
the PR with a pointer to that item.

### 5. Run the parity tests yourself if budget permits

The porter said it passes. Confirm:

```bash
# DiT/VAE/encoder component parity
pytest tests/local_tests/<bucket>/test_<family>_*.py -v -s

# Pipeline parity (the gate per add-model step 13(a))
pytest tests/local_tests/pipelines/test_<family>_pipeline_parity.py -v -s

# Smoke (no GPU)
pytest tests/local_tests/pipelines/test_<family>_pipeline_smoke.py -v
```

If any *parity* test SKIPs rather than PASSes on your machine and
you have the prerequisites set up, that means the porter never
actually verified parity (REVIEW item 16). Block.

### 6. Read REVIEW.md for any new failure modes the porter didn't address

The PR may also amend REVIEW.md with newly-discovered failure modes.
Skim those — they're typically the most accurate signal of what to
look for in *this specific* port. If the porter added a REVIEW item
and the linked code in the PR doesn't actually mitigate that item,
that's a contradiction — flag it.

### 7. Write the verdict

Structure your review comment as three buckets:

- **Block (must fix before merge)** — REVIEW item 30 violations,
  skipped parity tests, missing required Files-table rows, dead
  `batch.extra` writes that hide caller bugs, footguns like `0 or
  default`.
- **Nit (would be better)** — naming convention drift, narrative
  comments per REVIEW item 36, missing user-story docstrings,
  hot-path allocations that don't change correctness.
- **Follow-up (not this PR)** — extracting test boilerplate to a
  shared helper, future variant pipelines, performance benchmarks.

Always link each item back to the `add-model/REVIEW.md` item number
when applicable; that's the institutional memory and lets the porter
fix the issue with full context.

## Pitfall map (REVIEW.md items by where they show up in a diff)

Use this when you've spotted something off and want to find the
documented failure mode it corresponds to.

| If the diff has... | Suspect REVIEW item(s) |
|---|---|
| `from diffusers import <ModelClass>` in `fastvideo/...` | **30** (hard ban) |
| Raw `nn.Linear` in DiT projections | 10, 11 (only acceptable for MoE-packed weights with comment) |
| Custom RoPE that's *not* `_apply_rotary_emb` | Probably fine if the convention differs (e.g. halves-swap), but require a one-line WHY comment + a parity-test row |
| `parity_test.py` that calls `pytest.skip` unconditionally | **16** (silent no-op trap) — block |
| Component parity tests missing | **16** — block |
| New pipeline file with no `register_configs` call in registry.py | Files-table row 11 missing |
| `_required_config_modules` that doesn't match `model_index.json` keys | Silent loading degradation; block |
| `text_encoder_configs=tuple()` but other text-encoder tuples non-empty | Length-equality validator will fail at runtime |
| ArchConfig has `num_inference_steps` / `guidance_scale` / `flow_shift` | **15a** — pipeline-level fields leaking into ArchConfig; block |
| `<family>vae.py` for an arch-shared VAE (e.g. Oobleck) | **29** — should be `<arch>.py` (e.g. `oobleck.py`) |
| Pipeline imports `from transformers import T5EncoderModel` | OK only if the conditioner intentionally hides T5 from `named_parameters()` and the checkpoint omits T5 keys; document the exception |
| `torch.backends.*` set in a stage's `forward()` | **33** — should be one-shot in `load_modules` |
| `init_X = 0` silently treated as default via `or` | **33-adjacent** — Python truthy footgun; flag |
| Per-step `torch.zeros_like` / `randn_like` in sampler callback | **33** — pre-allocate; can also fix accuracy regressions |
| Magic `_DOWNSAMPLING_RATIO = 2048` in stage code | **33** — derive from the VAE config |
| `examples/.../basic_<family>*.py` >30 LOC | **35** — decode/resample/shape-norm belongs in the pipeline |
| Example uses `torchaudio.load(...)` on mp4 | **35** — torchcodec → NVRTC dep chain; use PyAV |
| Example docstring narrates code | **31** — needs `User story (<persona>):` block |
| Pipeline class docstring narrates upstream provenance | **36** — strip "Vendored from X" / "Mirrors upstream Y" |
| Comment says "previously this was X, we moved it because Y" | **36** — strip; goes in commit message, not code |
| Diff adds `tests/local_tests/helpers/<family>_upstream.py` that's a no-op | **32** — delete the no-op shim + its call sites |
| Diff adds `init_audio` / `inpaint_audio` / similar to `SamplingParam` | OK; verify they default to `None` and other families ignore |
| Multiple pipeline classes that share weights/components | **34** — should be one class with kwargs-driven modes |
| Config under `fastvideo/configs/models/encoders/` for a VAE | **24** — wrong bucket; should be under `vaes/` with `VAEConfig` base |
| New pipeline that depends on a not-yet-ported component (placeholder import) | **30** — hold the pipeline back until the component is ported |

## Outputs

A structured review comment on the PR with:

1. **Verdict** — `approve` / `request-changes` / `comment-only`.
2. **Block list** — REVIEW-item-linked issues that must be fixed.
3. **Nit list** — style / convention drift.
4. **Follow-up list** — items the porter should know about but
   shouldn't fix in this PR.
5. **Parity numbers you confirmed** — diff_max / diff_mean / drift /
   element-wise bound, per parity test you ran. Lets the next reviewer
   skip re-running.

## Example review comment skeleton

```
## Review summary

**Verdict:** request-changes

I ran the smoke + DiT-component parity tests and read the full diff.
The native ports look right; two REVIEW-30 violations in the pipeline
file need resolving before merge, plus a few smaller nits.

### Block (must fix)

- `fastvideo/pipelines/basic/<family>/<family>_pipeline.py:NN` — `from
  diffusers import <ModelClass>` violates REVIEW item 30 (hard ban).
  Either ship the first-class port now or hold this pipeline back
  until the component is ported.
- `tests/local_tests/pipelines/test_<family>_pipeline_parity.py` skips
  on my machine even with HF token set (the upstream-clone path check
  fails). REVIEW item 16: a parity test that always skips is worse
  than no test. Update the path resolution or document it in
  `tests/local_tests/<family>.md`.

### Nit

- `examples/inference/basic/basic_<family>.py` docstring is code-narration;
  REVIEW item 31 wants a `User story (<persona>):` block.
- `<family>_pipeline.py:NN` has a per-call `torch.backends.cuda.matmul.allow_tf32 = False`;
  REVIEW item 33 says move to one-shot in `load_modules`.

### Follow-up

- HF-token boilerplate is duplicated across 4 parity files. REVIEW
  item 1-tier work, but doesn't need to land in this PR.

### Parity numbers I confirmed locally

| Test | diff_max | diff_mean | drift |
|---|---|---|---|
| DiT component parity | 0.0 | 0.0 | bit-identical |
| VAE decode parity | 0.0 | 0.0 | bit-identical |
| Pipeline parity (T2V, 25 steps) | 0.012 | 0.0009 | 0.31% |
```

## References

- `.agents/skills/add-model/SKILL.md` — the canonical procedure;
  every "should be there" claim in this skill is a row in that
  skill's Files-table.
- `.agents/skills/add-model/REVIEW.md` — 36 documented failure modes;
  the Pitfall map above indexes them by where they show up in a diff.
- `.agents/skills/seed-ssim-references/SKILL.md` — for SSIM regression
  add-on PRs (separate from this skill's scope).
