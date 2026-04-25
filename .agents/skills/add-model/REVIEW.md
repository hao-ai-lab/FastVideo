# add-model skill — open items for review

Generated during the Wan2.1 / Wan2.2 verification pass on 2026-04-24.

Each item below is something the skill currently says or implies that I
could not fully resolve from the Wan code alone. Please triage. The
skill has been updated with what was verifiable; these are the
judgment calls I deferred.

---

## 1. No Wan parity or smoke tests exist in the repo

**Where in the skill:** Step 12 (smoke-test), step 13(a) (mandatory
pipeline parity test), step 6 (per-component parity tests).

**What I found:** `tests/local_tests/` has parity tests for LTX-2,
GameCraft, Gen3C, Kandinsky5 — but **none for Wan**. No
`test_wan_pipeline_smoke.py`, no `test_wan_pipeline_parity.py`, no
`tests/local_tests/transformers/test_wan*.py`,
`tests/local_tests/vaes/test_wan*.py`, or
`tests/local_tests/encoders/test_wan*.py`.

**Why this matters:** The skill calls Wan out as the canonical
reference family, but the "every ported family ships pipeline +
component parity tests" convention the skill enforces is not actually
observed for Wan in the current tree.

**Options:**
- (a) Keep the skill strict for *new* ports — Wan predates the
  convention, call it out explicitly as grandfathered.
- (b) Relax the skill — make parity tests strongly recommended but not
  a hard gate.
- (c) Add Wan parity tests retroactively (separate work item, not
  covered by the skill).

**My guess:** (a) is closest to your intent given how firmly the skill
currently phrases step 13(a) as a handoff gate — but flagging it so
you decide.

---

## 2. `wan_to_diffusers.py` is a legacy conversion-script shape

**Where in the skill:** "Weight conversion" section — "Conversion
script conventions" and the reference-scripts table.

**What I found:** `scripts/checkpoint_conversion/wan_to_diffusers.py`
has **no argparse CLI**. It's a module of regex mapping dicts
(`_param_names_mapping`,
`_self_forcing_to_diffusers_param_names_mapping`) imported by other
code, not a runnable script. The modern pattern
(`convert_ltx2_weights.py`) is clearly different: argparse CLI,
`PARAM_NAME_MAP` + `COMPONENT_PREFIXES`, synthesized `model_index.json`.

**What I did in the skill:** Marked `wan_to_diffusers.py` explicitly
as "legacy pattern — do not copy the module shape" and continued to
point at `convert_ltx2_weights.py` as the template.

**Open question:** Is `wan_to_diffusers.py` still imported somewhere,
or is it actually dead? If dead, consider deleting it. If live,
document where it's invoked (e.g. which caller imports those dicts).

---

## 3. `ConditioningStage` — optional or always-present?

**Where in the skill:** "Standard stages — subclass targets" and
(previously) the canonical-order diagram in "FastVideo's single
architecture".

**What I found:** Both `WanPipeline` (T2V) and
`WanImageToVideoPipeline` (I2V) always add `ConditioningStage`
unconditionally. No Wan variant skips it. Haven't audited every
other family.

**What I did in the skill:** Changed the canonical-order diagram to
call it a default rather than optional.

**Open question:** Is `ConditioningStage` skippable for any existing
family (LTX-2, SD3.5, Cosmos2.5, etc.), or is it universal? If
universal, the skill could go further and remove "optional" framing
everywhere. If sometimes-skippable, we should list which families
skip it.

---

## 4. Image-encoder selection for I2V

**Where in the skill:** New "Adding an I2V variant" section, "Config
delta" subsection.

**What I found:** Wan I2V 480P/720P use stock `CLIPVisionConfig`. Wan
V2V swaps to `WAN2_1ControlCLIPVisionConfig` (a Wan-specific variant).
I didn't verify what other I2V families in the repo use (Cosmos25,
MatrixGame) — some likely use non-CLIP image encoders.

**What I did in the skill:** Defaulted the I2V config example to
`CLIPVisionConfig` and noted that non-standard variants swap the
`image_encoder_config` default (with WANV2V as example).

**Open question:** Should the skill provide a selection rule
("prefer CLIP unless the reference repo uses X")? I don't have enough
signal across families to write one confidently.

---

## 5. Wan2.2 MoE / dual-guidance fields not covered in the skill

**Where in the skill:** No dedicated section.

**What I found:** Wan2.2 A14B configs and presets introduce:
- `boundary_ratio` (MoE routing parameter).
- `dmd_denoising_steps` list (for DMD variants).
- `guidance_scale_2` (second guidance scale for dual-CFG).
- `transformer_2` module — referenced in both T2V and I2V pipelines
  via `self.get_module("transformer_2", None)` (T2V) or
  `self.get_module("transformer_2")` (I2V, no default).

**What I did in the skill:** Mentioned `transformer_2` in the pipeline
class example comments but did not add a dedicated MoE section.

**Open question:** How broad is MoE / dual-guidance adoption going
forward? If mostly a Wan2.2 peculiarity, leaving it out keeps the
skill focused. If a generalizable pattern, it deserves a section on
par with I2V. My guess is Wan-specific for now, but flag.

Sub-question: The I2V Wan pipeline calls
`self.get_module("transformer_2")` without a default, while T2V uses
`self.get_module("transformer_2", None)`. That inconsistency looks
like a real bug in `wan_i2v_pipeline.py:66` that would hard-error for
non-MoE I2V families. Not in scope for the skill, but worth noting
separately.

---

## 6. `WanT2VCrossAttention` / `WanI2VCrossAttention` pattern name

**Where in the skill:** New "Adding an I2V variant" section, "DiT
branching" subsection.

**What I found:** Wan names its two cross-attention variants
`Wan{T2V,I2V}CrossAttention` and switches based on
`added_kv_proj_dim is None`. I used `<Family>{T2V,I2V}CrossAttention`
as a placeholder in the skill.

**Open question:** Should the skill prescribe this naming convention
for new families, or leave it to author preference? Other families
(Cosmos, Hunyuan) may use different naming.

---

## 7. Missing CI coverage for the pipeline-parity gate

**Where in the skill:** Step 13(a) insists on a green pipeline parity
test before handoff.

**What I found:** No CI job I can identify that runs
`tests/local_tests/pipelines/test_*_pipeline_parity.py`. These tests
are local-only (hence `local_tests/`).

**Implication:** The gate the skill enforces is author-run, not
CI-enforced. That's fine, but worth being explicit — once the author
hands off, only the SSIM tests provide ongoing regression coverage.

**Open question:** Should the skill say "local-only gate" explicitly?
Or is that obvious from the directory name? I left it implicit.

---

## 8. Files-table coverage of variant pipelines

**Where in the skill:** "Files you will create or touch" table
(rows 8, 13).

**What I found:** A full Wan port touches multiple sibling pipeline
files (`wan_pipeline.py`, `wan_i2v_pipeline.py`, `wan_v2v_pipeline.py`,
`wan_dmd_pipeline.py`, `wan_causal_pipeline.py`,
`wan_causal_dmd_pipeline.py`, `wan_i2v_dmd_pipeline.py`) — seven in
total for Wan. The skill's table implies one pipeline file per family.

**What I did in the skill:** Added a note on row 8 that variants get
sibling files. Did not enumerate every possible variant kind.

**Open question:** Should the skill enumerate the common variant
buckets (T2V, I2V, V2V, DMD, Causal, Causal-DMD, SR/Refine) up front,
or keep that implicit? If enumerated, it makes the skill longer but
much more concrete for a porter staring at Wan.

---

## 9. Conversion-script publish step assumes internet + HF write

**Where in the skill:** "Weight conversion" recipe step 10
("Publish to HF via create_hf_repo.py").

**What I found:** Works, but requires the HF token collected in
step 1 to have both read (to `snapshot_download` the base diffusers
repo) and write (to `push_to_hub`) scope. Step 1 currently asks only
about write scope.

**What I did in the skill:** Skill step 1 already asks for
write-enabled; HF write tokens typically include read. Not changed.

**Open question:** Is there a scenario in the Anthropic-style
organization where write tokens are scoped narrowly enough to block
`snapshot_download`? If so, step 1 should mention "read + write" to
be unambiguous.

---

## 10. How strict is the "no raw nn.Linear" rule?

**Where in the skill:** Common pitfalls #11.

**What I found:** A handful of DiTs still have raw `nn.Linear` calls:
- `wanvideo.py:608` (`proj_out`).
- `causal_wanvideo.py:382` (`proj_out`).
- `cosmos.py:28, 50, 52, 90` (patch embed, modulation).
- `hunyuanvideo15.py:141-162, 574` (timestep/context embedders).
- `lingbotworld/model.py:42-43, 294` (cam layers, proj_out).

**What I did in the skill:** Described these as "legacy, do not copy"
and told new code to default to `ReplicatedLinear`.

**Open question:** Is there an explicit policy that new code must
use `ReplicatedLinear` everywhere, or is the occasional `nn.Linear`
(for a single final projection) accepted? If accepted in some cases,
the skill should say where.

---

---

## 11. Packed-expert linears (MoE / multi-modality DiTs)

**Where in the skill:** "Linear layers" section — "DiTs use ReplicatedLinear
everywhere" rule.

**What I found (during the MagiHuman port, 2026-04-24):** MagiHuman's DiT
packs per-modality weight chunks along the output axis of every Linear in
its "mm-layers": `weight.shape = [out_features * num_experts, in_features]`,
with a runtime `ModalityDispatcher` that permutes tokens so each chunk sees
a contiguous slice before running `F.linear` per expert. `ReplicatedLinear`
does not model this layout — its quant-method dispatch path assumes a
single `[out, in]` weight. I had to implement a custom `PackedExpertLinear`
using raw `nn.Parameter` (documented deviation) to keep weight loading
byte-exact against the upstream checkpoint.

The skill currently says (pitfall #11, pitfall #12) "always use
`ReplicatedLinear` in DiTs" with no named exceptions. This is genuinely
infeasible for MoE-style weight layouts where the output axis is
multi-expert-packed.

**Options:**
- (a) Add a named exception to the linear rule: "packed-expert /
  per-modality / MoE-along-output weight layouts are an accepted
  deviation — use raw `nn.Parameter` + a small wrapper. Document the
  deviation in the DiT module docstring." Applies to MagiHuman; may apply
  to future MoE DiTs.
- (b) Extend `ReplicatedLinear` to support `num_experts > 1` directly
  (bigger change; rewires the quant path).
- (c) Leave as-is and let each new porter rediscover the workaround.

**My guess:** (a). The exception is rare but real; naming it saves the
next porter the same debugging loop.

---

## 12. VAE reuse: prefer the `-Diffusers` suffix variant

**Where in the skill:** "Weight conversion" — `_bundle_*_vae` helper
pattern and the "when to reuse a VAE" advice.

**What I found (during the MagiHuman port):** I defaulted the
`_bundle_wan_vae` helper to `Wan-AI/Wan2.2-TI2V-5B` (the canonical repo
id) and it failed with `No vae/ subdir in Wan-AI/Wan2.2-TI2V-5B`. That
repo ships the official non-Diffusers layout:

    Wan2.2_VAE.pth                                    # monolithic .pth, no subdir
    diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors  # DiT, not VAE
    google/umt5-xxl/...                               # text encoder stuff

The Diffusers-layout repo is the *separately-named*
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, which has the expected
`vae/config.json` + `vae/diffusion_pytorch_model.safetensors`. This
`-Diffusers` suffix convention is consistent across Wan (`Wan2.1-T2V-14B`
vs `Wan2.1-T2V-14B-Diffusers`, etc.) but the skill doesn't surface it.

**Options:**
- (a) Add a one-liner to the "Weight conversion" recipe:
  *"When bundling a VAE from another family, point at the `-Diffusers`
  suffix variant (e.g. `Wan-AI/Wan2.2-TI2V-5B-Diffusers`), not the
  canonical repo — the canonical repo often ships a monolithic `.pth`
  at the root, not a `vae/` subdir."*
- (b) Add a lookup helper / `_find_vae_repo(family)` that tries
  `{repo}-Diffusers` first.

**My guess:** (a). One sentence, prevents a 10-min debugging loop.

---

## 13. HF token needed at conversion time, not just publish time

**Where in the skill:** Step 1 prerequisite — HF write token.

**What I found (during the MagiHuman port):** Step 1 phrases the HF
token requirement as: *"write-scoped, used for `create_hf_repo.py
--push_to_hub` and SSIM reference uploads."* But T5-Gemma (MagiHuman's
text encoder, gated Google repo) requires the token for *read-only*
`snapshot_download` during conversion. Same for Gemma3 (LTX-2),
Llama-family encoders, and any future gated encoder. Without the token,
conversion step fails at `401 Unauthorized` on `text_encoder/` bundling
or at first-forward lazy-load.

Write tokens usually include read, so this hasn't actually broken
anything — but the skill's framing is misleading. A porter who only
sees "write for publish" might think the token is needed only at the
very end.

**Options:**
- (a) Reword step 1: *"read-scoped is strictly required at conversion
  time (for gated text encoders). Write-scoped is required at publish
  time. In practice, a single write token satisfies both."*
- (b) No change — trust that write tokens always include read.

**My guess:** (a). Small rewording, much clearer to a first-time porter.

---

## 14. Cross-modality flat-stream self-attention

**Where in the skill:** "FastVideo layers and attention" section —
attention-layer selection table.

**What I found (during the MagiHuman port):** MagiHuman's DiT performs
joint self-attention over a *flat concatenated stream* of video + audio
- text tokens in every layer, with a `ModalityDispatcher` permute/unpermute
around each linear. There is no clean "spatial sequence" to SP-shard the
way `DistributedAttention` expects — the attention needs to see all three
modalities at once. The skill's selection rule ("DiT full-sequence
self-attention → `DistributedAttention`") doesn't cleanly apply.

For the first port I fell back to `F.scaled_dot_product_attention`
directly and left multi-GPU SP as a follow-up. But this raises the
question: is there a principled way to combine `DistributedAttention`
with a flat concat stream?

**Options:**
- (a) Extend the attention table with a third row:
  *"Cross-modality flat-stream self-attention (video + audio + text
  concatenated in one sequence) — use raw SDPA for single-GPU; SP
  integration is a known gap."*
- (b) Write an adapter that teaches `DistributedAttention` about
  modality-aware sharding (bigger project, not skill scope).

**My guess:** (a). Naming the gap is more useful than pretending the
existing table covers this case.

---

## 15a. ArchConfig scope = the DiT's `transformer/config.json` only

**Where in the skill:** Step 2 ("Study the reference implementation"), the
DiT config section, and the "Weight conversion" recipe.

**What I found (during the MagiHuman port, after direct user
correction):** The FastVideo arch config class (e.g. `MagiHumanArchConfig`)
and the `transformer/config.json` emitted by the conversion script **must
match key-for-key**. FastVideo's `TransformerLoader.load` reads
`transformer/config.json`, pops `_class_name`/`_name_or_path`/`_diffusers_version`,
then calls `ArchConfig.update_model_arch(remaining_dict)` which hard-errors
on any key that isn't a declared arch-config field.

I initially stuffed pipeline-level defaults (`num_inference_steps`,
`video_txt_guidance_scale`, `cfg_number`, `flow_shift`), data-proxy knobs
(`vae_stride`, `frame_receptive_field`, `coords_style`, `ref_audio_offset`,
`text_offset`), and eval knobs (`fps`, `t5_gemma_target_length`) onto both
the ArchConfig *and* the emitted `transformer/config.json`. Load failed
with `AttributeError: ArchConfig has no field 'tread_config'` (the nested
dict from the official source) and later would have failed on every
pipeline-level knob.

**Principle (now consistent with Wan):** ArchConfig fields = exactly what
the `transformer/config.json` shipped in the Diffusers repo carries. Everything
else (sampling schedule, CFG scales, VAE stride, FPS, text-embed target
length, data-proxy knobs) lives on the `PipelineConfig` subclass
(`MagiHumanT2VConfig` here).

**Where does the arch source-of-truth come from?** Two tiers:

1. **Preferred:** the `transformer/config.json` inside the Diffusers-format
   HF repo (e.g. `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json`).
   Use its keys verbatim as the arch config fields.
2. **Fallback:** when the HF repo is not in Diffusers format and the
   root `config.json` is empty / missing (as with `GAIR/daVinci-MagiHuman`
   — `config.json` is literally `{}` and the per-subfolder files are just
   `model.safetensors.index.json`), use the **official Python reference**
   as source of truth. For MagiHuman that is
   `inference/common/config.py`'s `ModelConfig` (not `DataProxyConfig`, not
   `EvaluationConfig`).

   The conversion script then emits a synthesized `transformer/config.json`
   mirroring only the `ModelConfig` fields; anything in `DataProxyConfig` /
   `EvaluationConfig` lives on the pipeline config, never the arch.

**Options:**
- (a) Add a paragraph to step 2 ("Study the reference"): *"Before writing
  the arch config, locate the canonical per-subfolder `transformer/config.json`
  — either the HF Diffusers repo or, when the HF repo is raw/empty, the
  official Python `ModelConfig` equivalent. Arch config fields must
  match this 1:1; pipeline-level knobs (steps, CFG, flow_shift, sampling
  defaults) go on `PipelineConfig` instead."*
- (b) Add a hard pitfall: *"Pitfall: pipeline-level fields leaking onto
  the arch config. Every field on the arch config must appear in the
  emitted `transformer/config.json` and vice versa. If it's not a
  transformer-architecture field (steps, CFG scales, VAE stride, fps,
  data-proxy knobs), put it on the `PipelineConfig` instead."*

**My guess:** both. Plus a short example cross-referencing Wan's
`transformer/config.json` (17 keys, all DiT-arch) next to
`WanVideoArchConfig` (same 17 keys + `boundary_ratio` / `exclude_lora_layers`
FastVideo additions).

---

## 15. Scoping a port when upstream has multiple variants

**Where in the skill:** No dedicated section.

**What I found (during the MagiHuman port):** Upstream had four
checkpoints (`base`, `distill`, `540p_sr`, `1080p_sr`) plus an audio
VAE and a turbo VAE decoder. The user said *"start with base model
ONLY"* and I had to interpret what that meant — keep the DiT's audio
path (since joint denoise is the trained behavior) but skip audio
decoding? Skip SR entirely? Write a single pipeline config or one per
variant? No guidance from the skill.

**Options:**
- (a) Add a "Scoping a port" paragraph to the "When to use" section:
  *"If upstream ships multiple variants (distill, SR, I2V, audio),
  agree with the user up front on which belong in the first PR. The
  common narrow scope is **base T2V only** — one pipeline class, one
  preset, no audio decode, no SR, no distill. Variants are added in
  follow-up PRs using the I2V pattern (sibling pipeline file + config
  subclass + second preset + second register_configs call)."*
- (b) Leave as-is; scoping is a per-port conversation.

**My guess:** (a). One paragraph avoids a common source of scope
ambiguity at port-start.

---

## 16. The "skip-on-missing" parity pattern is a silent no-op trap

**Where in the skill:** "Parity test pattern" section — "Skip-on-missing,
never fail-on-missing" convention. Also step 6 (component parity) and
step 13(a) (pipeline parity).

**What happened (during the MagiHuman port, user-corrected):** I produced
the full scaffold — model code, configs, pipeline, presets, registry,
example script — and then wrote parity test files that *looked*
rigorous but never actually loaded real weights:

- `tests/local_tests/pipelines/test_magi_human_pipeline_smoke.py` —
  only a preflight that builds the DiT on `meta` device and cross-
  checks shapes against the HF index. No forward call. No numerics.
- `tests/local_tests/pipelines/test_magi_human_pipeline_parity.py` —
  a skeleton that gated on an upstream module that's hard to construct
  in-process, then called `pytest.skip("... is a follow-up milestone")`
  unconditionally. Zero assertions ever executed.
- **No per-component parity tests** under `tests/local_tests/transformers/`,
  `tests/local_tests/encoders/`, or `tests/local_tests/vaes/` at all.
  Step 6's "each subagent produces a component parity test" was simply
  skipped.

From pytest's perspective, the smoke test PASSED and the parity skeleton
SKIPPED. I reported the port as "scaffold complete with parity tests" —
but in reality no numerical parity was ever verified anywhere.

The user caught this with the observation: *"the fact that weights were
not converted at all makes me believe that these parity test are not
accurate right now"* — which turned out to be exactly correct. I had
never run the conversion, so the tests would have skipped anyway; when
I later ran the conversion and wrote **real** parity tests, the DiT
showed a measurable 1% mean drift that was completely invisible in the
skeleton's "all green" output.

**Why this happened (root causes in the skill):**

1. **Pytest-skip is visually indistinguishable from pytest-pass** in CI
   output and summary lines. A test that skips every time is a liability,
   not an asset — the skill says "skip-on-missing" but doesn't warn
   that a skip-only test provides zero ongoing regression coverage.
2. **The skill doesn't prescribe a local verification checklist** tying
   "conversion (step 4)" → "component parity (step 6)" → "pipeline
   parity (step 13(a))" into a linear ordering that forces non-skip
   passes at each stage before proceeding. Steps 4 and 6 read as
   parallel; they are not.
3. **Step 6's subagent prompt template** mentions "Run: `pytest ...` —
   the test must produce a non-skip pass", but the skill's main-agent
   instructions around step 6 don't make the non-skip gate a blocker.
   Easy to miss when the main agent isn't dispatching subagents (I did
   the component work myself).
4. **Step 13(a) is the ONLY place the skill explicitly says "non-skip
   pass required before user handoff"**. A conscientious porter who
   only reads step 13 (because step 6 says "optional, subagents do this")
   can and will ship without component parity.

**Options:**

- (a) **Add a blocking pre-handoff checklist** at the end of the skill:
  > Before telling the user the port is ready, run all of these and
  > confirm **non-skip passes** on each:
  > 1. `pytest tests/local_tests/<bucket>/test_<family>_*parity*.py`
  >    for each non-reused component (DiT + any new VAE + any new
  >    encoder). Skips here mean you didn't convert weights first.
  > 2. `pytest tests/local_tests/pipelines/test_<family>_pipeline_parity.py`
  >    (the step-13(a) gate).
  > 3. `python examples/inference/basic/basic_<family>.py` produces a
  >    non-corrupt mp4.
  > If any of the above skips, STOP and fix — a skipping test is a
  > missing test.

- (b) **Reorder the steps** so conversion is a hard prerequisite for
  step 6: move step 4 ("Convert weights") above the parity work and
  note: "component parity tests in step 6 load from the output of
  step 4; if step 4 hasn't been run, step 6 cannot be completed."

- (c) **Strengthen the "Skip-on-missing" language** to add: *"But if
  YOU, the porter, get a skip on your local machine, that means you
  haven't done the conversion or clone. Fix that before proceeding."*

- (d) **Add a grep-able "DONE" criterion** to step 6: the main agent
  must verify `pytest tests/local_tests/<bucket>/test_<family>_parity.py
  -v` output contains `PASSED`, not `SKIPPED` or `xfailed`, before
  marking the step complete.

**My guess:** all four. This is the single biggest gap the port
exposed. A skeleton test that silently skips is worse than no test,
because it creates false confidence.

---

## 17. Upstream private-DSL dependencies (`magi_compiler`, etc.)

**Where in the skill:** Nowhere currently.

**What I found (during the MagiHuman parity round):** The upstream
`daVinci-MagiHuman/inference/model/dit/dit_module.py` hard-imports
three modules from `magi_compiler` (SandAI's internal graph compiler),
none of which are on PyPI. This makes `from inference.model.dit import
DiTModel` fail at module-load with `ModuleNotFoundError: No module named
'magi_compiler'` — so you cannot run the official DiT side-by-side with
your FastVideo port for parity testing without stubbing.

This is not unique to MagiHuman. Other upstream research codebases are
likely to ship their own internal compilers / ops (TorchTitan,
`megatron_ops`, proprietary FlashAttention variants, custom op
registries). The skill currently treats "clone official repo + pip
install -r requirements.txt" as sufficient for parity; it's not when
the upstream has private deps.

**What worked for MagiHuman (tests/local_tests/helpers/magi_human_upstream.py):**

A small `install_stubs()` helper that before importing `inference.*`:
1. Registers a fake `magi_compiler` module in `sys.modules` with
   identity decorators for `magi_compile` and `magi_register_custom_op`.
2. Monkey-patches `inference.infra.distributed.get_cp_world_size` to
   return 1 (so Ulysses scatter/gather are no-ops).
3. Replaces `scatter_to_context_parallel_region` /
   `gather_from_context_parallel_region` with identity functions.
4. Registers the `torch.ops.infra.*` custom ops that upstream calls
   via `torch.ops.infra.flash_attn_func(q, k, v)` — routing them
   through `F.scaled_dot_product_attention` so attention kernels
   align between FastVideo and upstream (see item 19).

**Options:**

- (a) Add a "**Upstream private deps**" subsection to step 5 ("Clone
  official repo") and/or the parity-test pattern: *"If the upstream
  module imports fail with `ModuleNotFoundError` on non-PyPI packages
  (private compilers, custom op registries), write a small stub helper
  under `tests/local_tests/helpers/<family>_upstream.py`. Typical stubs
  needed: (1) identity decorators for compile/op-registration macros;
  (2) `get_cp_world_size → 1` + identity scatter/gather; (3)
  `torch.ops.<ns>.*` registration preservation."* Point at the
  MagiHuman helper as the template.

- (b) Leave as-is; assume each porter figures it out per-family.

**My guess:** (a). Stubs are a common enough reality in research
codebases that documenting the pattern saves every future porter half
a day.

---

## 18. `torch.ops.<ns>.<op>` registration is a side-effect you can't stub away

**Where in the skill:** Nowhere.

**What I found (during the MagiHuman parity round):** Upstream's
`dit_module.py` uses `@magi_register_custom_op(name="infra::flash_attn_func", ...)`
as a decorator, then calls the op elsewhere as
`torch.ops.infra.flash_attn_func(q, k, v)`. The decorator's real
implementation registers the op with `torch.library`, which makes the
`torch.ops.infra.flash_attn_func` attribute resolvable.

My first stub made `magi_register_custom_op` an identity decorator —
it returned the function unchanged but never registered it. Module
import succeeded; the call site later failed with `'_OpNamespace' 'infra'
object has no attribute 'flash_attn_func'`. That error is delayed from
import time to runtime, making it harder to debug.

**Fix (in the helper):** the stub decorator must still call
`torch.library.Library("infra", "FRAGMENT").define(...)` and
`torch.library.impl(...)` so `torch.ops.infra.*` resolves — just with
a test-friendly backing implementation (SDPA in our case).

**Options:**

- (a) Mention this as a **pitfall** in the new "Upstream private
  deps" subsection (see item 17): *"Identity-decorator stubs silently
  lose custom-op registration side-effects. If the upstream decorator
  registers with `torch.library`, your stub must too — otherwise call
  sites using `torch.ops.<ns>.<op>(...)` will fail at runtime with
  attribute errors that point at the wrong module."*

**My guess:** (a). Short callout, prevents a 30-min head-scratch.

---

## 19. GQA kernel alignment in cross-kernel parity tests

**Where in the skill:** Partially in the "FastVideo layers and
attention" section's attention selection rules, but not specifically
for parity testing.

**What I found (during the MagiHuman DiT parity):** Upstream calls
`flash_attn_func` directly with GQA-shaped inputs — `q: [B, L, 40, 128]`,
`k/v: [B, L, 8, 128]`. flash_attn handles GQA natively. When I routed
the stubbed `torch.ops.infra.flash_attn_func` through
`F.scaled_dot_product_attention`, SDPA errored with shape mismatch on
dim 1 because SDPA requires `num_heads_k == num_heads_q`.

The fix is `k.repeat_interleave(num_heads_q // num_heads_kv, dim=2)`
before calling SDPA — but only one side (SDPA / parity stub) needs it.

The broader point: **parity tests that align two attention kernels
(flash_attn vs SDPA vs VMoba vs Sage) must explicitly handle GQA
expansion on the non-native side.** The skill's attention section
mentions GQA in passing but doesn't name this pitfall.

**Options:**

- (a) Add a pitfall to the "Parity test pattern" section:
  *"When your parity test routes upstream attention through a different
  kernel than upstream uses (e.g. upstream=flash_attn, your stub=SDPA),
  remember to expand KV heads manually on the SDPA side —
  `k = k.repeat_interleave(num_heads_q // num_heads_kv, dim=head_axis)`.
  flash_attn handles GQA natively; SDPA does not."*

**My guess:** (a). Adds 3 lines to the parity pattern, saves the next
porter from a confusing shape error mid-test.

---

## 20. VAE parity: normalization-convention mismatches across wrappers

**Where in the skill:** Nowhere explicitly.

**What I found (during the MagiHuman VAE parity):** The upstream
`Wan2_2_VAE.decode(z)` does:

    # Inside _video_vae.decode (inference/model/vae2_2/vae2_2_module.py:874)
    z = z / scale[1] + scale[0]    # where scale = [mean, 1/std]
    #   = z * std + mean            # denormalizes BEFORE feeding the decoder

— i.e. the upstream API takes latents in *normalized diffusion space*
and denormalizes internally.

Diffusers' `AutoencoderKLWan.decode(z)` is the opposite: it expects
the caller to have **already applied** `z = z * std + mean`. In
FastVideo, this denormalization lives in the `DecodingStage`, not in
the VAE class.

My first VAE parity test fed `z = randn(...)` to both and got 86% of
elements failing tolerance (max diff 1.38). After applying the
denormalization only on the Diffusers side — `z_denorm = z * std + mean`
then `fv_vae.decode(z_denorm)` — parity dropped to `diff_max=8e-4,
diff_mean=5e-5` (near fp32 noise). The VAEs agree; the wrappers just
place the normalization step in different layers of the call stack.

This pattern is common when comparing **"research-code" wrappers that
bundle normalization inside `decode()`** vs. **"library-code" wrappers
that expect pre-normalized input** (the Diffusers convention).

**Options:**

- (a) Add a pitfall to the "Parity test pattern" section or to VAE
  reuse advice: *"When the upstream VAE's `decode()` bundles the
  `z → z*std + mean` denormalization but the Diffusers-format VAE
  expects pre-denormalized input (the FastVideo/Diffusers convention
  — the DecodingStage applies `std/mean` before calling `vae.decode()`),
  a naive parity test that feeds the same `z` to both will diverge
  drastically. Apply the denormalization on the Diffusers side
  explicitly."*

**My guess:** (a). One paragraph, gives the next porter the exact
fix.

---

## 21. Realistic parity tolerances for bf16 through a deep DiT

**Where in the skill:** "Parity test pattern" — the `atol=1e-4, rtol=1e-4`
default with "loosen deliberately, never silently".

**What I found (during the MagiHuman DiT parity):** The 40-layer bf16
DiT with upstream-flash_attn vs FastVideo-SDPA kernel crossing showed:

- diff_max: 0.057 (on abs-mean 0.68 signal — ~8% tail drift)
- diff_mean: 0.005 (~0.75%)
- diff_median: 0.003
- abs_mean match: 0.6833 vs 0.6825 (< 0.2%)

At `atol=1e-4, rtol=1e-4` this is RED with 99% of elements failing —
but the structure is clearly correct (median < 5e-3, abs_mean matches,
text zero-padding exact). The drift is cross-kernel bf16 accumulation
through 40 layers, not a structural bug.

`atol=1e-4` is the right default for **single-block, single-kernel,
shallow** parity tests (e.g. comparing one Wan DiT block in isolation).
For **full-DiT, multi-kernel, deep** parity the realistic tolerance is
~2 orders of magnitude larger, and element-wise assertion alone is the
wrong metric. What I ended up with that's actually informative:

- `assert_close(text_rows, ref_text_rows, atol=1e-6, rtol=1e-6)` —
  text rows are zero-padded on both sides; any drift here is a bug.
- `assert_close(fv_full, ref_full, atol=0.1, rtol=0.1)` — catches
  structural bugs (sign flips, missing layers, bad permutations)
  while tolerating realistic bf16 drift.
- `assert abs(ref_abs_mean - fv_abs_mean) / ref_abs_mean < 0.05` —
  sanity check that mean magnitudes agree to 5% — catches gross
  dropouts (e.g. a modality branch silently zeroed out).
- Printed per-modality breakdown (video / audio / text abs_mean,
  diff_max, diff_mean) as a diagnostic for future iteration.

**Options:**

- (a) Replace the skill's "default `atol=1e-4, rtol=1e-4`" advice
  with a table:
  | Scope | `atol` | `rtol` | Notes |
  |---|---|---|---|
  | Single block, single kernel | 1e-4 | 1e-4 | tight; flags any real bug |
  | Full DiT, aligned kernels | 1e-2 | 1e-2 | cross-layer accumulation |
  | Full DiT, cross-kernel (flash_attn vs SDPA) | 0.1 | 0.1 | needs abs_mean drift check below 5% alongside |
  | VAE decode, fp32 | 5e-2 | 5e-2 | after normalization alignment |
  | Encoder wrapper (both wrap same HF class) | 1e-3 | 1e-3 | should be near-zero |

- (b) Add a note: *"Element-wise `assert_close` alone is not enough
  for full-DiT parity. Always combine with (1) a global abs_mean
  drift check (<5%) and (2) per-modality diagnostic prints so
  failures are debuggable."*

**My guess:** both. The skill currently sets porters up for a false
red and then implicit silent loosening; explicit tolerance-by-scope
guidance is more honest.

---

## 22b. "T2V-only" is not a valid scope interpretation of "base model"

**Where in the skill:** Step 1 (inputs), "Scoping a port" open item (15
above), `When to use` / `When not to use`.

**What happened (during the MagiHuman audio round, user-corrected):**
The user said *"start with the base model ONLY"*. I interpreted that as
two axes at once:

  1. Variant axis — base, distill, 540p_sr, 1080p_sr → **base only** ✅
  2. Output-modality axis — audio + video vs video-only → **video-only** ❌

Axis (2) was my reading; the user had only meant axis (1). "Base" in
MagiHuman's README is a model *variant* (undistilled full-quality
joint-AV output), not a modality subset. Shipping a silent-mp4 pipeline
and labeling it `MagiHumanT2VConfig` with workload `"t2v"` was a
semantically wrong scope.

The downstream damage: the DiT's `final_linear_audio` head was trained
to be consumed, not discarded. Dropping the audio output subtly changes
the trained-intended pipeline purpose, even though the DiT forward path
itself kept the audio token stream (so the video quality was unchanged).

**Options:**

- (a) **In the "Scoping a port" paragraph** (item 15): add an explicit
  sentence about modality-axis scope: *"'Base model' refers to a
  checkpoint variant (base vs distill vs SR vs I2V), NOT to an output
  modality. If the upstream README lists audio, face, pose, or
  additional output channels as core features of the base variant,
  those belong in the first PR. Ask the user to spell out BOTH axes
  explicitly at step 1 if in doubt."*
- (b) **Add a README-first checklist to step 2 ('Study the reference')**:
  *"Before implementing anything, list all **output modalities** the
  base checkpoint produces (video, audio, pose, segmentation, depth,
  …). Every modality with a dedicated output head in the DiT must be
  either decoded + emitted by the pipeline, or explicitly documented
  as dropped with the user's agreement."*
- (c) **Pitfall to pipeline-config section**: *"If the upstream DiT
  has N output heads (`final_linear_*`) but your `_required_config_modules`
  only covers M<N, you have silently scoped out some modalities. That
  is a scoping decision, not an implementation detail — surface it
  explicitly to the user."*

**My guess:** all three. This was the single biggest *scope-interpretation*
error on the MagiHuman port.

---

## 22c. Align with in-progress first-class ports, don't import their PyPI counterparts

**Where in the skill:** VAE reuse + encoder reuse advice.

**What I found (during the MagiHuman audio round, user-corrected):** My
first audio integration used `from diffusers import AutoencoderOobleck`
because it's identical architecture to Stable Audio Open's VAE and the
weights load directly from `stabilityai/stable-audio-open-1.0/vae/`.
That "works" — parity was exact, end-to-end pipeline was green — but
the user pointed out that an open PR (`happy-harvey/FastVideo:harvey/audio_dev`,
#1080) is adding **first-class** Stable Audio support to FastVideo
(`fastvideo/models/stable_audio/`). Importing from `diffusers` when a
first-class native port is in flight creates a duplicate-abstraction
debt: when PR #1080 merges, MagiHuman still imports diffusers; two
teams end up maintaining parallel Stable Audio integrations.

**Resolution on MagiHuman:** I ported Diffusers' `AutoencoderOobleck`
locally into `fastvideo/models/vaes/oobleck.py` (first-class, no
runtime diffusers import), kept the architecture bit-identical so
weights still load from the same HF repo, and left a comment pointing
at PR #1080's `StableAudioPretransform` as the future consolidation
target.

**The broader principle:** When the skill says "reuse existing
FastVideo components", it should also say: *"before reaching for
`from diffusers import ...` or `from transformers import ...` as a
shortcut, (a) check if FastVideo already has a native port (search
`fastvideo/models/**`), (b) check open PRs for in-flight first-class
ports (`gh pr list --state open --search <component_name>`), (c) if
both miss, consider porting locally — many upstream Apache-licensed
modules are small enough (`AutoencoderOobleck` is ~450 lines) to
port cleanly without the third-party runtime dependency."*

**Options:**

- (a) Add a **reuse priority order** to the "Decide what to reuse"
  step (step 3 in the skill):
  1. Existing FastVideo native class (`fastvideo/models/...`).
  2. Open PR adding a FastVideo native class (check with `gh pr list`).
  3. Local port of the upstream / Diffusers / Transformers class
     under `fastvideo/models/`.
  4. Last resort: runtime `from diffusers/transformers import ...` —
     document why and flag for follow-up.
- (b) Add a pitfall: *"A `from diffusers import AutoencoderX` in a
  FastVideo pipeline class signals 'finish-porting-later'. It
  freezes you to Diffusers' release cadence, opts you out of
  FastVideo's FSDP/CP/loader infrastructure, and forks you from
  any in-flight first-class port."*

**My guess:** (a) + (b). The user explicitly called this out as
first-class is the expected bar; the skill doesn't currently reinforce
that expectation strongly enough.

---

## 22. Custom VAE `scaling_factor` / `shift_factor` are per-channel vectors, not scalars

**Where in the skill:** Weight-conversion section + VAE reuse notes
mention `scaling_factor` but don't warn about the per-channel case.

**What I found (during the VAE parity):** The Wan 2.2 TI2V-5B VAE
has `latents_mean` + `latents_std` as **per-channel vectors of length
z_dim=48** (not the more common scalar `scaling_factor` that Stable
Diffusion uses). My initial parity test tried

    z_norm = (z - latents_mean) / latents_std

without the per-channel `.view(1, z_dim, 1, 1, 1)` broadcast, which
would produce silently-wrong shapes for a 5D latent tensor (it might
broadcast along time or spatial dims instead of channel).

For ported pipelines using a Wan-family VAE (or any modern video VAE
that ships per-channel normalization), the normalization must be
applied with explicit `.view(1, -1, 1, 1, 1)` reshape on the
mean/std vectors.

**Options:**

- (a) Add a pitfall in the VAE reuse section: *"Modern video VAEs
  (Wan, AutoencoderKLWan, HunyuanVAE) ship **per-channel**
  `latents_mean` / `latents_std` vectors. When applying the
  normalization in a `DecodingStage` or parity test, always reshape
  with `.view(1, z_dim, 1, 1, 1)` to broadcast along C, not along
  T/H/W."*

**My guess:** (a). One-liner prevents a silent broadcast bug that
produces nonsense video output.

---

## 23. Skill assumes a "full model port"; "first-class component contribution" has no slot

**Where in the skill:** "Files you will create or touch" table,
"Outputs" section, every Steps mention of "the pipeline".

**What I found (during the will/stable-audio audit, 2026-04-25):**
The skill's 17-row Files-table assumes a porter is adding an entire
model family — DiT + VAE + encoder + pipeline + presets + registry +
tests + example. Stable Audio's VAE (Oobleck) was added as a
**first-class component contribution** (just rows 4–6 + 14 of the
table) so it could be reused by multiple downstream pipelines
(daVinci-MagiHuman audio decode, the in-flight T2A pipeline in
PR #1080, etc.) without each pipeline depending on `from diffusers
import AutoencoderOobleck`. The skill has no framing for this kind of
contribution: every reference to "the pipeline" implicitly assumes
the contribution includes a pipeline.

Component-only contributions are common in practice and arguably
under-encouraged. They de-risk shared infrastructure (one PR landing
the VAE, then independent PRs landing each pipeline that uses it)
and let multiple ports share validation work.

**Options:**

- (a) Add a **"When the contribution is a component, not a model"**
  subsection to "When to use" / "When not to use", with a reduced
  Files-table specifying only rows 4–6 (or analogous: 1–3 for a
  DiT-only contribution; encoder rows for an encoder-only contribution)
  - row 14 (component parity test). Explicitly note the contribution
  *will* skip the pipeline / preset / registry / smoke / pipeline-parity
  rows and that's OK as long as a downstream pipeline PR is in flight
  or planned.
- (b) Leave the skill strictly model-port-focused; treat
  component-only contributions as an unrelated workflow with its own
  skill (`add-component`?).
- (c) No change; let porters figure it out.

**My guess:** (a). The pattern is real and recurring (Stable Audio
VAE here, future Stable Audio T5 conditioner, future shared video
VAEs across families). Naming the workflow saves the next porter from
guessing whether "skip the pipeline" is acceptable.

---

## 24. Component-bucket inheritance is implicit

**Where in the skill:** "Linear layers" table picks bucket-by-bucket,
but the skill never explicitly enumerates "config base class for each
bucket".

**What I found (during the will/stable-audio audit):** Initially I put
`SAAudioVAEConfig(EncoderConfig)` — the wrong base class. The
underlying model is a VAE, not a text/image encoder, but the file
landed in `fastvideo/configs/models/encoders/` because it wraps a
"text-encoder-style" lazy loader. This was a real bug: it meant the
config didn't have `load_encoder` / `load_decoder` / tiling fields,
and no caller could feed an `OobleckVAEConfig` into a `vae_config`
slot that expects a `VAEConfig` subclass.

The skill's existing tables list the right bucket per *model*:
- `fastvideo/models/dits/<family>.py` → DiT bucket
- `fastvideo/models/vaes/<family>vae.py` → VAE bucket
- `fastvideo/models/encoders/<family>.py` → encoder bucket

But it doesn't tie config inheritance back: per bucket, the new
config *must* inherit from a bucket-specific base.

| Bucket | Config base | Arch base |
|---|---|---|
| `dits/` | `DiTConfig` | `DiTArchConfig` |
| `vaes/` | `VAEConfig` | `VAEArchConfig` |
| `encoders/` (text) | `TextEncoderConfig` | `TextEncoderArchConfig` |
| `encoders/` (image) | `ImageEncoderConfig` | `ImageEncoderArchConfig` |

Choosing the wrong base typechecks fine but blows up at pipeline
wire-up time when the slot expects the bucket's specific base.

**Options:**

- (a) Add a "Bucket → config base" table to step 6 (component porting)
  and to the "Quick reuse audit" subsection, so the right base is
  picked the first time.
- (b) Add a runtime check / lint that flags model-bucket vs config-
  inheritance mismatches.

**My guess:** (a). One table, one row per bucket.

---

## 25. Audio is a first-class workload but the skill is video-shaped

**Where in the skill:** Many places — `SSIM regression test (row 15)`
assumes video frames; `fps`, `height`, `width`, `num_frames` are baked
into preset / pipeline-config defaults; the I2V section assumes spatial
conditioning; example scripts all save mp4.

**What I found (during the will/stable-audio audit):** Stable Audio
Open is text-to-audio. None of the video-shaped conventions apply:

- **No SSIM** — audio similarity needs different metrics (mel-scale
  spectrogram L1, multi-resolution STFT, learned audio metrics like
  CLAP cosine).
- **No `height` / `width`** — replaced by `audio_seconds` /
  `sampling_rate`.
- **No `fps`** — replaced by sample rate.
- **No mp4 output** — `.wav` (or muxed audio track on an existing mp4
  for AV pipelines).
- **No "preset workload_type t2v / i2v"** — `t2a` (text-to-audio),
  `a2a` (audio-to-audio), `av` (combined).

The skill doesn't acknowledge audio as a workload at all. The MagiHuman
port worked around this by inheriting video conventions and bolting
audio on (random audio latent in latent prep, audio decode in a
separate stage that writes to `batch.extra["audio"]` for FastVideo's
existing audio-mux path). PR #1080's standalone Stable Audio T2A
pipeline will need to invent its own per-stage conventions because
nothing in the skill guides them.

**Options:**

- (a) Add a **"Modalities other than video"** section near the
  "FastVideo's single architecture" intro with a small table:

  | Field | Video (default) | Audio | Joint AV |
  |---|---|---|---|
  | preset workload_type | "t2v" / "i2v" | "t2a" / "a2a" | "av" |
  | shape coords | `(height, width, num_frames, fps)` | `(seconds, sampling_rate)` | both |
  | regression metric | SSIM | mel-spectrogram L1 / CLAP | both |
  | example output | `.mp4` | `.wav` | `.mp4` (muxed) |

- (b) Generalize the canonical-stage-order diagram to mention audio:
  add an "Audio T2A" variant alongside the existing T2V and I2V
  diagrams.

- (c) Cross-reference PR #1080's stable_audio pipeline (once merged)
  as the canonical T2A example.

**My guess:** all three. Audio is a real workload and likely to grow.

---

## 26. Pipeline-glue wrappers vs. plain VAE classes

**Where in the skill:** Files-table rows 4–6 (VAE + config) but no
mention of where wrapper / loader files belong.

**What I found (during the will/stable-audio audit):** I added two
files for the Stable Audio VAE:

- `fastvideo/models/vaes/oobleck.py` — the plain VAE class with
  `from_pretrained`. Standalone-usable.
- `fastvideo/models/vaes/sa_audio.py` — a thin lazy-loader nn.Module
  wrapper used by pipeline-component-loaded contexts (lazy fetch on
  first forward, hide weights from `named_parameters`).

The wrapper exists because FastVideo's `ComposedPipelineBase.load_modules`
walks `named_parameters` to match against the host pipeline's
converted-repo state dict; for VAEs/encoders that are externally
fetched (gated repo, not bundled), the wrapper is the standard pattern
to skip that match. `T5GemmaEncoderModel` does the same thing.

The skill's Files-table has one row per (model, config) but no row for
this wrapper — porters either (a) discover the pattern by reading
existing encoder files, (b) inline the wrapper into the model file
(messy), or (c) skip the wrapper and crash at pipeline load.

**Options:**

- (a) Add a row 5b to the Files-table:
  *"`fastvideo/models/<bucket>/<family>_loader.py` (optional) — thin
  lazy-loader nn.Module wrapper around the model class. Required when
  the model is fetched from an external HF repo (gated, not bundled
  in the converted family directory) and is consumed by a pipeline
  that calls `ComposedPipelineBase.load_modules`."*
- (b) Document the wrapper pattern in the "Parallel component porting"
  subagent prompt template + a small template snippet.

**My guess:** (a) + (b). Naming the file pattern + showing the
template avoids the rediscovery cost.

---

## Summary

| # | Topic | Severity | Action |
|---|-------|----------|--------|
| 1 | No Wan tests exist | Medium | User decides: grandfather Wan, or retrofit tests. |
| 2 | `wan_to_diffusers.py` legacy | Low | Consider deleting if unused. |
| 3 | `ConditioningStage` universality | Low | Audit other families; finalize framing. |
| 4 | I2V image-encoder selection rule | Low | Add rule if one exists. |
| 5 | Wan2.2 MoE / dual-guidance fields | Medium | Decide if they deserve a dedicated section. |
| 6 | T2V/I2V cross-attn naming | Low | Decide if skill should prescribe. |
| 7 | Pipeline-parity gate is local-only | Low | Consider explicit callout. |
| 8 | Variant-pipeline enumeration | Low | Decide whether to enumerate. |
| 9 | HF token scope phrasing | Low | Consider "read + write". |
| 10 | Raw `nn.Linear` policy | Low | Clarify in pitfall #11 if you have a position. |
| 11 | Packed-expert linears (MoE / multi-modality) | Medium | Name as accepted exception to the ReplicatedLinear rule. (From MagiHuman port.) |
| 12 | VAE reuse — `-Diffusers` suffix | Low | One-line note in the conversion recipe. (From MagiHuman port.) |
| 13 | HF token at conversion, not just publish | Low | Reword step 1 prerequisite. (From MagiHuman port.) |
| 14 | Cross-modality flat-stream attention | Low | Extend the attention selection table. (From MagiHuman port.) |
| 15a | ArchConfig scope = DiT-only; fallback to official Python when HF config is empty | High | Rewrite step 2 + add hard pitfall. (From MagiHuman port, user-corrected.) |
| 15 | Scoping a port with multiple variants | Low | Add a "Scoping a port" paragraph. (From MagiHuman port.) |
| 16 | **Skip-on-missing parity is a silent no-op trap** | **Critical** | **Pre-handoff checklist + reorder conversion before parity + "skip = missing, not pass" language.** (From MagiHuman port, user-corrected — the single biggest gap.) |
| 17 | Upstream private-DSL stubs (`magi_compiler` etc.) | Medium | Add a "Upstream private deps" subsection with stub-helper template. (From MagiHuman port.) |
| 18 | `torch.ops.<ns>.<op>` registration side-effect | Low | Pitfall callout in the private-deps subsection. (From MagiHuman port.) |
| 19 | GQA kernel alignment in parity tests | Low | 3-line pitfall in "Parity test pattern". (From MagiHuman port.) |
| 20 | VAE normalization convention mismatches | Medium | Pitfall paragraph in VAE reuse or parity pattern. (From MagiHuman port.) |
| 21 | Realistic parity tolerances by scope | Medium | Replace the scalar `atol=1e-4` default with a scope-indexed table. (From MagiHuman port.) |
| 22 | Per-channel VAE scaling vectors | Low | One-liner pitfall in VAE reuse. (From MagiHuman port.) |
| 22b | **"Base" is a checkpoint variant, not a modality subset** | **High** | Add modality-axis scope clarifier to step 1 + README-first checklist in step 2 + output-head pitfall in pipeline-config. (From MagiHuman audio round, user-corrected.) |
| 22c | First-class over `from diffusers import` | Medium | Add explicit reuse priority order (FastVideo-native > in-flight PR > local port > diffusers import as last resort) + pitfall. (From MagiHuman audio round, user-corrected.) |
| 23 | **No "first-class component" workflow** | **High** | Add a "When the contribution is a component, not a model" subsection + reduced Files-table. (From will/stable-audio audit.) |
| 24 | Component-bucket → config-base inheritance is implicit | Medium | Add a "Bucket → config base" table to step 6 + Quick reuse audit. (From will/stable-audio audit.) |
| 25 | **Audio is a first-class workload but skill is video-shaped** | **High** | Add a "Modalities other than video" section + audio variant of the canonical stage diagram + cross-ref PR #1080. (From will/stable-audio audit.) |
| 26 | Pipeline-glue wrappers vs. plain VAE/encoder classes | Medium | Add Files-table row 5b (`<family>_loader.py`) + template snippet in the subagent prompt. (From will/stable-audio audit.) |

None of these block a new porter from using the skill today; they're
polish / consistency items that need a codebase-owner call.

**Exception: item 16 is load-bearing.** The skill as currently written
lets a port ship without any non-skip parity coverage, because the
"skip-on-missing" convention means green pytest output does not
actually mean parity was verified. Without a pre-handoff checklist
that explicitly bans skips on the porter's local machine, this will
keep happening.
