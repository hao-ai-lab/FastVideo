# GLM-Image Local Tests

Local-only parity and smoke tests for the `glm_image` FastVideo port. These
tests compare FastVideo against the official HF `transformers` + `diffusers`
GLM-Image implementation and are not expected to run in CI unless explicitly
promoted later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/glm_image/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `glm_image` |
| Workload types | `T2I` (primary), `I2I` (planned follow-up) |
| Official reference | HF `transformers.models.glm_image` + HF `diffusers.pipelines.glm_image` |
| Local reference dir | `/home/hal-kaiqin/dreamverse/FastVideo-PR/reference/transformers/src/transformers/models/glm_image/`, `/home/hal-kaiqin/dreamverse/FastVideo-PR/reference/diffusers/src/diffusers/pipelines/glm_image/`, `/home/hal-kaiqin/dreamverse/FastVideo-PR/reference/diffusers/src/diffusers/models/transformers/transformer_glm_image.py` |
| Official commit/version | transformers `a65bf6c9d03322c1bf1963ef7c64289c4b3f0757`, diffusers `ff3b86b4755b46a7b5656dfcf84d25bd25ad4740` |
| HF weights | `zai-org/GLM-Image` |
| HF revision | default (`main`) |
| Local weights dir | `/home/hal-kaiqin/dreamverse/FastVideo-PR/official_weights/glm_image` (downloaded) |
| Source layout | `diffusers` |
| Needs conversion | `no` (HF repo already in diffusers component layout) |

Token env var (set externally if needed for gated downloads): `HF_TOKEN`.
The repo is currently public; no token required for inspect / public read.

## Shared Environment Setup

Run from the FastVideo repo root in the same conda/env used for FastVideo.
Do not create a separate upstream environment for parity tests.

Sparse-checkout HF reference repos (already done during prep):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers reference/transformers
git -C reference/transformers sparse-checkout init --cone
git -C reference/transformers sparse-checkout set src/transformers/models/glm_image

git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/diffusers reference/diffusers
git -C reference/diffusers sparse-checkout init --cone
git -C reference/diffusers sparse-checkout set \
    src/diffusers/pipelines/glm_image \
    src/diffusers/models/transformers \
    src/diffusers/models/autoencoders \
    src/diffusers/schedulers
```

Editable installs are intentionally **not** used here: parity tests will import
`transformers.models.glm_image` and `diffusers.pipelines.glm_image` from the
already-installed HF packages on the active env, while the cloned reference
trees serve as source-readable parity references (Phase 1 architecture study,
key/shape dumps).

**`pyproject.toml` pins `transformers==5.0.0rc3`** — the first release shipping
GLM-Image's AR encoder (`GlmImageForConditionalGeneration`). So the AR model and
the full T2I/I2I pipeline run on the committed pins; no local override is needed.

Diffusers: production uses FastVideo's native DiT + a stock `AutoencoderKL`, so
`diffusers>=0.33.1` is enough. The diffusers `GlmImageTransformer2DModel` /
`GlmImagePipeline` classes are used **only by the parity tests** and need
`diffusers>=0.37.0.dev0`; those tests skip with actionable messages on older
diffusers (install it locally to run them):

```bash
uv pip install -U "git+https://github.com/huggingface/diffusers.git@ff3b86b4"
```

| Class | Used in | First available in |
|---|---|---|
| `transformers.GlmImageForConditionalGeneration` | production (AR encoder) | `transformers>=5.0.0rc0` — **the committed pin** |
| `diffusers.GlmImageTransformer2DModel`, `GlmImagePipeline` | parity tests only | `diffusers>=0.37.0.dev0` |

Do not change core dependency versions (`torch`, `flash-attn`, `triton`, CUDA
packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: pyproject pins transformers==5.0.0rc3 (committed); diffusers test-only classes installed locally
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: none
```

## Weight Setup

Run from the repo root (`/home/hal-kaiqin/dreamverse/FastVideo-PR`):

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "zai-org/GLM-Image" \
    "/home/hal-kaiqin/dreamverse/FastVideo-PR/official_weights/glm_image"
```

Weights are ~30 GB (vision_language_encoder 4 shards, transformer 3 shards,
vae + text_encoder). **Downloaded** to
`/home/hal-kaiqin/dreamverse/FastVideo-PR/official_weights/glm_image`. Cache via
`HF_HOME` if `/` is small.

## Prototype And Conversion Artifacts

`needs_conversion=no` for the base layout: HF weights are already in diffusers
component subfolders that FastVideo's diffusers-style loaders can consume,
**after** native components exist with state-dict surfaces that match. Param
mapping work belongs in `fastvideo/configs/models/dits/glm_image.py`
(`param_names_mapping`) and the new native VAE / encoder configs, not in a
conversion script.

```text
official_key_dumps:
  transformer: converted_weights/glm_image/_mapping/transformer_official_keys.json
  vae: converted_weights/glm_image/_mapping/vae_official_keys.json
  text_encoder: converted_weights/glm_image/_mapping/text_encoder_official_keys.json
  vision_language_encoder: converted_weights/glm_image/_mapping/vision_language_encoder_official_keys.json
fastvideo_key_dumps:
  transformer: converted_weights/glm_image/_mapping/transformer_fastvideo_keys.json
  vae: converted_weights/glm_image/_mapping/vae_fastvideo_keys.json
  text_encoder: converted_weights/glm_image/_mapping/text_encoder_fastvideo_keys.json
  vision_language_encoder: converted_weights/glm_image/_mapping/vision_language_encoder_fastvideo_keys.json
conversion_script: none (param_names_mapping in configs handles in-place rename at load)
conversion_source_layout: diffusers
converted_weights_dir: none
strict_load_status: not_run
```

## Expected Parity Tests

Local tests for this family (non-skip PASS on GB200; transformers 5.0.0rc3 — the
committed pin — plus diffusers 0.37.1 for the test-only reference classes):

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| `transformer` (DiT) | `pipeline_glm_image.py::GlmImagePipeline.__init__` → `GlmImageTransformer2DModel` from `transformer/config.json`; defined in `transformer_glm_image.py` | `tests/local_tests/glm_image/test_transformer_parity.py` | Re-ported from diffusers ref. fp32 cosine 1.000000 (100% within 5e-3); bf16 cosine 0.999969 — the bf16 element-wise drift is rounding over 30 layers vs a different SDPA kernel, not a bug (I020). RoPE proven exactly equivalent. | `non_skip_pass` (fp32 + bf16 + strict-load) |
| `vae` (AutoencoderKL) | `pipeline_glm_image.py::GlmImagePipeline.__init__` → `AutoencoderKL` from `vae/config.json` | `tests/local_tests/glm_image/test_vae_parity.py` | Native `AutoencoderKL` under the lazy-wrapper exception file; config mirrors `vae/config.json` (block_out_channels, latents_mean/std). | `non_skip_pass` (decode + encode + config, 3/3) |
| `text_encoder` (T5) + `tokenizer` (**ByT5**) | `T5EncoderModel` + `ByT5Tokenizer` per `pipeline_glm_image.py:191-193` | `tests/local_tests/glm_image/test_t5_parity.py` | ByT5Tokenizer asserted; `compute_glyph_embeds` mirrors diffusers `_get_glyph_embeds` (Q003 answered — stock T5/ByT5 path works). | `non_skip_pass` (tokenizer + glyph extract + embeds, 3/3) |
| `vision_language_encoder` (GLM-4-9B AR) | `GlmImageForConditionalGeneration` from `modeling_glm_image.py`; instantiated via `from_pretrained(vision_language_encoder/)` | `tests/local_tests/glm_image/test_ar_parity.py` | Lazy-wrapper exception (E001) in `fastvideo/models/encoders/glm_image_ar_loader.py` confines the HF import. | `non_skip_pass` (surface + to/eval + greedy determinism, 3/3) |
| `processor` (`GlmImageProcessor`) | `transformers/models/glm_image/processing_glm_image.py` | covered alongside AR encoder | Loaded via `ProcessorLoader` (`AutoProcessor`, allowed utility). | `non_skip_pass` (via AR + pipeline) |
| `scheduler` (`FlowMatchEulerDiscreteScheduler`) | shared with other FastVideo flow-matching pipelines; `scheduler/scheduler_config.json` ships shift defaults | covered by pipeline parity test | Dynamic-shift schedule confirmed in practice by pipeline parity (I006). | `non_skip_pass` (via pipeline) |
| `pipeline` | `pipeline_glm_image.py::GlmImagePipeline.__call__` | `tests/local_tests/glm_image/test_pipeline_parity.py` | Full-pipeline pixel parity is ill-posed (stochastic AR + independent latent RNG), so the test removes both: it injects matched `(prompt_embeds, prior_token_ids, latents)` into the diffusers pipeline and the real `GlmImageDenoisingStage`, then compares denoised latents + decoded image. GB200: latent cosine `0.999997`, decoded-image MAE `0.159/255`. | `non_skip_pass` (validity + deterministic parity, 2/2) |
| `pipeline` (I2I / edit) | `pipeline_glm_image.py::GlmImagePipeline.__call__` with `image=` (KV-cache write/read path) | `tests/local_tests/glm_image/test_edit_pipeline_parity.py` | Condition image conditions the DiT via a KV-cache write pass (not noise-the-latent). Validity/wiring gate; diffusers distribution parity deferred (same stochastic-AR caveat). Live-verified on GB200 (temp 5.0.0rc3 bump). | `non_skip_pass` (live validity); diffusers parity `deferred` |

Include reused components in this table. Reuse is accepted only after the
FastVideo component definition and official instantiation arguments have both
been checked and the component parity test passes non-skip.

Run the relevant tests with:

```bash
pytest tests/local_tests/glm_image/test_transformer_parity.py -v -s
pytest tests/local_tests/glm_image/test_vae_parity.py -v -s
pytest tests/local_tests/glm_image/test_t5_parity.py -v -s
pytest tests/local_tests/glm_image/test_ar_parity.py -v -s
pytest tests/local_tests/glm_image/test_pipeline_parity.py -v -s
pytest tests/local_tests/glm_image/test_edit_pipeline_parity.py -v -s
```

## Review Notes

- This is a **re-design** of an existing branch (`feature/glm-image-inference`).
  The current branch contains a port adapted from SGLang with the following
  defects that the re-design must fix:
  1. Runtime `from transformers import GlmImageForConditionalGeneration` /
     `AutoProcessor` / `AutoConfig` in `fastvideo/models/loader/component_loader.py`
     and `from diffusers import AutoencoderKL` — violates production-boundary
     rule.
  2. Stray `print("DEBUG: …")` calls in
     `fastvideo/pipelines/stages/glm_image_decoding.py`.
  3. `FV_USE_SGL_PRIORS` debug branch in
     `fastvideo/pipelines/stages/glm_image_before_denoising.py` that loads
     dumped tensors from a hardcoded `/workspace/debug_dumps/sgl/...` path.
  4. SSIM regression sits at **0.715** on L40S
     (`fastvideo/tests/ssim/test_glm_image_similarity.py`) — too low to
     function as a real quality bar.
  5. T5 tokenizer is plain T5 instead of **ByT5** (diffusers reference uses
     `ByT5Tokenizer`).
  6. AR sampling temperature/top_p inconsistent between
     `glm_image_before_denoising.py` (`top_p=0.7, top_k=50, temperature=1.0`)
     and the pipeline config (`ar_top_p=0.75, ar_temperature=0.95`).
  7. Pipeline stages live under `fastvideo/pipelines/stages/glm_image_*.py`
     rather than `fastvideo/pipelines/basic/glm_image/stages/` per the
     `add-model` Files Map.
  8. Missing local parity tests (none under `tests/local_tests/glm_image/`,
     `transformers/`, `vaes/`, `encoders/`, or `pipelines/`).
- Required before handoff: non-skip PASS for each required component parity
  test, including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold, but final handoff requires non-skip
  PASS or an explicit blocker accepted through the escape-hatch process.
- User decisions and pause points are tracked as `E###` rows in
  `PORT_STATUS.md`; do not rely on chat history for escape-hatch context.
