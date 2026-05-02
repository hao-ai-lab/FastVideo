# FastLTX-2.3 Gradio local demo

Local Gradio + FastAPI demo for FastLTX-2.3 text-to-video generation. This
directory is a package (`gradio_local_demo_ltx2_3/`) split out from the
original single-file version to make each concern independently reviewable.
The folder name matches the sibling `gradio_local_demo*.py` demos in this
directory so both flat and packaged demos read consistently.

> **Status: draft.** This package is structurally in place but will not run
> against the current upstream `fastvideo` package. See *Blocking prereqs*
> below.

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | `main()` — CLI args, `VideoGenerator` / `SamplingParam` boot, FastAPI mount with logo/favicon/generated-clip routes, `uvicorn.run`. |
| `config.py` | Constants, defaults, env-var resolution, Inductor tuning flags, `setup_model_environment`, `resolve_model_path`, `resolve_refine_upsampler_path`, `apply_ltx2_defaults`. |
| `safety.py` | fastText NSFW + hate-speech classifiers, `PromptSafetyCheck`, `get_prompt_safety_check`. |
| `prompt_rewrite.py` | Cerebras-backed prompt enhancer wrapper (`maybe_enhance_prompt`, `get_prompt_enhancer`). Curated prompts bypass enhancement. |
| `prompt_enhancer.py` | Cerebras API client used by `prompt_rewrite.py`. |
| `rendering.py` | HTML helpers: timing cards, error cards, completed-clip gallery, image-upload status. |
| `examples.py` | `load_example_prompts` — reads `selected_ltx2_prompts.jsonl`. |
| `ui.py` | `create_gradio_interface` — Gradio Blocks, CSS, event wiring, generation closure. |
| `__main__.py` | Enables `python -m gradio_local_demo_ltx2_3`. |
| `__init__.py` | Re-exports `main` from `app`. |
| `selected_ltx2_prompts.jsonl` | Curated example prompts. |
| `prompts/prompt_extension_system_prompt.md` | System prompt for the Cerebras enhancer. |
| `download_fasttext_classifiers.py` | Helper to download NSFW/hate-speech classifier binaries from Hugging Face Hub. |

## How to run

```bash
cd examples/inference/gradio/local
python -m gradio_local_demo_ltx2_3 --port 7860
```

GPU requirement: a single FP4-capable GPU (B200 or comparable) for the
"real-time 1080p" speed claim. Lower tiers will still run but slower.

## Blocking prereqs (why this draft PR cannot be merged yet)

The upstream `fastvideo` package is missing three pieces that the demo
currently depends on verbatim. Each needs its own upstreaming PR before this
demo can actually boot:

1. **`fastvideo.layers.quantization.fp4_config.FP4Config`** — the demo sets
   `pipeline_config.dit_config.quant_config = FP4Config()` in `app.py`.
   Upstream only ships `absmax_fp8.py` and `base_config.py` under
   `fastvideo/layers/quantization/`.
2. **LTX-2.3 refine / image-conditioning kwargs on `VideoGenerator`** —
   `ltx2_refine_enabled`, `ltx2_refine_upsampler_path`, `ltx2_refine_lora_path`,
   `ltx2_refine_num_inference_steps`, `ltx2_refine_guidance_scale`,
   `ltx2_refine_add_noise`, `ltx2_images`, `ltx2_image_crf`. Upstream
   `fastvideo/fastvideo_args.py` currently wires only `ltx2_vae_tiling`.
   The backing stages (`ltx2_refine.py`, `ltx2_i2v_conditioning.py`) are
   also missing from `fastvideo/pipelines/stages/`.
3. **`fastvideo.configs.sample.base.SamplingParam`** — the import path used
   by this demo. Upstream moved sampling params to
   `fastvideo.api.sampling_param`. A re-export shim at the old path, or an
   import update here once the other two prereqs land, will resolve it.

## Environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `LTX2_3_MODEL_PATH` | `FastVideo/LTX-2.3-Distilled-Diffusers` | Model ID or local snapshot. |
| `LTX2_CLASSIFIER_DIR` | package dir | Where to look for fastText classifiers. |
| `LTX2_NSFW_CLASSIFIER_PATH` | — | Explicit path to NSFW classifier `.bin`. |
| `LTX2_HATESPEECH_CLASSIFIER_PATH` | — | Explicit path to hate-speech classifier `.bin`. |
| `LTX2_REFINE_UPSAMPLER_PATH` | — | Explicit path to the spatial upsampler dir. |
| `FASTVIDEO_PROMPT_API_KEY` / `CEREBRAS_API_KEY` | — | Cerebras API key for prompt enhancement. When missing, enhancer returns the raw prompt. |
| `LTX2_PROMPT_MODEL` | `gpt-oss-120b` | Cerebras model name for the enhancer. |
| `LTX2_PROMPT_TEMPERATURE` | `1.0` | Enhancer LLM temperature. |
| `LTX2_PROMPT_EXTENSION_SYSTEM_PROMPT_PATH` | `prompts/prompt_extension_system_prompt.md` | System prompt for the enhancer. |

## Fetching the safety classifiers

```bash
python examples/inference/gradio/local/gradio_local_demo_ltx2_3/download_fasttext_classifiers.py
```

The classifiers come from `allenai/dolma-jigsaw-fasttext-bigrams-{nsfw,hatespeech}`
on Hugging Face Hub.
