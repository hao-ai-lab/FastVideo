# Plan: LTX-2 Model and Pipeline Port

## Goal
Port LTX-2 models and pipelines into FastVideo while reusing existing FastVideo layers wherever possible, and add numerical parity tests for each component.

## References & Constraints
- LTX-2 source: `../LTX-2/packages/ltx-core/` and `../LTX-2/packages/ltx-pipelines/`
- Existing parity pattern: `fastvideo/tests/transformers/test_wanvideo.py`
- Prior port template: `pr_883.diff`
- Tests may import directly from the local `../LTX-2/` repo for reference models.
- Weights will be provided manually (LTX-2 download done outside the repo).

## Porting Template (from `pr_883.diff`)
- Add model config under `fastvideo/configs/models/dits/` and register it in `__init__.py`.
- Add pipeline config under `fastvideo/configs/pipelines/` and register in `registry.py`.
- Implement native model classes under `fastvideo/models/` with parameter-name mapping for weight conversion.
- Add pipeline module under `fastvideo/pipelines/basic/<model>/` and any new stages under `fastvideo/pipelines/stages/`.
- Add conversion/validation scripts under `scripts/checkpoint_conversion/`.
- Add/adjust parity tests under `fastvideo/tests/`.

## Plan
1. **Inventory + mapping (LTX-2 -> FastVideo)**
   - Transformer: `ltx_core/model/transformer/` -> `fastvideo/models/dits/`
   - Video VAE encoder/decoder: `ltx_core/model/video_vae/` -> `fastvideo/models/vaes/`
   - Audio VAE decoder + vocoder: `ltx_core/model/audio_vae/` -> `fastvideo/models/audio/`
   - Text encoder (Gemma AV): `ltx_core/model/clip/gemma/` -> `fastvideo/models/encoders/`
   - Pipeline schedulers/guiders/patchifiers: `ltx_core/pipeline/components/` -> `fastvideo/pipelines/`
   - Identify which existing FastVideo layers can be reused vs. new LTX-2-specific blocks.

2. **Core component ports (one at a time)**
   - Add configs in `fastvideo/configs/models/` for each component.
   - Add loaders in `fastvideo/models/loader/` and register in `fastvideo/models/registry.py`.
   - Implement any new layers in `fastvideo/layers/` (reuse existing files or add generic ones).
   - Add weight conversion utilities for `.safetensors` (key remapping + splitting QKV/KV).
   - Use or extend conversion patterns from `scripts/checkpoint_conversion/wan_to_diffusers.py`.
   - Write helper scripts to download/inspect LTX-2 checkpoints and audit key names
     (initial scaffold: `scripts/checkpoint_conversion/convert_ltx2_weights.py`).

3. **Numerical parity tests (component-level)**
   - Add tests under `fastvideo/tests/` using the `test_wanvideo.py` pattern.
   - For each component: load FastVideo model and LTX-2 reference from `../LTX-2/`.
   - Compare parameter sums and forward outputs with `assert_close` (bf16/float16).
   - Keep tests minimal and deterministic; gate GPU-only tests as needed.

4. **Pipeline ports**
   - Implement pipelines in `fastvideo/pipelines/` mirroring LTX-2:
     - `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`, `DistilledPipeline`,
       `ICLoraPipeline`, `KeyframeInterpolationPipeline`.
   - Reuse FastVideo stages (CFG, scheduler, decoding) and add LTX-2-specific stages if required.
   - Add integration tests for shape/dtype determinism and minimal smoke runs.

5. **Docs + examples**
   - Add a short LTX-2 usage section to `README.md`.
   - Add one minimal example under `examples/ltx2/`.
   - Document required checkpoints and any env vars or attention backend flags.

## Open Questions (to resolve early)
- Which LTX-2 checkpoint(s) are first-class (dev vs distilled, FP8 vs FP16)?
- Should conversion scripts live per-component or a single `convert_ltx2_weights.py`?
- Where should model downloads live locally (e.g., `data/` vs `weights/`) to align with existing test expectations?

## Progress Log
- Added LTX-2 config in `fastvideo/configs/models/dits/ltx2.py`, registered in `fastvideo/configs/models/dits/__init__.py`.
- Added native LTX-2 transformer in `fastvideo/models/dits/ltx2.py` and registered it in `fastvideo/models/registry.py`.
- Implemented LTX-2-specific helpers directly in `fastvideo/models/dits/ltx2.py` (timestep embed, AdaLN, patchifier, FFN, rope).
- Switched LTX-2 attention to FastVideo `LocalAttention` and fixed cross-attn reshaping and output length handling.
- Added audio + AV cross-attention support (audio patchify/adaln/projection, AV gate/scale-shift, multimodal preprocessors).
- Fixed patch grid bounds creation (use `einops.repeat` vs `rearrange`).
- Added LTX-2 VAE wrappers + parity test (`fastvideo/models/vaes/ltx2vae.py`, `fastvideo/tests/vaes/test_ltx2_vae.py`), registered `LTX2VAEConfig` and `CausalVideoAutoencoder`.
- Added conversion script `scripts/checkpoint_conversion/convert_ltx2_weights.py` with metadata parsing and diffusers-style split output.
- Split `ltx-2-19b-distilled.safetensors` into component dirs (`transformer/`, `vae/`, `audio_vae/`, `vocoder/`, `text_embedding_projection/`).
- Routed `audio_embeddings_connector`/`video_embeddings_connector` into text encoder component during conversion.
- Expanded transformer config export to include audio + AV cross-attention fields.
- Updated LTX-2 param name mapping to prefix `model.` for component loader compatibility.
- Updated `fastvideo/tests/transformers/test_ltx2.py` to use `TransformerLoader`, load official LTX-2 reference, and gate CUDA-only.
- Added optional block-level sum logging in `fastvideo/tests/transformers/test_ltx2.py` via `LTX2_DEBUG_LOGS=1`.
- Added optional per-submodule sum logging in `fastvideo/tests/transformers/test_ltx2.py` via `LTX2_DEBUG_DETAIL=1`.
- Debug recipe: run parity test with `LTX2_DEBUG_LOGS=1` and compare `FastVideo/ltx2_debug/fastvideo.log` vs `FastVideo/ltx2_debug/reference.log` to find first divergent block; then rerun with `LTX2_DEBUG_DETAIL=1` to isolate attention/FFN divergence within that block.
- Fixes applied during debug: cross-attention reshaping uses `k_len` instead of `q_len`, output reshape uses `q_len`, and patch grid bounds uses `einops.repeat` to add batch dimension.
- Parity strategy: run both FastVideo and official LTX-2 with matched attention backends (set `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` and `LTX2_REFERENCE_ATTN=pytorch` for SDPA parity).
- Verified transformer parity passes when both sides use SDPA with debug logging enabled.
- Added audio-only parity test `fastvideo/tests/transformers/test_ltx2_audio.py` that compares audio outputs using the same loader/reference setup and debug logging.
- Added Gemma AV text encoder config in `fastvideo/configs/models/encoders/gemma.py` and exported it from `fastvideo/configs/models/encoders/__init__.py`.
- Implemented `fastvideo/models/encoders/gemma.py` with LTX-2 feature extractor, connector blocks, and lazy Gemma model loading.
- Registered `LTX2GemmaTextEncoderModel` in `fastvideo/models/registry.py`.
- Updated `scripts/checkpoint_conversion/convert_ltx2_weights.py` to emit a `text_embedding_projection/config.json` for the Gemma text encoder.
- Added text encoder parity test `fastvideo/tests/encoders/test_ltx2_gemma_encoder.py` comparing FastVideo vs LTX-2 outputs (video + audio encodings) with local Gemma weights.
- Added LTX-2 pipeline config `fastvideo/configs/pipelines/ltx2.py` wiring `LTX2GemmaConfig` into text encoding and defaulting to LTX2 DiT/VAE configs.
- Registered LTX-2 pipeline config in `fastvideo/configs/pipelines/__init__.py` and `fastvideo/configs/pipelines/registry.py` with `ltx2` detection fallback.
- Added pipeline smoke test `fastvideo/tests/pipelines/test_ltx2_pipeline_smoke.py` to load the LTX-2 transformer and Gemma text encoder via `LTX2T2VConfig` (env-gated).
- Downloaded Gemma 3 text encoder weights per official instructions into `weights/gemma-3-12b-it-qat-q4_0-unquantized` (local only).
- Fixed LTX-2 text encoder parity by honoring attention masks in `LTXSelfAttention`; masked path now reuses FastVideo attention modules (SDPA backend).
- Verified `fastvideo/tests/encoders/test_ltx2_gemma_encoder.py` passes with CUDA + local Gemma weights.
- Fixed VAE parity test to remap `vae.per_channel_statistics.*` keys and confirmed `fastvideo/tests/vaes/test_ltx2_vae.py` passes against the distilled weights.
- Added LTX-2 audio VAE + vocoder wrappers (`fastvideo/models/audio/ltx2_audio_vae.py`) and parity test (`fastvideo/tests/vaes/test_ltx2_audio_vae.py`) with weight remapping for `audio_vae.*`/`vocoder.*`.
- Audio VAE/vocoder parity test passes with `torchaudio` installed.
- Added audio model configs (`fastvideo/configs/models/audio/ltx2_audio_vae.py`) and registered `LTX2AudioEncoder`/`LTX2AudioDecoder`/`LTX2Vocoder` in `fastvideo/models/registry.py`.
- Wired LTX-2 pipeline config with audio decoder/vocoder config + precision fields and expanded pipeline smoke test to load audio decoder/vocoder weights and run a tiny decode/vocode forward.
- Updated `scripts/checkpoint_conversion/convert_ltx2_weights.py` to emit a diffusers-style repo layout with `model_index.json` and a `text_encoder/` mirror of Gemma projection weights.
- Updated LTX-2 pipeline smoke test to target the diffusers-style repo root (`LTX2_DIFFUSERS_PATH`) and require `model_index.json`.
- Updated `examples/inference/basic/basic_ltx2.py` to use the diffusers-style repo root and exercise transformer + text encoder + audio decoder/vocoder loads.
- Added LTX-2 pipeline under `fastvideo/pipelines/basic/ltx2/` with LTX2-specific latent prep + denoising stages and standard decoding.
- Added LTX2 sigma schedule + Euler denoising stage (`fastvideo/pipelines/stages/ltx2_denoising.py`) and LTX2 latent preparation stage (`fastvideo/pipelines/stages/ltx2_latent_preparation.py`).
- Updated component loader to support `audio_vae`/`audio_decoder` and `vocoder` modules.
- Updated LTX2 VAE loader to accept LTX-2-style configs (`{"vae": ...}`) without arch overrides.
- Updated LTX-2 conversion script to wrap audio/vae configs with `_class_name` and include tokenizer/audio/vocoder entries in `model_index.json`.
- Updated LTX-2 smoke test config parsing and rewired `basic_ltx2.py` to use the LTX2 pipeline via `VideoGenerator`.
- Added optional Gemma bundling (`--gemma-path`) in the LTX-2 conversion script and repo-local `gemma/` fallback in loaders/examples.
- Switched Gemma bundling to `text_encoder/gemma/` and loader fallback now resolves within `text_encoder/`.
- Updated parity tests to use diffusers component paths (via `LTX2_DIFFUSERS_PATH`) while comparing against official LTX-2 weights.
- Removed `FASTVIDEO_LIGHT_IMPORT` from LTX-2 parity tests; recommend editable install + `PYTHONPATH` for pytest.
- Removed `LTX2_GEMMA_MODEL_PATH` override and now rely on `text_encoder/gemma` for Gemma weights in loaders/tests/examples.
- Added LTX-2 sampling defaults and temp path mapping for `ltx2_diffusers` in pipeline + sampling registries.
- Fixed Gemma tokenizer padding to use `padding="max_length"` to match LTX-2 and avoid connector register assertion.
- Reworked `fastvideo/tests/transformers/test_ltx2.py` to load the official transformer using `SingleGPUModelBuilder` + `LTXModelConfigurator` + `LTXV_MODEL_COMFY_RENAMING_MAP` (metadata-based config).
- Added VAE parity test using official LTX-2 loaders (`fastvideo/tests/vaes/test_ltx2_vae_official.py`).
- Added LTX-2 denoising + VAE diagnostics (NaN detection + verbose stats) gated by env flags.
