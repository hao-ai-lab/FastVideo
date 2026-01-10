# Repository Guidelines

## Project Structure & Module Organization
- `fastvideo/`: core Python package (models, pipelines, entrypoints).
- `fastvideo/pipelines/`: composed pipelines + stages; `fastvideo/pipelines/basic/` houses per-arch pipelines.
- `fastvideo/tests/`: primary pytest suite grouped by domain (encoders, transformers, ssim, training, inference).
- `fastvideo-kernel/`: kernel implementations and CUDA/Triton-related code; its own tests live in `fastvideo-kernel/tests/`.
- `examples/` and `demo/`: runnable scripts and showcases.
- `docs/`: documentation source; `docs/contributing/testing.md` is the testing guide.
- `assets/`, `data/`, `images/`, `outputs*/`: large assets and generated artifacts (avoid committing new outputs).
- `LTX-2/`: local upstream repo used for parity/reference in LTX-2 tests.
- `converted/`: diffusers-style converted weights (local, not committed).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: install editable package with lint/test extras.
- `fastvideo --help`: verify CLI entrypoint is available.
- `python examples/inference/basic/basic_dmd.py`: run a local inference script (see README’s minimal example).
- `pytest fastvideo/tests/ssim -vs`: run SSIM regression tests (GPU required).
- `pytest fastvideo/tests/encoders -v`: run a smaller component test slice.
- LTX-2 parity tests (CUDA required) assume `LTX2_DIFFUSERS_PATH` points to a diffusers-style repo root and `LTX2_OFFICIAL_PATH` points to the official safetensors file.
- If pytest can’t find local `fastvideo` modules, run `pip install -e ".[dev]"` and set `PYTHONPATH` to the repo root.

## Coding Style & Naming Conventions
- Python is primary; follow 4-space indentation and keep lines near 80 chars (see `pyproject.toml` tool settings).
- Lint/format tools referenced in config: `ruff`, `yapf`, `isort`, and `mypy`.
- Naming: modules and functions use `snake_case`; classes use `CamelCase`; tests use `test_*.py`.

## Testing Guidelines
- Framework: `pytest`.
- Test categories live under `fastvideo/tests/` (unit, component, SSIM, training, inference).
- SSIM tests require reference videos under `fastvideo/tests/ssim/*_reference_videos/` and are documented in `docs/contributing/testing.md`.
- LTX-2 parity tests compare FastVideo modules against `LTX-2/` reference implementations and official weights.

## Commit & Pull Request Guidelines
- Commit messages commonly use bracketed tags, e.g. `[bugfix]`, `[docs]`, `[chore]`.
- PRs should describe the change, link related issues, and note hardware/GPU assumptions.
- Include test evidence or explain why tests were skipped (e.g., GPU-only workloads).

## Configuration Tips
- Inference backends are controlled via env vars like `FASTVIDEO_ATTENTION_BACKEND` (see README example).
- Avoid adding large model weights; reference hosted checkpoints instead.
- LTX-2 diffusers layout expects `model_index.json` at the repo root, component folders (`transformer/`, `vae/`, `audio_vae/`, `vocoder/`, `text_encoder/`), and Gemma under `text_encoder/gemma/`.
