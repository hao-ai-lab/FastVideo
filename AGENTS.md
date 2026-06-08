# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `fastvideo/` (models, pipelines, training, distributed runtime, CLI entrypoints).
- CUDA/custom kernels: `fastvideo-kernel/` (separate build/test flow).
- Tests:
  - `fastvideo/tests/` for package-level tests (dataset, encoders, inference, training, SSIM, workflow).
  - `tests/local_tests/` for additional local/component checks.
- Docs and guides: `docs/` (MkDocs source), with contributor docs in `docs/contributing/`.
- Runnable examples and scripts: `examples/` and `scripts/`.
- Static assets: `assets/` (including `assets/images/`, `assets/videos/`, and `assets/prompts/`) and `comfyui/assets/`.

## Build, Test, and Development Commands
- `uv pip install -e ".[dev]"`: editable install with lint/test extras.
- `pre-commit install --hook-type pre-commit --hook-type commit-msg`: enable local hooks.
- `pre-commit run --all-files`: run formatter/lint/type/spelling checks.
- `pytest tests/`: run top-level test suite.
- `pytest fastvideo/tests/ -v`: run package tests.
- `pytest fastvideo/tests/ssim/ -vs`: run SSIM regression tests (GPU-heavy).
- `cd fastvideo-kernel && ./build.sh`: build kernel extensions.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; keep code and imports readable and explicit.
- Style tools are configured in `pyproject.toml` and `.pre-commit-config.yaml`:
  - `yapf` (format), `ruff` (lint, auto-fix), `mypy` (typing), `codespell`.
- Lint via `pre-commit run --files <changed paths>` (or `pre-commit run --all-files` for a full sweep) before committing. Do not shell out to `yapf`/`ruff`/`codespell`/`mypy` directly — pre-commit chains them with the project's config and respects the `.pre-commit-config.yaml` excludes (e.g. `fastvideo/tests/` is intentionally skipped). If pre-commit reports `(no files to check)` for your paths, that exclude is deliberate — don't bypass it.
- Target line length is 120 (configured in `pyproject.toml` for ruff, yapf, and isort).
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Use `pytest` and place tests near relevant domains (e.g., `fastvideo/tests/encoders/`).
- Prefer descriptive names like `test_<feature>_<expected_behavior>.py`.
- For new pipelines/backends, include at least one regression-oriented test; add SSIM coverage when output quality must be preserved.
- Document GPU assumptions in tests that require specific hardware.

## Commit & Pull Request Guidelines
- Follow existing commit style: short subject with optional tag prefix, e.g. `[bugfix]: ...`, `[feat]: ...`, `[misc]: ...`, and include PR reference like `(#1234)` when applicable.
- Keep commits focused by concern (feature, refactor, fix).
- PRs should include:
  - clear problem/solution summary,
  - test evidence (`pytest`/SSIM outputs or rationale if skipped),
  - linked issue/PR context,
  - screenshots or sample outputs for UI/demo/docs changes.

## Agent Infrastructure

This repository is agent-friendly. Before doing any work, read:

1. `.agents/onboarding/README.md` — full onboarding guide with step-by-step instructions.
2. `.agents/memory/codebase-map/README.md` — structural index of the entire repository.
3. `.agents/skills/` — available agent skills (check if one exists before writing code).
4. `.agents/workflows/` — SOPs for common procedures (experiment lifecycle, evaluation, etc.).
5. `.agents/lessons/` — known pitfalls and their documented fixes.

If you are exploring a new procedure that has no existing SOP, document your
progress in `.agents/exploration/` and flag it for review at the end of your
session.

## Per-Directory AGENTS.md

Local guidance lives next to the code. Read the in-scope file before editing:

| Directory | What it covers |
|-----------|----------------|
| `fastvideo/AGENTS.md` | Core package map, public API, registry-driven model dispatch |
| `fastvideo/configs/AGENTS.md` | Arch + pipeline config dataclasses, `param_names_mapping` |
| `fastvideo/models/AGENTS.md` | DiT / VAE / encoder / scheduler / loader layout (pre-commit excluded) |
| `fastvideo/layers/AGENTS.md` | Tensor-parallel linear/attention layer rules for ports |
| `fastvideo/attention/AGENTS.md` | Backend registry + env-var override |
| `fastvideo/pipelines/AGENTS.md` | Stage ABC, `basic/<model>/`, `preprocess/`, presets |
| `fastvideo/training/AGENTS.md` | Legacy monolithic pipelines (frozen for existing models) |
| `fastvideo/train/AGENTS.md` | New modular trainer (methods × models × callbacks, YAML) |
| `fastvideo/tests/AGENTS.md` | Test taxonomy, conftest, pre-commit-excluded path |
| `fastvideo/tests/ssim/AGENTS.md` | GPU SSIM regression authoring + reference video sync |
| `scripts/checkpoint_conversion/AGENTS.md` | Adding a converter for a new HF/official checkpoint |

## Critical: Two Training Stacks Coexist

- `fastvideo/training/` — legacy, monolithic per-model `*_training_pipeline.py` and
  `*_distillation_pipeline.py`. Still authoritative for shipped models.
- `fastvideo/train/` — new modular framework (composable methods × models × callbacks
  driven by YAML). Preferred for new training work.

Pick the matching stack before editing. Do not migrate a pipeline between them
without an explicit ask — the conventions and config surfaces differ.
