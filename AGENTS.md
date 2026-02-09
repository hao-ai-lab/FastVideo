# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `fastvideo/` (models, pipelines, training, distributed runtime, CLI entrypoints).
- CUDA/custom kernels: `fastvideo-kernel/` (separate build/test flow).
- Tests:
  - `fastvideo/tests/` for package-level tests (dataset, encoders, inference, training, SSIM, workflow).
  - `tests/local_tests/` for additional local/component checks.
- Docs and guides: `docs/` (MkDocs source), with contributor docs in `docs/contributing/`.
- Runnable examples and scripts: `examples/` and `scripts/`.
- Static assets: `assets/`, `images/`, `videos/`, and `comfyui/assets/`.

## Build, Test, and Development Commands
- `uv pip install -e .[dev]`: editable install with lint/test extras.
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
- Target line length is 80.
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Use `pytest` and place tests near relevant domains (e.g., `fastvideo/tests/encoders/`).
- Prefer descriptive names like `test_<feature>_<expected_behavior>.py`.
- For new pipelines/backends, include at least one regression-oriented test; add SSIM coverage when output quality must be preserved.
- Document GPU assumptions in tests that require specific hardware.

## Commit & Pull Request Guidelines
- Follow existing commit style: short subject with optional tag prefix, e.g. `[bugfix]: ...`, `[feat] ...`, `[misc] ...`, and include PR reference like `(#1234)` when applicable.
- Keep commits focused by concern (feature, refactor, fix).
- PRs should include:
  - clear problem/solution summary,
  - test evidence (`pytest`/SSIM outputs or rationale if skipped),
  - linked issue/PR context,
  - screenshots or sample outputs for UI/demo/docs changes.
