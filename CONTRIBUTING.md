# Contributing to FastVideo

Thanks for helping. FastVideo takes fixes, model ports, performance work, tests
and docs. This page covers the common path. The longer guides live in
`docs/contributing/`, and the in-scope `AGENTS.md` next to the code lists that
area's conventions.

## What you need

- Python 3.10 to 3.12, on Linux or WSL.
- An NVIDIA GPU for inference and training. Some unit tests run on CPU.
- CUDA 12.6 or 13.0 for the CUDA path.
- macOS with Apple Silicon only if you are changing the MLX runtime.

The full checklist is in `docs/getting_started/installation/gpu.md`.

## Set up a development environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate

# cu130 on CUDA 13
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dev]"

# Optional: FlashAttention builds native kernels
uv pip install flash-attn --no-build-isolation -v

pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Conda works too: `conda create -n fastvideo python=3.12 -y`. Prefer `uv pip`
for the install inside the environment. For a container, see
`docs/contributing/developer_env/docker.md`.

## Run the checks

```bash
pre-commit run --all-files         # lint, format, types, spelling
pytest tests/                      # top-level tests
pytest fastvideo/tests/ -v         # package tests
```

Do not call yapf, ruff, mypy or codespell directly. Pre-commit chains them with
the project config and respects the excludes. SSIM and golden-gate tests need a
GPU and model weights; `docs/contributing/testing.md` explains how to pick the
narrowest one.

## Open a pull request

- Title with a bracketed tag such as `[feat]`, `[bugfix]`, `[perf]`,
  `[refactor]`, `[docs]`, `[ci]`, `[infra]` or `[new-model]`. The full accepted
  set is in `docs/contributing/pull_requests.md`.
- One concern per PR. Describe the problem, the change, how you verified it and
  the risk.
- After a maintainer approves, a contributor with write access comments
  `/merge` to enter the merge queue; external contributors ask a maintainer to
  run it or add the `ready` label. The path-aware merge gate runs only the lanes
  your diff needs.
- The full workflow, CI tiers and `/test` commands are in
  `docs/contributing/pull_requests.md`.

## Add a model or pipeline

`docs/contributing/coding_agents.md` walks through a new port with a reference
example, the files to touch and the acceptance criteria. Update
`docs/inference/support_matrix.md` when you add a model. New model PRs need the
serving YAML and a cookbook entry, or a short note on why they do not apply.

## Where to ask

- Questions and design ideas: [GitHub
  Discussions](https://github.com/hao-ai-lab/FastVideo/discussions), in Q&A or
  Ideas.
- Chat: the Slack invite is in the [README](README.md).
- Bugs and feature requests: use the issue templates.

## Code of conduct

Everyone participating in FastVideo is covered by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

Do not report vulnerabilities in public issues. See
[SECURITY.md](SECURITY.md).
