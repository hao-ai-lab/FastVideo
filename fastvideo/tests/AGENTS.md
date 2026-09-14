# `fastvideo/tests/` — Package-Level Test Suite

**Generated:** 2026-09-14

> **Pre-commit excludes `fastvideo/tests/`.** Lint/format hooks do not run on
> files here. Match style of neighboring tests manually.

## Layout

```
tests/
├── conftest.py            # distributed_setup fixture (1×1 SP/TP init+cleanup)
├── utils.py               # Shared test helpers
├── test_partial_hf_download.py  # Partial HF snapshot download behavior
├── api/                   # api/ schema + presets
├── attention/             # Backend selector + layer parity
├── audio/                 # Audio encoder/decoder tests (LTX-2 BWE)
├── contract/              # CI planner + cross-module shape contracts
├── dataset/               # Dataloader smoke tests
├── distributed/           # SP / TP collectives
├── encoders/              # Per-encoder parity (t5, clip, llama, qwen, ...)
├── entrypoints/           # CLI + streaming + OpenAI-compatible server
│   └── streaming/         #   Streaming-specific tests
├── eval/                  # Evaluation metric tests + reference scores
├── golden_gate/           # Model-family golden tests
├── hooks/                 # Runtime hook system
├── inference/             # End-to-end inference smoke
├── layers/                # Tensor-parallel layer unit tests
├── loader/                # Component loader / FSDP / LoRA patch tests
├── lora_extraction/       # LoRA merge/extract round-trip
├── mlx/                   # Apple MLX runtime tests
├── modal/                 # Dormant manual Modal rollback helpers + their unit tests
├── nightly/               # Long-running suites (gated)
├── ops/                   # Custom kernel / op tests
├── performance/           # Throughput / memory benchmarks (informational)
├── pipelines/             # Pipeline composition tests
├── platforms/             # Platform capability / unified-memory tests
├── ssim/                  # GPU SSIM regressions — see ssim/AGENTS.md
├── stages/                # Per-stage unit tests
├── train/                 # New modular trainer tests
├── training/              # Legacy training pipeline tests
├── transformers/          # transformers-shim tests
├── vaes/                  # Per-VAE encode/decode parity
├── worker/                # Worker / executor tests
└── workflow/              # Preprocessing workflow tests
```

## Run Commands (model-domain order)

```bash
pytest fastvideo/tests/ -v                        # All package tests
pytest fastvideo/tests/encoders/ -v               # One domain
pytest fastvideo/tests/ssim/ -vs                  # SSIM (GPU-heavy; see ssim/AGENTS.md)
pytest fastvideo/tests/ssim/ -vs --ssim-full-quality   # Full-quality SSIM params
pytest tests/ -v                                  # Top-level repo tests (different scope)
python fastvideo/tests/ssim/ci_runner.py          # Four-GPU Slurm-style SSIM scheduler
```

`tests/local_tests/` (top-level) holds component checks that need a local
working tree but are not part of the package suite.

## Conventions

- Name files `test_<feature>_<expected_behavior>.py`. Place near the domain
  (`tests/encoders/test_t5_*.py`, not `tests/test_everything.py`).
- Use the `distributed_setup` fixture for any test that touches
  `fastvideo.distributed.*`. It seeds torch + numpy and tears down the PG.
- GPU tests must `pytest.skip(...)` when hardware is missing — never `xfail`.
  Document `REQUIRED_GPUS` near the top for SSIM tests (see ssim/AGENTS.md).
- `nightly/` and `performance/` tests should be guarded by an environment marker
  so the default `pytest fastvideo/tests/` stays fast.

## CI Wiring

Adding a test file does not add it to CI. `.buildkite/scripts/unit_test.sh` is
an explicit file list, and each domain lane script (for example
`.buildkite/scripts/lanes/distillation_dmd.sh`) names its files. A new test
under an existing directory will not run in Fastcheck or the merge gate until
it is added to a lane script or a new lane is registered; see the
[testing guide](../../docs/contributing/testing.md).

CPU-only tests still import GPU backends transitively
(`attention/backends/vmoba.py` and `sla.py` import `fastvideo_kernel`), which
fails on a GPU-less host with `RuntimeError: 0 active drivers`. Stub
`fastvideo_kernel` (`moba_attn_varlen`, `process_moba_input`,
`process_moba_output`, and `triton_kernels.sla_triton._attention`) or run
inside the CI container.

## Slurm CI

`tests/ssim/ci_runner.py` orchestrates the SSIM lane. It auto-discovers
`test_*.py` files under `ssim/` and schedules one subprocess per `*_MODEL_TO_PARAMS`
key across the four GPUs granted by the Slinky Slurm lane. The change-aware
merge planner may pass repeated `--test-file` basenames for focused model-family
coverage; direct and scheduled full runs omit the filter. New SSIM tests need no
pipeline wiring — just declare `REQUIRED_GPUS = N`.

The rollback helper modules in `tests/modal/` are retained for dormant manual
rollback and must not be invoked by active Buildkite or slash-command routes;
their unit tests are still collected by `unit_test.sh`.

## Anti-Patterns

- Adding a hard-coded path to a model checkpoint without a corresponding
  `pytest.skip` when the path is missing.
- Forgetting `cleanup_dist_env_and_memory()` in tests that bypass the
  `distributed_setup` fixture — tears the next test in the worker.
- Putting Slurm host policy or credentials in repository-owned lane scripts.
