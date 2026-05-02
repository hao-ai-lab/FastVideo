# `fastvideo/tests/` — Package-Level Test Suite

> **Pre-commit excludes `fastvideo/tests/`.** Lint/format hooks do not run on
> files here. Match style of neighboring tests manually.

## Layout

```
tests/
├── conftest.py            # distributed_setup fixture (1×1 SP/TP init+cleanup)
├── utils.py               # Shared test helpers
├── api/                   # api/ schema + presets
├── attention/             # Backend selector + layer parity
├── dataset/               # Dataloader smoke tests
├── distributed/           # SP / TP collectives
├── encoders/              # Per-encoder parity (t5, clip, llama, qwen, ...)
├── entrypoints/           # CLI + streaming + OpenAI-compatible server
│   └── streaming/         #   Streaming-specific tests
├── hooks/                 # Runtime hook system
├── inference/             # End-to-end inference smoke
├── lora_extraction/       # LoRA merge/extract round-trip
├── modal/                 # Modal CI orchestrators (ssim_test.py, nightly_test.py)
├── nightly/               # Long-running suites (gated)
├── ops/                   # Custom kernel / op tests
├── performance/           # Throughput / memory benchmarks (informational)
├── ssim/                  # GPU SSIM regressions — see ssim/AGENTS.md
├── stages/                # Per-stage unit tests
├── train/                 # New modular trainer tests
├── training/              # Legacy training pipeline tests
├── transformers/          # transformers-shim tests
├── vaes/                  # Per-VAE encode/decode parity
└── workflow/              # Preprocessing workflow tests
```

## Run Commands (model-domain order)

```bash
pytest fastvideo/tests/ -v                        # All package tests
pytest fastvideo/tests/encoders/ -v               # One domain
pytest fastvideo/tests/ssim/ -vs                  # SSIM (GPU-heavy; see ssim/AGENTS.md)
pytest fastvideo/tests/ssim/ -vs --ssim-full-quality   # Full-quality SSIM params
pytest tests/ -v                                  # Top-level repo tests (different scope)
modal run fastvideo/tests/modal/ssim_test.py      # Orchestrated CI-style SSIM run
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

## Modal CI

`tests/modal/ssim_test.py` is the orchestrator that auto-discovers
`test_*.py` files under `ssim/` and schedules one subprocess per `*_MODEL_TO_PARAMS`
key. New SSIM tests need no CI wiring — just declare `REQUIRED_GPUS = N`.

## Anti-Patterns

- Adding a hard-coded path to a model checkpoint without a corresponding
  `pytest.skip` when the path is missing.
- Forgetting `cleanup_dist_env_and_memory()` in tests that bypass the
  `distributed_setup` fixture — tears the next test in the worker.
- Putting Modal-specific code outside `tests/modal/`.
