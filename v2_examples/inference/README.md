# v2_examples/inference — Wan2.1-1.3B

Runnable, self-contained **inference** examples for the Wan2.1-1.3B *(recipe, runtime)* card on the
`v2` runtime. CPU + numpy only — the architecture (cards, driven loops, scheduler, caches, parity) is
real; the neural forwards are numpy toys (see [`../../v2/README.md`](../../v2/README.md) and
[`../../design_summary.md`](../../design_summary.md)). On a GPU box you swap `ComponentSpec.factory` for the torch
Wan adapter and **none of these scripts change**.

Each script bootstraps the repo onto `sys.path`, so run it from anywhere:

```bash
python3 v2_examples/inference/01_basic_t2v.py
```

| Script | What it shows |
|---|---|
| `01_basic_t2v.py` | the minimal path: build engine → typed T2V request → `run` → read the video artifact |
| `02_params_and_reproducibility.py` | `DiffusionParams` knobs (steps / CFG guidance / resolution) + seeded bit-identical reproducibility |
| `03_streaming.py` | per-denoise-step preview chunks via `OutputSpec(stream=...)` (off by default) |
| `04_concurrent_interleaved.py` | step-interleaved batching, the **interleave parity gate** (serial == interleaved), shared-prompt feature-cache reuse |
| `05_async_serving.py` | `AsyncEngine`: concurrent generate, live event streaming, step-boundary cancellation |

All five use only the public API (`v2.recipes.build_default_engine`, `v2.core.request.make_request`,
`v2.runtime.AsyncEngine`). For the OpenAI HTTP server, disaggregated pools, and the fleet/Dynamo path,
see `python3 -m v2.examples` (example *h*).
