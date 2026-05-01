# `fastvideo.eval`

In-process evaluation suite for video and world-model generations. Native
metrics (SSIM, PSNR, LPIPS, optical flow), audio metrics, the full VBench
suite, physics_iq, and a small VLM scorer all live behind a single
registry-driven API.

## Install

Eval lives in FastVideo's main venv. Add the eval extras:

```bash
pip install -e .[eval]
```

To use VBench specifically, also pull the upstream submodule:

```bash
git submodule update --init --recursive  # fetches vbench + kernel deps
```

That's the entire setup. There is no eval-specific bootstrap script.
The submodule is a clean upstream pin; modern-dep compat is achieved
at import time, in `fastvideo/eval/metrics/vbench/__init__.py`, by
attribute-level monkey-patches against `transformers`, `numpy`, and
`timm`. Nothing on disk in the submodule is modified.

## Public API

```python
from fastvideo.eval import (
    create_evaluator,        # build a reusable Evaluator
    evaluate,                # one-shot helper
    Evaluator,               # the class itself
    BaseMetric, MetricResult,
    register, list_metrics, get_metric,
    ensure_checkpoint, get_cache_dir,
)

ev = create_evaluator(metrics=["common.ssim", "vbench.aesthetic_quality"],
                      device="cuda")
scores = ev.evaluate(video=tensor, reference=ref, fps=8.0)
```

### CLI

```bash
fastvideo eval list                              # list registered metrics
fastvideo eval list --group vbench               # filter by group
fastvideo eval run --videos clip.mp4 \
    --metrics vbench.aesthetic_quality \
    --output scores.json
fastvideo eval run --videos generated/*.mp4 \
    --reference reference/ \
    --metrics common.ssim,common.lpips
```

### Generate-then-score example

`examples/inference/eval/eval_ltx2_vbench.py` runs an LTX2 prompt
through `VideoGenerator` and scores the resulting mp4 with
`vbench.aesthetic_quality` and `vbench.subject_consistency`. Use it as
a template for end-to-end "generate → score" pipelines.

## Layout

```
fastvideo/
├── eval/
│   ├── api.py, evaluator.py, registry.py, models.py, ...
│   ├── io/                        # video loading helpers
│   └── metrics/
│       ├── base.py                # BaseMetric + @register contract
│       ├── common/                # SSIM, PSNR, LPIPS, optical_flow
│       ├── audio/                 # CLAP, WER, audiobox, ...
│       ├── vlm/                   # VideoScore-2
│       ├── physics_iq/            # PhysicsIQ + sub-metrics
│       └── vbench/                # ← adapter: sys.path bootstrap + shims
│           ├── __init__.py
│           └── <16 sub-metric pkgs>
└── third_party/
    └── eval/
        └── vbench/                # ← git submodule (Vchitect/VBench)
```

## Adding a new metric

The full porting guide is at
[`docs/contributing/eval-metrics.md`](../../docs/contributing/eval-metrics.md).
Summary below.

### Native, no submodule

```python
# fastvideo/eval/metrics/common/<your_metric>/metric.py
from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("common.your_metric")
class YourMetric(BaseMetric):
    name = "common.your_metric"
    requires_reference = True
    needs_gpu = False
    dependencies: list[str] = []  # add e.g. ["pyiqa"] if relevant

    def compute(self, sample) -> list[MetricResult]:
        ...
```

The metric is auto-discovered by `fastvideo/eval/metrics/__init__.py`
(walks all non-underscore subdirectories and imports their `metric`
module).

### With a submodule for upstream code

Pattern: see `fastvideo/eval/metrics/vbench/`. The contract is:

1. `<bench>/external/upstream/` is a git submodule pinned to a SHA
   (registered in repo-root `.gitmodules`).
2. `<bench>/__init__.py` inserts that path on `sys.path` and installs
   any compat shims (attribute-level monkey-patches) needed for modern
   torch/transformers/numpy. **Do not modify upstream files on disk.**
3. Per-sub-metric `metric.py` files use `@register("<bench>.<name>")`.

There is no per-benchmark setup script. Patches live as Python in
the metric's `__init__.py`, where they are grep-able and reviewable.

## Caches

Eval cache root: `${FASTVIDEO_CACHE_ROOT}/eval/` (default
`~/.cache/fastvideo/eval/`). Override with `FASTVIDEO_EVAL_CACHE`.

```
${FASTVIDEO_CACHE_ROOT}/eval/
├── models/      # URL-fetched checkpoints (LAION head, AMT, GRiT)
├── torch/       # redirected TORCH_HOME (DINO via torch.hub, lpips)
└── clip/        # passed as download_root= to clip.load callsites
```

HF-hosted models stay in HF's default cache (`~/.cache/huggingface/hub/`)
so they dedupe with other ML projects on the same host.

### Convention for new metrics

If your metric wraps a third-party loader that has its own cache
directory, **route it through `get_cache_dir()`** so users get one knob:

```python
# CLIP — pass download_root explicitly
import clip
from fastvideo.eval.models import get_cache_dir
model, _ = clip.load("ViT-B/32", device=device,
                     download_root=str(get_cache_dir() / "clip"))

# torch.hub — already redirected by fastvideo.eval.__init__
# via TORCH_HOME; no per-callsite work needed.

# transformers / huggingface_hub — leave alone; HF's default cache
# is shared with other tools.
```

Libraries with non-standard caches that don't honour any env var or
kwarg (pyiqa, funasr) currently land in their own dirs. Acceptable for
now; document in the metric's docstring if it matters.

## Out of scope (follow-up PRs)

- **MIND** metrics — depend on a separate `vipe` upstream submodule.
- **VBench-2.0** — sibling vbench2 package; needs its own port.
- **FVD as a registered metric** — currently still at
  `benchmarks/fvd/`; FVD is fundamentally a set-vs-set distribution
  distance and doesn't fit the per-sample `BaseMetric.compute` API
  without a stateful accumulator. Conversion is a designed follow-up.
- **Training-time eval callback** (`EvalCallback`) and the
  `RolloutEvaluator` helper.
