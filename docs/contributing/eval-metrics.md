# Porting Eval Metrics into `fastvideo.eval`

This guide is for contributors adding new evaluation metrics to FastVideo's
eval suite. If you just want to *run* the existing metrics, see
[`fastvideo/eval/README.md`](../../fastvideo/eval/README.md).

## When you need this guide

Use this guide when you are:

- Adding a new metric (native or wrapping a third-party library).
- Porting a benchmark (e.g. VBench, MIND, EvalCrafter) whose Python code
  needs to be importable from a pinned upstream.
- Adding a new metric group (audio, vlm, etc.).

## TL;DR

Metrics are auto-discovered from
`fastvideo/eval/metrics/<group>/<name>/metric.py`. Each declares itself
with `@register("<group>.<name>")` and subclasses `BaseMetric`. Three
recipes:

1. **Native metric** (pure-PyTorch, no submodule). Drop a file, declare
   deps, implement `compute(sample)`.
2. **Library-wrapped metric** (CLIP, torch.hub, transformers, pyiqa).
   Same as above, plus route the library's cache through `get_cache_dir()`
   if it has a `download_root=` / `cache_dir=` kwarg.
3. **Upstream-submodule-wrapped metric** (vbench-style). Pin upstream as
   a git submodule under `fastvideo/third_party/eval/<bench>/`. Adapter
   `__init__.py` does the `sys.path` insert + any runtime compat shims
   for modern dep versions. **No `setup.sh`. No on-disk patches.**

The full recipes are below.

---

## 0) Layout & auto-discovery

```
fastvideo/eval/metrics/
├── base.py                  # BaseMetric + lifecycle contract
├── common/                  # group: SSIM, PSNR, LPIPS, optical_flow
├── audio/                   # group: CLAP, WER, audiobox, …
├── vlm/                     # group: VideoScore-2
├── physics_iq/              # group + sub-metrics
└── vbench/                  # group: 16 sub-metrics
    ├── __init__.py          # sys.path bootstrap + runtime compat shims
    ├── _grit_helper.py      # shared upstream-touching helpers
    └── <sub_metric>/metric.py
```

Auto-discovery (`fastvideo/eval/metrics/__init__.py`) walks each group
dir and imports every `metric.py` it finds, which fires the `@register`
decorators. **Names starting with `_` are skipped** — use that prefix
for shared helpers or vendored code that shouldn't register itself.

---

## 1) The `BaseMetric` contract

Every metric subclasses `fastvideo.eval.metrics.base.BaseMetric` and
declares:

```python
class YourMetric(BaseMetric):
    name: str = "common.your_metric"          # must match @register
    requires_reference: bool = True           # needs sample["reference"]
    higher_is_better: bool = True             # for ranking / aggregates
    dependencies: list[str] = []              # importable module names;
                                              # registry surfaces clean
                                              # ImportError if missing
    needs_gpu: bool = False
    backbone: str | None = None               # e.g. "clip_vit_l14"
    batch_unit: str = "video"                 # "video", "frame", or
                                              # "frame_pair"
```

You must implement:

```python
def compute(self, sample: dict) -> list[MetricResult]:
    """sample['video'] is (B, T, C, H, W). Return one MetricResult per B."""
```

You may override:

- `setup(self) -> None` — eager model loading. Called once by
  `create_evaluator`. Idempotent (re-entrant). Use the `if self._model
  is not None: return` pattern.
- `to(self, device)` — move the metric (and its submodels) to `device`.
- `trial_forward(self, batch_size, *, height, width, num_frames)` — used
  by `Evaluator.calibrate()`. Default runs `compute()` on dummies; override
  if your `batch_unit` isn't `"video"` or your inputs need special shapes.

If a required input is missing (e.g. an fps-aware metric called without
`fps`), return `self._skip(sample, reason)` — **do not raise**.

---

## 2) Recipe A — Native metric (no external deps)

Smallest case. Pixel math, simple closed-form.

```python
# fastvideo/eval/metrics/common/your_metric/metric.py
from __future__ import annotations
import torch
from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("common.your_metric")
class YourMetric(BaseMetric):
    name = "common.your_metric"
    requires_reference = True
    higher_is_better = True
    needs_gpu = False
    batch_unit = "video"
    dependencies: list[str] = []  # nothing extra

    def compute(self, sample: dict) -> list[MetricResult]:
        gen, ref = sample["video"], sample["reference"]   # (B,T,C,H,W) each
        per_video = ((gen - ref) ** 2).mean(dim=(1, 2, 3, 4)).sqrt()
        return [
            MetricResult(name=self.name, score=float(s), details={})
            for s in per_video
        ]
```

That's it. Drop the file, the registry picks it up.

---

## 3) Recipe B — Library-wrapped metric (CLIP, torch.hub, transformers, pyiqa)

If your metric loads a backbone from a Python package, **you are
responsible for pointing the library at the eval cache** so users get
one knob (`FASTVIDEO_EVAL_CACHE`) to redirect everything.

### Cache routing rules

| library | how to route | location after redirect |
|---|---|---|
| `clip.load("ViT-X")` | pass `download_root=str(get_cache_dir() / "clip")` | `${FASTVIDEO_EVAL_CACHE}/clip/` |
| `torch.hub.load(...)` | nothing — `TORCH_HOME` is redirected at `fastvideo.eval` import time | `${FASTVIDEO_EVAL_CACHE}/torch/hub/` |
| `transformers.from_pretrained(...)` | nothing — leave HF's default cache (`~/.cache/huggingface/hub/`) so users dedupe with other ML projects | `~/.cache/huggingface/hub/` |
| `huggingface_hub.snapshot_download` / `hf_hub_download` | use `ensure_checkpoint(...)` (it wraps these with filelock) | same as above |
| `pyiqa.create_metric(...)` | no env var or kwarg honored — accept the spread, document in metric docstring | pyiqa-internal |
| `lpips`, `ptlflow` | torch.hub-based — auto-redirected | `${FASTVIDEO_EVAL_CACHE}/torch/hub/` |
| Raw URL (no HF Hub) | use `ensure_checkpoint(name, source="https://...")` | `${FASTVIDEO_EVAL_CACHE}/models/<name>` |

### Example — CLIP backbone + LAION head

```python
# fastvideo/eval/metrics/your_group/your_metric/metric.py
from __future__ import annotations
import torch
import torch.nn as nn
from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("your_group.your_metric")
class YourMetric(BaseMetric):
    name = "your_group.your_metric"
    requires_reference = False
    needs_gpu = True
    dependencies = ["clip"]      # "openai-clip" PyPI; importable as `clip`

    def __init__(self) -> None:
        super().__init__()
        self._clip = None
        self._head = None

    def setup(self) -> None:
        if self._clip is not None:
            return
        import clip
        from fastvideo.eval.models import ensure_checkpoint, get_cache_dir

        # Backbone: route CLIP's cache through our root.
        self._clip, _ = clip.load(
            "ViT-L/14",
            device=self.device,
            download_root=str(get_cache_dir() / "clip"),
        )
        self._clip.eval()

        # URL-fetched head: ensure_checkpoint downloads to
        # ${FASTVIDEO_EVAL_CACHE}/models/ with filelock + atomic rename.
        ckpt = ensure_checkpoint(
            "your_head.pth",
            source="https://example.com/path/to/your_head.pth",
        )
        self._head = nn.Linear(768, 1)
        self._head.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )
        self._head.to(self.device).eval()

    def to(self, device):
        super().to(device)
        if self._clip is not None:
            self._clip = self._clip.to(self.device)
        if self._head is not None:
            self._head = self._head.to(self.device)
        return self

    def compute(self, sample: dict) -> list[MetricResult]:
        ...
```

### Don't reach into other `~/.cache/...` dirs

If a third-party library hard-codes `~/.cache/<lib>/` and offers no
override, **don't try to redirect it** by setting `os.environ` or
patching `os.path.expanduser`. Just document the exception in the
metric's docstring. Forcing redirection is fragile and breaks user
expectations of where the library's cache lives.

---

## 4) Recipe C — Upstream-submodule-wrapped metric (vbench pattern)

Use this when the upstream benchmark ships Python code (`vbench/`,
`MIND/`, etc.) that's not pip-installable cleanly and we need to import
it. The pattern (see `fastvideo/eval/metrics/vbench/__init__.py` for the
full worked example):

### 4.1 Pin the upstream as a submodule

```bash
git submodule add <upstream-url> fastvideo/third_party/eval/<bench>
cd fastvideo/third_party/eval/<bench>
git checkout <pinned-sha>
cd -
git add .gitmodules fastvideo/third_party/eval/<bench>
```

The submodule pulls under the standard `git submodule update --init
--recursive` flow that users run for kernel deps. **No special opt-in
flag, no `update = none`, no setup script.**

### 4.2 Bootstrap on `sys.path`

```python
# fastvideo/eval/metrics/<bench>/__init__.py
from __future__ import annotations
import sys
from pathlib import Path

# fastvideo/eval/metrics/<bench>/__init__.py → ../../../../third_party/eval/<bench>
_UPSTREAM = Path(__file__).resolve().parents[3] / "third_party" / "eval" / "<bench>"
if _UPSTREAM.is_dir() and str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
```

We don't `pip install` the upstream — its egg-link/.pth would just
re-do this `sys.path.insert`. Skipping the install also skips the
upstream's `setup.py` (which often gates on a specific CUDA version).

### 4.3 Modern-dep compat: runtime shims, NOT on-disk patches

Upstream code pinned to e.g. `transformers==4.33.2`, `numpy<2` typically
breaks against modern versions in 3-4 known places (API renames). Fix
those at import time, in the same `__init__.py`:

```python
def _install_compat_shims() -> None:
    # Example: transformers.modeling_utils API moved.
    try:
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
        for _n in ("apply_chunking_to_forward",
                   "find_pruneable_heads_and_indices",
                   "prune_linear_layer"):
            if not hasattr(_mu, _n) and hasattr(_pu, _n):
                setattr(_mu, _n, getattr(_pu, _n))
    except ImportError:
        pass

    # Example: numpy.lib.function_base.disp removed in numpy>=2.
    try:
        import types, numpy.lib as _nl
        if not hasattr(_nl, "function_base"):
            _stub = types.ModuleType("numpy.lib.function_base")
            _stub.disp = lambda *a, **k: None
            sys.modules["numpy.lib.function_base"] = _stub
            _nl.function_base = _stub
    except ImportError:
        pass

_install_compat_shims()
```

For function-level patches that can't be expressed as attribute writes
(e.g. wrapping a model factory function), use a `sys.meta_path` finder
that wraps the loader. See the `_install_modeling_finetune_hook()`
helper in `fastvideo/eval/metrics/vbench/__init__.py` for the pattern —
about 30 lines.

**Why shims and not `git apply` patches:** patches go stale (new upstream
SHA → patches don't apply); shims are versioned Python code in our
repo, grep-able, and only ever run if the targeted module is being
imported.

### 4.4 Per-sub-metric files

Each sub-metric is a normal `BaseMetric` subclass that imports from the
upstream:

```python
# fastvideo/eval/metrics/<bench>/<sub>/metric.py
from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("<bench>.<sub>")
class YourSubMetric(BaseMetric):
    ...

    def setup(self) -> None:
        if self._model is not None:
            return
        # The sys.path bootstrap fired when fastvideo.eval.metrics.<bench>
        # was imported (which auto-discovery does before importing this
        # sub-package). So upstream imports just work:
        from <bench>.something import SomeModel
        ...
```

### 4.5 Conditional registration when upstream is missing

If a user installed `fastvideo[eval]` but didn't run `git submodule
update --init`, `<bench>.*` metrics should *not* register. The
auto-discovery walker imports each sub-package's `metric` module; have
that import bail out cleanly:

```python
# fastvideo/eval/metrics/<bench>/__init__.py — at the bottom
_AVAILABLE = (_UPSTREAM / "<bench>" / "__init__.py").is_file()
```

```python
# fastvideo/eval/metrics/<bench>/<sub>/__init__.py
from fastvideo.eval.metrics.<bench> import _AVAILABLE
if _AVAILABLE:
    from .metric import YourSubMetric  # noqa
```

`fastvideo eval list` then reflects what the user actually has, not what
they could have.

### 4.6 Don't fix upstream bugs unless they block the metric

The upstream is pinned. If a metric works against the pinned SHA, leave
it. If it actively breaks against modern deps (the import-drift cases
above), shim it. Resist the urge to "improve" upstream code in a
fastvideo-side fork — that's how patches go stale and parity drifts.

---

## 5) Model checkpoints — `ensure_checkpoint`

Use `ensure_checkpoint(name, source, filename=None)` for any non-package
weights. It resolves a local path, downloading on miss, with filelock
safety across processes and SLURM ranks.

| `source` form | what happens |
|---|---|
| `"/abs/path/to/file.pth"` | passthrough — returned unchanged |
| `"https://..."` | downloaded to `${FASTVIDEO_EVAL_CACHE}/models/<name>` via `huggingface_hub.http_get`, atomic rename, filelock |
| `"org/repo"` (no `filename`) | `snapshot_download(repo_id)` → `~/.cache/huggingface/hub/` |
| `"org/repo"` (with `filename`) | `hf_hub_download(repo_id, filename)` → `~/.cache/huggingface/hub/` |

`name` is only used as the local filename for URL sources; HF sources
ignore it (HF manages its own cache key by content hash).

```python
from fastvideo.eval.models import ensure_checkpoint

# URL — name matters
ckpt = ensure_checkpoint(
    "amt-s.pth",
    source="https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
)

# HF single file — name is decorative
ckpt = ensure_checkpoint(
    "raft-things.pth",                      # ignored; HF cache uses repo+sha
    source="OpenGVLab/VBench_Used_Models",
    filename="raft-things.pth",
)
```

---

## 6) Declaring `dependencies`

Set `dependencies = ["pkg1", "pkg2"]` on your metric class (importable
module names, not PyPI distribution names). The registry checks each via
`importlib.util.find_spec` at instantiation time and raises a clean
ImportError pointing the user at `pip install fastvideo[eval]`:

```python
class YourMetric(BaseMetric):
    dependencies = ["clip", "timm"]   # importable as `import clip`, `import timm`
```

If a dep is in `[project.optional-dependencies.eval]`, you don't need to
do anything more. If it's a new dep, **add it to that group** in
`pyproject.toml`.

---

## 7) Common gotchas

- **Do not write a `setup.sh`** for your benchmark. Submodule init via
  the standard `git submodule update --init --recursive` is enough; any
  modern-dep compat goes into your `__init__.py` as runtime shims.
- **Do not modify upstream files on disk.** The submodule should always
  match its pinned SHA. Compat lives in our `__init__.py`.
- **Do not pip-install the upstream.** The egg-link is just a glorified
  `sys.path.insert` — we do that directly in `__init__.py`.
- **Don't `import torch.hub.set_dir(...)` from your metric.** It's done
  globally in `fastvideo/eval/__init__.py`.
- **Don't put cache-redirection env vars in your metric's `setup()`.**
  By the time `setup()` runs, the library has likely already cached the
  default-location decision. Set env vars at package-init time only.
- **Skip rather than raise** when an input is missing. Use
  `self._skip(sample, reason)` for any expected-missing input — it
  returns a list of `MetricResult(score=None)` so other metrics in the
  same evaluator continue.
- **Watch for upstream re-registration conflicts.** If the upstream uses
  a global registry (detectron2's `META_ARCH_REGISTRY`, MMCV, etc.),
  loading the same model twice in the same process will throw. The
  evaluator already loads each metric once; if you write a custom
  setup-then-call pattern, mirror that single-load discipline.

---

## 8) Training-time eval — keep evaluators hot, free caches between calls

When wiring eval into a training loop, the working pattern is:

1. Construct the ``Evaluator`` **once**, attach it to the pipeline
   (``self._eval = create_evaluator(...)``). Don't recreate it per
   validation round — that re-pays the model load cost.
2. Save validation videos to disk (the diffusion path already does
   this). Pass paths to ``evaluator.evaluate``, not in-memory tensors
   that share GPU memory with the training model.
3. Run validation **only on rank 0** of each sequence-parallel group;
   gather paths from other ranks and let rank 0 score everything.
4. After every ``evaluate(...)`` call, call
   ``evaluator.release_cuda_memory()`` in a ``finally`` block —
   ``gc.collect()`` + ``torch.cuda.empty_cache()`` +
   ``torch.cuda.ipc_collect()``. The eval model stays loaded; only
   transient activation buffers from the just-finished call get freed::

       for video_path in batch:
           try:
               scores = self._eval.evaluate(video=load_video(video_path))
           finally:
               self._eval.release_cuda_memory()

5. If memory pressure spikes (rare on H200), call
   ``evaluator.unload()`` to drop every metric reference and let the
   GPU memory be GC'd. ``unload`` is reversible: ``evaluator.reload()``
   rebuilds the same metrics with the original config (re-paying the
   model load cost). Calling ``evaluate`` between ``unload`` and
   ``reload`` raises a clear ``RuntimeError``.

This is the pattern in
``fastvideo.training.{ptlflow,vbench}_validation`` (mhuo's branch).
For most metrics (sub-1 GB backbones, e.g. CLIP/DINO/RAFT/AMT) the
eval model can stay co-resident with the training model in
``transformer.eval()`` mode without any swap. For larger ones
(VideoScore2 at 14 GB), measure first; if it fits on the rank-0 GPU
during validation (training model in eval mode = no grads/optimizer
updates), keep it hot. If not, ``unload`` between rounds.

## 9) Local verification

Native and library-wrapped metrics — single-GPU smoke is enough:

```python
import torch
from fastvideo.eval import create_evaluator

ev = create_evaluator(metrics=["<group>.<your_metric>"], device="cuda")
video = torch.randn(1, 49, 3, 256, 256, device="cuda").clamp(0, 1)
print(ev.evaluate(video=video))
```

Submodule-wrapped metrics — also do a parity check against the upstream
once. The pattern from `research/` (see commit history) gives a clean
recipe: clone upstream into a separate venv, run the same video through
both, expect EXACT match on bit-deterministic metrics and ≤1% drift on
backbone-heavy ones (driven by transformers/torch version differences).

For quick-and-dirty parity in CI: pin a tiny test video, record expected
scores ± tolerance, and add a calibration test under
`fastvideo/tests/eval/`.

---

## 10) When *not* to add a metric

- **Set-vs-set distribution metrics** (FVD, FID-style) don't fit
  `BaseMetric.compute(sample)` cleanly — they need a population. Adding
  them requires a stateful accumulator interface that doesn't exist
  yet. Open an issue first.
- **Metrics requiring a single-GPU model > available memory.** Eval is
  not the place for tensor-parallel sharding; metrics are expected to
  fit on one GPU.
- **Metrics that need `mmcv` with conflicting CUDA ABI.** Document the
  affected sub-metrics as unsupported and skip them — do not build
  isolation infrastructure (subprocess engine, per-metric venv).
