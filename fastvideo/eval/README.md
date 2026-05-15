# `fastvideo.eval`

In-process evaluation suite for video generations. Includes pixel
metrics (SSIM, PSNR, LPIPS), optical-flow comparisons, the full VBench
suite, Physics-IQ, audio metrics, and a VLM scorer behind a single
registry-driven API.

## Install

| Use case | Install |
|---|---|
| Default (common, optical_flow, vbench, physics_iq, videoscore2) | `uv pip install -e .[eval]` |
| Just VBench (11 of 16 by default; +4 with detectron2) | `uv pip install -e .[eval-vbench]` |
| Just Physics-IQ (covered by `[eval]`) | `uv pip install -e .[eval-physics-iq]` |
| Audio metrics (CLAP, FAD, KL, WER, AudioBox, DeSync, ImageBind) | `uv pip install -e .[eval-audio]` |
| Everything: `[eval]` + `[eval-audio]` + `vbench.scene` (AVoCaDO) | `uv pip install -e .[eval-full]` |

`[eval-audio]` covers every `audio.*` metric. ImageBind
(`facebookresearch/ImageBind`, CC BY-NC-SA 4.0) is git-sourced via
`[tool.uv.sources]` rather than vendored. `torchaudio` at the cu128
wheel is pulled transitively by `audiobox_aesthetics`; on cu128 hosts
using raw `pip`, install `torchaudio` from
`https://download.pytorch.org/whl/cu128` first.

`audio.desync` and `audio.wer (glm_asr)` import vendored upstream from
`fastvideo/third_party/eval/synchformer/` (MIT) and
`fastvideo/third_party/eval/glmasr/` (Apache-2.0). Both trees keep
their upstream `LICENSE` files alongside.

### `audio.*` metric input contracts

Every audio metric reads from these sample-dict keys (extra keys are
ignored):

| Metric | Per-sample? | Required keys |
|---|---|---|
| `audio.clap_score` | yes | `audio` (path), `text_prompt` (str) |
| `audio.audiobox_aesthetics` | yes | `audio` (path) |
| `audio.kl_divergence` | yes | `audio` (path), `reference_audio` (path) |
| `audio.frechet_distance` | **set-vs-set** | `audio` (path), `reference_audio` (path) — accumulated across ≥2 samples; `corpus["audio.frechet_distance"]` carries the score |
| `audio.wer` | yes | `audio` (path), `reference_text` (str) or `text_prompt` (str) |
| `audio.desync` | yes | `video` (decoded tensor or path), `audio` (path) |
| `audio.imagebind_score` | yes | `video_path` (str) **and** `audio` (path) — needs the path, not the pool-decoded tensor, because ImageBind's preprocessing decodes its own clips |

`audio.frechet_distance` is the only set-vs-set metric. The kwargs
form (`ev.evaluate(audio=...)`) raises with a clear message because a
single sample cannot produce a corpus result; use
`ev.evaluate(samples=[...])`.

### Reference repos for audio

The audio set ports its math 1:1 from `hkchengrex/av-benchmark` (the
V2A literature's de-facto eval harness — used by MMAudio, FoleyCrafter,
V2A-Mapper). Per-metric upstream:

| Metric | Upstream |
|---|---|
| `audio.frechet_distance` (PaSST-FAD) | `av_bench/metrics/fad.py::compute_fd` over `hear21passt` 768-d embeds |
| `audio.kl_divergence` | `av_bench/metrics/kl.py::compute_kl` over PaSST 527-d logits |
| `audio.clap_score` | HF `transformers.ClapModel` (`laion/clap-htsat-fused` — closest HF mirror of `630k-audioset-fusion-best`) |
| `audio.audiobox_aesthetics` | `facebookresearch/audiobox-aesthetics` (PQ as primary score, CE/CU/PC in details) |
| `audio.wer` | MagiHuman-style: NFKC + CJK char-level via `jiwer`, GLM-ASR or Whisper backbone |
| `audio.desync` | `av_bench/synchformer/` (vendored under `third_party/eval/synchformer/`); checkpoint from `hkchengrex/MMAudio/releases/v0.1/synchformer_state_dict.pth` |
| `audio.imagebind_score` | `facebookresearch/ImageBind` (`imagebind_huge` pretrained) |
| Plus `vbench.{color, multiple_objects, object_class, spatial_relationship}` (GRiT) | `uv pip install -e .[eval-vbench]` then `uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'` |

To use VBench, also pull the upstream submodule:

```bash
git submodule update --init --recursive  # fetches vbench + kernel deps
```

The submodule is a clean upstream pin. Compat with current
transformers/numpy/timm versions is applied at import time in
`fastvideo/eval/metrics/vbench/__init__.py` via attribute-level
monkey-patches; the submodule files are unchanged.

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

`evaluate` accepts either a pre-loaded `(T, C, H, W)` tensor or a path
string for `video` and `reference`. Paths are decoded inside the worker
that picks up the sample, so peak memory stays bounded by `num_gpus`
when scoring large batches.

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
a template for end-to-end "generate then score" pipelines.

## Layout

```
fastvideo/
├── eval/
│   ├── api.py, evaluator.py, registry.py, models.py, ...
│   ├── io/                        # video loading helpers
│   ├── datasets/                  # prompt corpora (vbench, physics_iq)
│   └── metrics/
│       ├── base.py                # BaseMetric + @register contract
│       ├── common/                # SSIM, PSNR, LPIPS
│       ├── optical_flow/          # gt_optical_flow, synthetic_optical_flow
│       ├── audio/                 # clap_score, audiobox_aesthetics, kl_divergence,
│       │                          # frechet_distance, wer, desync, imagebind_score
│       ├── videoscore2/           # VideoScore-2 (Qwen2.5-VL)
│       ├── physics_iq/            # PhysicsIQ + sub-metrics
│       └── vbench/                # adapter: sys.path bootstrap + shims
│           ├── __init__.py
│           └── <16 sub-metric pkgs>
└── third_party/
    └── eval/
        ├── vbench/                # git submodule (Vchitect/VBench)
        ├── synchformer/           # vendored (MIT), used by audio.desync
        └── glmasr/                # vendored (Apache-2.0), used by audio.wer (glm_asr)
```

### Prompt datasets

```python
from fastvideo.eval.datasets import get_dataset, list_datasets

list_datasets()                    # ['physics_iq', 'vbench']

ds = get_dataset("physics_iq", limit=4)   # auto-fetches assets on first miss
for row in ds:
    # row contains 'prompt', 'reference', 'reference_take2', and
    # metric-specific aux fields. Drop straight into Evaluator.evaluate(**row).
    ...
```

The Physics-IQ manifest CSV is vendored at
`fastvideo/eval/metrics/physics_iq/_vendored/descriptions.csv`.
Per-scenario videos, masks, and switch-frames auto-fetch on first use
into `${FASTVIDEO_EVAL_CACHE}/datasets/physics_iq/`. For air-gapped
runs, pass `auto_download=False` or `dataset_root=` a pre-downloaded
copy. Set `FASTVIDEO_PHYSICS_IQ_BUCKET_URL` to redirect the fetch to
an internal mirror.

## Adding a new metric

The full porting guide is at
[`docs/contributing/eval-metrics.md`](../../docs/contributing/eval-metrics.md).
Summary below.

### Native metric (no submodule)

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
    dependencies: list[str] = []  # e.g. ["pyiqa"] if relevant

    def compute(self, sample) -> MetricResult:
        ...
```

The metric is auto-discovered by `fastvideo/eval/metrics/__init__.py`,
which walks all non-underscore subdirectories and imports their
`metric` module.

### Wrapping upstream code

Three patterns coexist depending on how the upstream ships and what
licence it's under. All three keep upstream files on disk unmodified;
behavioural patches live as runtime shims in the consuming code.

1. **Git submodule** — large research packages pinned to a SHA, accessed
   via `sys.path` bootstrap. See `fastvideo/eval/metrics/vbench/` (with
   `fastvideo/third_party/eval/vbench/`).
2. **Vendored under `third_party/eval/<name>/`** — small/surgical upstream
   trees with permissive licences (MIT, Apache-2.0). See
   `fastvideo/third_party/eval/synchformer/` and `.../glmasr/`.
3. **Git-source via `[tool.uv.sources]`** — license-restricted upstream
   that cannot be redistributed in the FastVideo source tree. See
   ImageBind (CC BY-NC-SA 4.0) in `pyproject.toml`.

## Caches

Eval cache root: `${FASTVIDEO_CACHE_ROOT}/eval/`, default
`~/.cache/fastvideo/eval/`. Override with `FASTVIDEO_EVAL_CACHE`.

```
${FASTVIDEO_CACHE_ROOT}/eval/
├── models/      # URL-fetched checkpoints (LAION head, GRiT, Synchformer, ImageBind)
├── torch/       # redirected TORCH_HOME (DINO via torch.hub, lpips)
├── clip/        # passed as download_root= to clip.load callsites
└── datasets/    # auto-fetched dataset assets, one subdir per benchmark
                 # (e.g. datasets/physics_iq/{split-videos,switch-frames,...})
```

HF-hosted models stay in HF's default cache
(`~/.cache/huggingface/hub/`) so they dedupe with other ML projects on
the same host.

### Convention for new metrics

If your metric wraps a third-party loader that has its own cache
directory, route it through `get_cache_dir()` so users get one knob
to redirect everything.

```python
# CLIP: pass download_root explicitly
import clip
from fastvideo.eval.models import get_cache_dir
model, _ = clip.load("ViT-B/32", device=device,
                     download_root=str(get_cache_dir() / "clip"))

# torch.hub is already redirected by fastvideo.eval.__init__ via
# TORCH_HOME; no per-callsite work needed.

# transformers / huggingface_hub: leave alone. HF's default cache is
# shared with other tools.
```

For libraries that do not honour any env var or kwarg (pyiqa, funasr),
their cache lands in the library's own dir. Document the exception in
the metric's docstring if it matters.

## Out of scope (follow-up PRs)

- **MIND** metrics. Depend on a separate `vipe` upstream submodule.
- **VBench-2.0**. Sibling vbench2 package; needs its own port.
- **FVD as a registered metric**. Currently still at `benchmarks/fvd/`.
  FVD is a set-vs-set distribution distance and does not fit the
  per-sample `BaseMetric.compute` API without a stateful accumulator;
  conversion is a designed follow-up.
- **Training-time eval callback** (`EvalCallback`) and the
  `RolloutEvaluator` helper.
