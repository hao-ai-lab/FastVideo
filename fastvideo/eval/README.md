# `fastvideo.eval`

In-process evaluation suite for video generations. Includes pixel
metrics (SSIM, PSNR, LPIPS), Fréchet Video Distance (FVD), optical-flow
comparisons, the full VBench suite, Physics-IQ, audio metrics, an
absolute VLM scorer (`videoscore2`), and a pairwise VLM judge
(`judge.third_person_separation`) — all behind a single registry-driven API.

## Install

| Use case | Install |
|---|---|
| Default (common, optical_flow, vbench, physics_iq, videoscore2) | `uv pip install -e .[eval]` |
| Just VBench (11 of 16 by default; +4 with detectron2) | `uv pip install -e .[eval-vbench]` |
| Just Physics-IQ (covered by `[eval]`) | `uv pip install -e .[eval-physics-iq]` |
| Audio metrics (CLAP, FAD, KL, WER, AudioBox, DeSync, ImageBind) | `uv pip install -e .[eval-audio]` |
| Everything: `[eval]` + `[eval-audio]` + `vbench.scene` (AVoCaDO) | `uv pip install -e .[eval-full]` |
| Optional faster video decode (x86_64 only; opt-in) | `uv pip install -e .[eval-fast-decode]` |

`[eval-audio]` covers every `audio.*` metric. ImageBind
(`facebookresearch/ImageBind`, CC BY-NC-SA 4.0) is git-sourced via
`[tool.uv.sources]` rather than vendored. `torchaudio` at the cu128
wheel is pulled transitively by `audiobox_aesthetics`; on cu128 hosts
using raw `pip`, install `torchaudio` from
`https://download.pytorch.org/whl/cu128` first.

`[eval-fast-decode]` is opt-in. It pulls `decord`, which is faster than
the default PyAV decoder but has no aarch64 wheels and is effectively
unmaintained upstream. The default `[eval]` install uses PyAV (via
`av`); decord only matters if you've measured decode as the bottleneck
on x86_64. `audio.imagebind_score` declares a hard `decord` dependency
and will skip without it.

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
    samples_from, as_video,  # path-style inputs → canonical samples list
    BaseMetric, MetricResult,
    register, list_metrics, get_metric,
    ensure_checkpoint, get_cache_dir,
)

# Single-sample form (per-sample metrics):
ev = create_evaluator(metrics=["common.ssim", "vbench.aesthetic_quality"],
                      device="cuda")
scores = ev.evaluate(video=tensor, reference=ref, fps=8.0)

# Many samples from two directories (paired metrics + corpus metrics):
results = ev.evaluate(samples=samples_from(video="gen/", reference="ref/"))
# results[i]["common.ssim"]              → per-pair SSIM
# results.corpus["common.fvd"]           → corpus FVD (if FVD is registered)
```

`samples_from` turns path-style inputs into the canonical samples list.
Cardinality determines the shape: equal `|gen| == |ref|` zips into paired
samples; unequal cardinality role-tags the unmatched references as
trailing set-style samples so corpus metrics like FVD see the full
reference set while paired metrics like LPIPS only score the first N
pairs. Per-sample attachments (`text_prompt(s)`, `fps`,
`auxiliary_info`, `extras=`) attach by kwarg; `extract_audio=True` pulls
audio tracks off video sources for audio metrics.

`evaluate` also accepts a pre-loaded `(T, C, H, W)` tensor or a path
string under `video` / `reference`. Paths are decoded inside the worker
that picks up the sample, so peak memory stays bounded by `num_gpus`
when scoring large batches. Use `evaluate(samples=..., metrics=[...])`
to run a subset of the Evaluator's registered metrics on this batch
(useful for scoring different corpora with different metric subsets in
successive calls without burning model loads).

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
│       ├── common/                # SSIM, PSNR, LPIPS, FVD
│       ├── optical_flow/          # gt_optical_flow, synthetic_optical_flow
│       ├── audio/                 # clap_score, audiobox_aesthetics, kl_divergence,
│       │                          # frechet_distance, wer, desync, imagebind_score
│       ├── videoscore2/           # VideoScore-2 (Qwen2.5-VL)
│       ├── judge/                 # pairwise VLM judges (third_person_separation)
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

## `common.fvd` — Fréchet Video Distance

Set-vs-set metric. Computes the Fréchet distance between Gaussian
moments of I3D features (Kinetics-400) over the generated set and a
reference set. Lower is better; standard protocol uses 2048 videos
and a warning fires below 256.

```python
from fastvideo.eval import create_evaluator, samples_from

ev = create_evaluator(metrics=["common.fvd"], device="cuda")
result = ev.evaluate(samples=samples_from(
    video="gen/", reference="ref/",
)).corpus["common.fvd"]
```

`samples_from` zips the directories into paired samples when their
cardinalities match (FVD pulls features from both `sample["video"]` and
`sample["reference"]`); when `|ref| > |gen|`, the unmatched references
become role-tagged trailing samples so FVD still sees the full
reference corpus while paired per-sample metrics (LPIPS / PSNR / SSIM)
running alongside only score the matched pairs.

Reference features are cached to
``${FASTVIDEO_EVAL_CACHE}/fvd/real_features_{extractor}.pt`` the first
time reference features are streamed; subsequent runs load the cache
automatically (omit `reference=` to score new generations against the
cached set). Override with `$FASTVIDEO_FVD_REF_FEATURES`, the
`cache_path=` constructor kwarg, or `cache_mode={"off","read","read_write"}`
to control read/write behavior. The example script
`examples/inference/eval/eval_fvd.py` demonstrates the full
two-directory workflow.

## `judge.third_person_separation` — pairwise VLM judge

A **preference** metric (a judge, not an absolute score), and the suite's first
remote-API one. For each pair the judge (Gemini) sees the shared first frame and
two rollouts — a candidate and a reference model under the same control signal —
and picks the one that better separates the third-person CHARACTER (foreground)
from the BACKGROUND. The corpus score is the candidate's win-rate, excluding
ties. Set-vs-set, motion-first; it reads native mp4s, so samples carry path
strings, not decoded tensors.

```bash
uv pip install -e .[eval-judge]      # opt-in: needs network + an API key
export GEMINI_API_KEY=...            # or GOOGLE_API_KEY, or ~/.gemini_token
```

```python
from fastvideo.eval import create_evaluator

ev = create_evaluator(metrics=["judge.third_person_separation"], device="cpu")
result = ev.evaluate(samples=[
    {"video_path": "cand/000.mp4", "reference_path": "base/000.mp4",
     "image_path": "frames/000.png", "text_prompt": "W: moves forward", "action": "W"},
    # ... more pairs ...
]).corpus["judge.third_person_separation"]
result.score    # candidate win-rate excl. ties; result.details has the breakdown
```

Only `video_path`/`reference_path` are required; `image_path`/`text_prompt`/
`action` are optional. Verdicts are cached under `${FASTVIDEO_EVAL_CACHE}/eval/judge/`.
The judge separates best when the control yields genuine parallax (e.g.
translation); rigid whole-frame motion (e.g. pure camera rotation) is harder. To
sweep several baselines into a table, see
`examples/inference/eval/eval_third_person_separation.py`.

## Out of scope (follow-up PRs)

- **MIND** metrics. Depend on a separate `vipe` upstream submodule.
- **VBench-2.0**. Sibling vbench2 package; needs its own port.
- **Training-time eval callback** (`EvalCallback`) and the
  `RolloutEvaluator` helper.
