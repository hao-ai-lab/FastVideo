---
description: How to develop, validate, and register a new evaluation metric
---

# Evaluation Development SOP

Standard procedure for adding new video quality evaluation metrics to
the FastVideo agent toolkit.

## When to use

- You need a metric that does not exist in
  `.agents/memory/evaluation-registry/README.md`.
- An existing metric needs significant changes to its methodology.
- You are exploring a new evaluation approach.

## Steps

### 1. Research

- Search `.agents/memory/related-work/` for existing evaluation
  approaches.
- Check `.agents/memory/evaluation-registry/README.md` for current
  metrics and their limitations.
- Review literature: FVD, CLIP-Score, human preference, etc.

### 2. Prototype

- Write a standalone script in `.agents/exploration/<metric-name>.md`.
- Keep it simple: one script, minimal dependencies.
- Test on a few known-good and known-bad video samples.

### 3. Validate

- **Known-good test**: metric should score high on reference-quality
  videos.
- **Known-bad test**: metric should score low on degraded or unrelated
  videos.
- **Sensitivity test**: small quality differences should produce
  meaningful score differences.
- Document thresholds and their justification.

### 4. Register

Update `.agents/memory/evaluation-registry/README.md`:

- Add the metric with status `Active`.
- Document location, thresholds, and trust level.

### 5. Integrate

Update `.agents/skills/evaluate-video-quality/SKILL.md`:

- Add the new metric as a section.
- Include code examples and interpretation guide.

### 6. Document

- Move the exploration log content into the skill.
- Clean up the exploration file or mark it as `promoted`.
- If anything went wrong during development, create a lesson.

## Where the metrics live

The eval suite is `fastvideo/eval/`. New metrics register themselves
via `@register("<group>.<name>")` and are auto-discovered when
`fastvideo.eval.metrics` is imported.

- **Native metrics** (SSIM, PSNR, LPIPS, optical flow, VLM): add a
  file under the appropriate group dir
  (`fastvideo/eval/metrics/common/`, `optical_flow/`, `videoscore2/`,
  `physics_iq/`).
- **Metrics that wrap upstream research code**: follow the vbench
  pattern in `fastvideo/eval/metrics/vbench/`. The contract is:
  - Upstream lives as a git submodule under
    `fastvideo/third_party/eval/<bench>/`, pinned to a SHA in repo-root
    `.gitmodules`.
  - The metric package's `__init__.py` inserts the submodule path on
    `sys.path` and installs runtime compat shims (attribute-level
    monkey-patches) for any modern-dep drift. Do not modify upstream
    files on disk, and do not ship a `setup.sh`.
  - See `fastvideo/eval/README.md` for the worked vbench example.
  - Full porting guide:
    [`docs/contributing/eval-metrics.md`](../../docs/contributing/eval-metrics.md).

## Out of scope of the initial eval port

The following land in follow-up PRs:

- **MIND** metrics (depends on a separate `vipe` submodule).
- **VBench-2.0** sibling package.
- Native conversion of **FVD** under `fastvideo/eval/metrics/fvd/`.
- The training-time `EvalCallback`.
