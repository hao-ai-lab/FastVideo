---
description: How to develop, validate, and register a new evaluation metric
---

# Evaluation Development SOP

Standard procedure for adding new video quality evaluation metrics to the
FastVideo agent toolkit.

## When to Use

- You need a metric that doesn't exist in `.agents/memory/evaluation-registry/README.md`.
- An existing metric needs significant changes to its methodology.
- You're exploring a new evaluation approach.

## Steps

### 1. Research

- Search `.agents/memory/related-work/` for existing evaluation approaches.
- Check the `evaluation_registry.md` for current metrics and their limitations.
- Review literature: FVD, CLIP-Score, human preference, etc.

### 2. Prototype

- Write a standalone script in `.agents/exploration/<metric-name>.md`.
- Keep it simple: one script, minimal dependencies.
- Test on a few known-good and known-bad video samples.

### 3. Validate

- **Known-good test**: Metric should score high on reference-quality videos.
- **Known-bad test**: Metric should score low on degraded/unrelated videos.
- **Sensitivity test**: Small quality differences should produce meaningful
  score differences.
- Document thresholds and their justification.

### 4. Register

Update `.agents/memory/evaluation-registry/README.md`:
- Add the metric with status `Active`.
- Document location, thresholds, and trust level.

### 5. Integrate

Update `.agents/skills/evaluate-video-quality.md`:
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

- **Native metrics** (SSIM, PSNR, LPIPS, optical flow, audio, VLM):
  drop a file under the appropriate group dir
  (`fastvideo/eval/metrics/common/`, `audio/`, `vlm/`).
- **Metrics that wrap upstream research code**: follow the vbench
  pattern in `fastvideo/eval/metrics/vbench/`:
  - upstream is a git submodule under `<metric>/external/upstream/`,
    pinned to a SHA in repo-root `.gitmodules`;
  - the metric package's `__init__.py` does the `sys.path` insert AND
    installs runtime compat shims (attribute-level monkey-patches) for
    any modern-dep drift. **Do not** modify upstream files on disk and
    **do not** ship a `setup.sh`.
  - See `fastvideo/eval/README.md` for the contract and the worked
    vbench example.
  - Full porting guide:
    [`docs/contributing/eval-metrics.md`](../../docs/contributing/eval-metrics.md).

## Out of scope of the initial eval port

Landing in follow-up PRs: **MIND** metrics (vipe submodule),
**VBench-2.0** sibling package, native conversion of **FVD** under
`fastvideo/eval/metrics/fvd/`, and the training-time `EvalCallback`.
