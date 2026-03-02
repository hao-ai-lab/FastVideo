# Evaluation Metrics Registry

Living catalog of all evaluation metrics available for video quality assessment.
Updated as new metrics are developed or existing ones evolve.

_Last updated: 2026-03-02_

## Active Metrics

| Metric | Type | Location | Status | Trust Level |
|--------|------|----------|--------|-------------|
| SSIM | Reference comparison | `fastvideo/tests/ssim/` | ✅ Active | High — well-tested in CI |
| Loss trajectory | Training signal | W&B `train_loss` field | ✅ Active | Medium — proxy, not direct quality |
| Grad norm stability | Training signal | W&B `grad_norm` field | ✅ Active | Medium — diagnostics |

## Draft Metrics

| Metric | Type | Location | Status | Notes |
|--------|------|----------|--------|-------|
| Caption consistency | LLM-based | `.agents/skills/evaluate-video-quality.md` | 🟡 Draft | Not yet calibrated against human judgments |

## Planned Metrics

| Metric | Type | Priority | Notes |
|--------|------|----------|-------|
| FVD (Fréchet Video Distance) | Distribution-based | Medium | Requires large sample set |
| CLIP-Score | Embedding similarity | Medium | Text-video alignment |
| Human preference | Manual | Low | Ground truth, expensive |

## How Metrics Evolve

This registry reflects the team's experience that evaluation metrics change
during a project's lifecycle:

1. **Early stage**: Loss is the only signal (often flat). Focus on video quality
   via visual inspection and SSIM against reference.
2. **Mid stage**: Add automated metrics (caption consistency, LLM-based scoring).
3. **Late stage**: Loss becomes meaningful. Layer in distributional metrics (FVD)
   and human evaluation.

When a metric status changes, update this file and the `evaluate-video-quality`
skill accordingly.

## Adding a New Metric

1. Follow the SOP in `.agents/workflows/evaluation-development.md`.
2. Update this registry with the new metric.
3. Update the `evaluate-video-quality` skill.
