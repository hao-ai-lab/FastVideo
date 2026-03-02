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
