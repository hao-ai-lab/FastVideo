---
name: index-related-work
description: Ingest a paper or repository into the related work index
---

# Index Related Work

## Purpose
Create a structured summary of a related paper, repository, or blog post and
add it to `.agents/memory/related_work/` for future reference. This builds the
agent's knowledge base for making informed decisions about training, evaluation,
and architecture choices.

## Prerequisites
- Access to the paper/repo (URL, PDF, or local clone).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | Yes | URL, citation, or local path |
| `type` | Yes | `paper`, `repo`, or `blog` |
| `tags` | No | List of tags (default: inferred from content) |

## Steps

### 1. Extract key information

For **papers**: Read abstract, method section, experimental setup, and results.
For **repos**: Read README, key source files, and training scripts.
For **blogs**: Read the full post.

Focus on:
- What problem does it solve?
- What architecture/technique is used?
- How does it relate to FastVideo's approach?

### 2. Create the index entry

Write to `.agents/memory/related_work/<slug>.md`:

```markdown
---
title: <title>
source: <URL or citation>
type: paper | repo | blog
date_indexed: <ISO-8601>
tags: [world-model, distillation, evaluation, ...]
---

## Summary
<1-2 paragraph summary.>

## Key Differences from FastVideo
- <comparison points>

## Actionable Insights
- <what we could adopt or adapt>
```

### 3. Update the catalog

If `.agents/memory/related_work/_catalog.md` exists, append the new entry.
If not, create it:

```markdown
# Related Work Catalog

| Slug | Title | Type | Tags | Date |
|------|-------|------|------|------|
| <slug> | <title> | <type> | <tags> | <date> |
```

## Outputs
- New file in `.agents/memory/related_work/<slug>.md`.
- Updated catalog.

## Example Usage
```
Index the Self-Forcing paper:

  source: https://arxiv.org/abs/2406.xxxxx
  type: paper
  tags: [world-model, self-forcing, distillation]
```

## References
- `.agents/memory/related_work/README.md` — schema documentation

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
