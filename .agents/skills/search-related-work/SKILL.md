---
name: search-related-work
description: Query the related work index for relevant papers, repos, or comparisons
---

# Search Related Work

## Purpose
Search through `.agents/memory/related_work/` to find indexed papers, repos,
or blog posts relevant to a query. Use this when you need to understand how
other work compares to FastVideo's approach, or when looking for techniques
to adopt.

## Prerequisites
- The related work index has entries (`.agents/memory/related_work/*.md`).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes | Natural language query |
| `tags` | No | Filter by tags (e.g., `[distillation, evaluation]`) |
| `type` | No | Filter by type (`paper`, `repo`, `blog`) |

## Steps

### 1. Search the index

Use grep-based search through `.agents/memory/related_work/`:

```bash
# Search by content
grep -rl "<query>" .agents/memory/related_work/

# Search by tags (in frontmatter)
grep -l "tags:.*<tag>" .agents/memory/related_work/*.md
```

### 2. Rank results

For each matching file:
1. Read the file.
2. Score relevance to the query based on:
   - Title match
   - Tag match
   - Content match (summary, differences, insights)
3. Return top results.

### 3. Format output

```markdown
## Related Work Search: "<query>"

### 1. <Title> (relevance: high)
- **Source**: <URL>
- **Tags**: <tags>
- **Key insight**: <most relevant excerpt>
- **File**: `.agents/memory/related_work/<slug>.md`

### 2. <Title> (relevance: medium)
...
```

## Outputs
- Ranked list of relevant related work entries with excerpts.

## Example Usage
```
Search for work related to video quality evaluation metrics:

  query: "video generation quality evaluation metrics"
  tags: [evaluation]
```

## References
- `.agents/memory/related_work/README.md` — index schema

## Changelog
| Date | Change |
|------|--------|
| 2026-03-02 | Initial version |
