---
description: Synchronize the STATUS.md dashboard by scanning .agent/ directories
---

# Sync Dashboard

Updates `.agent/STATUS.md` by scanning the skills, workflows, memory, lessons,
and exploration directories to reflect what actually exists on disk.

## When to Use

- After adding, removing, or renaming any file in `.agent/`.
- Periodically (e.g., at end of each conversation session).
- When the dashboard feels out of date.

## Steps

### 1. Scan directories

List all files in each directory:

```bash
echo "=== Skills ==="
ls -1 .agent/skills/*.md 2>/dev/null | grep -v SKILL_TEMPLATE

echo "=== Workflows ==="
ls -1 .agent/workflows/*.md 2>/dev/null

echo "=== Memory ==="
ls -1 .agent/memory/*.md 2>/dev/null
ls -1 .agent/memory/related_work/*.md 2>/dev/null | grep -v README

echo "=== Lessons ==="
ls -1 .agent/lessons/*.md 2>/dev/null | grep -v README

echo "=== Exploration ==="
ls -1 .agent/exploration/*.md 2>/dev/null | grep -v README
```

### 2. Compare with STATUS.md

For each file found:
- If it's in STATUS.md → leave it (preserve status/trust/tested fields).
- If it's NOT in STATUS.md → add it with status `🔴 Stub`, trust `None`, tested `❌`.

For each entry in STATUS.md:
- If the file no longer exists → mark it as `❌ Removed` or delete the row.

### 3. Update counts

Recalculate the summary table at the top:
- Count files per category.
- Count by status (Ready, Draft, Stub).

### 4. Update timestamp

Set `_Last synced: <current date>_` at the top of STATUS.md.

### 5. Review

Read through the updated STATUS.md for accuracy. Flag anything that looks wrong.

## Notes

- Do NOT change trust levels during sync — those are set manually after testing.
- Do NOT change status during sync — status changes require actual validation.
- This workflow only handles structural sync (file existence), not content review.
