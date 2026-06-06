# Review output format

All reviewers must produce output in this format so the dispatcher can
compose results across reviewers without re-parsing prose.

## Severity levels

| Level | Meaning | Examples |
|-------|---------|----------|
| **BLOCKER** | PR should not merge as-is. Correctness, security, data loss, or a regression with no path forward. | Wrong FP reduction dtype; missing `param_names_mapping` entry breaks loading; kernel UB. |
| **MAJOR** | Should be addressed before merge. Design issue, missing tests for a claim, API break. | No SSIM test for a new model; perf claim without benchmark; public API changed without migration note. |
| **MINOR** | Nice to fix, not required. Readability, consolidation, small redundancy. | Duplicate helper; obscure variable name; minor config drift. |
| **NIT** | Purely stylistic or preference. | "Prefer `x` over `y`"; "consider renaming". |

Reviewers **must not** use BLOCKER for style — pre-commit handles style. BLOCKER
is for things that would land broken code.

### Pre-merge vs post-merge semantics

- **Open PR**: BLOCKER means "do not merge as-is." Verdict `REQUEST_CHANGES`.
- **Merged PR** (calibration / audit run): BLOCKER means "follow-up required
  to fix what shouldn't have landed." Verdict `COMMENT` unless a revert is
  appropriate.

State up front whether the PR is open or merged in the report summary so the
reader calibrates the severity correctly.

### BLOCKER justification rule

If a reviewer emits a BLOCKER, the bullet must explain **why it's a BLOCKER
and not a MAJOR**. BLOCKER should be rare — if you can't articulate the
difference, use MAJOR.

## Output template

```markdown
# [<reviewer-name>] PR #<NNNN>: <title>

## Summary
<2–4 sentences: what the PR does, whether the reviewer recommends merge,
headline concerns.>

## Verdict
**<APPROVE | REQUEST_CHANGES | COMMENT>** <one-line reason>

## Blocking issues
<Each item as: **BLOCKER** `path/to/file.py:LN` — <issue> <fix suggestion>>
<If none, write: "None.">

## Major concerns
<Same format, severity MAJOR. If none, write "None.">

## Minor suggestions
<Same format, severity MINOR. If none, write "None.">

## Nits
<Same format, severity NIT. If none, write "None.">

## Checklist
<A condensed version of this reviewer's checklist.md, with ✅ / ❌ / ⚠️ / N/A per item.>

## Test plan review
<Did the author provide commands + output in the PR body? Is the coverage
appropriate for the claim? For model PRs: is there an SSIM test? For kernel
PRs: is there a correctness + benchmark test? For training PRs: is there a
parity run or a short training curve?>

## Out of scope (not reviewed)
<Paths this reviewer intentionally skipped, so the user knows another reviewer
needs to cover them.>
```

## Rules

1. **Cite every comment with `path/to/file.py:LN`.** Users should be able to
   jump to the code without grep.
2. **One issue per bullet.** Don't fold two unrelated issues into one line.
3. **Prefer direct quotes for claims.** If the PR body says "2x speedup",
   quote it before asking for a benchmark.
4. **No ceremony.** No "Thanks for the PR!" / "Great work on..." / emoji-laden
   preambles. The user reads dozens of these per week.
5. **If you're unsure, say so.** "I'm not sure this handles SP group 0 —
   please confirm" is better than flagging a false BLOCKER.
6. **Keep the final report focused.** If a reviewer has nothing substantive
   to add, output `## Verdict\n**APPROVE** — no blockers within <scope>.` and
   stop.
7. **Length discipline.** Reviewers with fewer than 3 findings should emit
   fewer than 40 lines of output total. Reports with only NITs should be
   consolidated into a single paragraph. Long reports for small diffs are
   noise.
8. **Read the PR body before flagging process issues.** CI-red status, "why
   did this merge anyway", and "where's the test evidence" are often
   answered in the PR body. A reviewer that flags these without having read
   the body loses credibility on the real findings in the same report.
