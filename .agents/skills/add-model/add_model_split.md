# add-model skill — split plan

The current `SKILL.md` is **1605 lines** in a single file. This document
proposes splitting it into ~12 satellite docs with a much shorter
top-level index, following the idiomatic Anthropic-skills pattern of
"short procedural index + targeted satellites."

## Why split

1. **Cold-load cost.** The whole 1605-line file is loaded into the
   model's context every time the skill fires. Most porting sessions
   only need a fraction of that — e.g. a VAE-only contribution doesn't
   need the I2V-variant section or the weight-conversion recipe. A
   shorter `SKILL.md` index + lazy-loaded satellites keeps the live
   context lean.

2. **Reviewability.** A single 1605-line markdown file is hard to
   diff. Splits keep changes scoped — adding the "audio workload"
   section (REVIEW item 25) becomes a new file in `how-to/` rather
   than a 100-line insert in the middle of an existing megafile.

3. **Findability.** Section anchors in a single file are hard to
   navigate; per-topic files surface in `ls .agents/skills/add-model/`
   and `grep -r` returns clean per-file matches.

4. **Component-only contributions** (REVIEW item 23) become a
   first-class workflow document instead of an awkward "but skip
   half the steps" exception inside the main file.

## Current section inventory

| Section | Approx lines | Stays in SKILL.md? | Target file |
|---|---|---|---|
| Purpose | 8 | Yes | SKILL.md |
| When to use / not to use | 25 | Yes | SKILL.md |
| Prerequisites | 40 | Yes | SKILL.md |
| Inputs (table) | 15 | Yes | SKILL.md |
| FastVideo's single architecture | 40 | No | how-to/architecture.md |
| Files you will create or touch (table) | 25 | Yes (link to how-to) | SKILL.md + how-to/files_table.md |
| Steps (1–16) | 440 | **Index only** in SKILL.md (3-line per step + link) | how-to/steps_<phase>.md ×4 |
| Standard stages — subclass targets | 20 | No | how-to/architecture.md |
| FastVideo layers and attention | 110 | No | how-to/layers_and_attention.md |
| Parity test pattern | 165 | No | how-to/parity_testing.md |
| Parallel component porting | 100 | No | how-to/parity_testing.md |
| Weight conversion | 155 | No | how-to/weight_conversion.md |
| `register_configs` cheatsheet | 25 | No | how-to/registry_cheatsheet.md |
| Adding an I2V variant | 270 | No | how-to/i2v_variant.md |
| Distributed support | 35 | No | how-to/distributed_support.md |
| Common pitfalls | 50 | No | reference/pitfalls.md |
| Outputs | 15 | Yes | SKILL.md |
| Example prompt snippet | 35 | No | reference/example_prompt.md |
| References | 50 | No | reference/links.md |
| Changelog | 20 | No | reference/changelog.md |

## Target tree

```
.agents/skills/add-model/
├── SKILL.md                              ~250 lines  — procedural index
├── REVIEW.md                             unchanged
├── add_model_split.md                    this doc
├── how-to/
│   ├── architecture.md                   ~70 lines   — single-architecture diagram + standard stages
│   ├── files_table.md                    ~50 lines   — annotated 17-row Files table
│   ├── steps_1_setup.md                  ~80 lines   — steps 1–5 (gather, study, reuse, convert, clone)
│   ├── steps_2_components.md             ~120 lines  — step 6 + parallel component porting
│   ├── steps_3_pipeline.md               ~100 lines  — steps 7–11 (config, stages, pipeline class, presets, registry)
│   ├── steps_4_validation.md             ~120 lines  — steps 12–16 (smoke, pipeline parity, SSIM, cleanup, ask)
│   ├── parity_testing.md                 ~250 lines  — conventions + component template + pipeline-level gate + subagent prompt
│   ├── weight_conversion.md              ~200 lines  — decision tree + recipe + reference scripts + gotchas
│   ├── layers_and_attention.md           ~150 lines  — linear/attention/primitive selection rules
│   ├── i2v_variant.md                    ~270 lines  — full I2V add section verbatim (cleanly extractable)
│   ├── distributed_support.md            ~50 lines   — SP/TP/VAE-tiling rules
│   ├── registry_cheatsheet.md            ~30 lines   — register_configs fields
│   ├── component_only_contributions.md   ~120 lines  — NEW (REVIEW #23): VAE-only / encoder-only PR shape
│   └── audio_workload.md                 ~150 lines  — NEW (REVIEW #25): audio output, T2A workload, no-SSIM metrics
├── reference/
│   ├── pitfalls.md                       ~80 lines   — the 14-item pitfalls list
│   ├── example_prompt.md                 ~40 lines   — the example user prompt snippet
│   ├── links.md                          ~50 lines   — file/repo references
│   └── changelog.md                      ~30 lines   — change history table
└── seed-ssim-references/                 unchanged
```

Total: ~2200 lines across 17 files (vs current 1605 lines × 1 file).
The line growth is intentional — each file gets a short "Purpose +
Status + Prerequisites" header so it can be loaded without context
from the others.

## SKILL.md target shape (~250 lines)

```markdown
# Add a Model to FastVideo

## Purpose
[8 lines, unchanged]

## When to use / When not to use
[25 lines, unchanged]

## Prerequisites — gather inputs (blocking)
[40 lines, unchanged]

## Inputs
[15-line table, unchanged]

## Files you will create or touch
The full table lives in `how-to/files_table.md`. Quick sketch:
- Component files (model + config + __init__ export): rows 1–6.
- Pipeline files (config + class + stages + presets): rows 7–10.
- Registry: row 11.
- Tests (smoke + pipeline parity + per-component parity + SSIM): rows 12–15.
- Conversion script (only if not Diffusers format): row 16.
- Example: row 17.

For component-only contributions (just a VAE / encoder), see
`how-to/component_only_contributions.md` — you can skip rows 7–13 + 15
+ 17.

## Steps (procedural index)

1. **Gather inputs (blocking).** See `how-to/steps_1_setup.md`.
2. **Study the reference implementation.** See `how-to/steps_1_setup.md`.
3. **Decide what to reuse.** See `how-to/steps_1_setup.md`.
4. **Convert weights to Diffusers format.** Only if not Diffusers-format.
   See `how-to/weight_conversion.md`.
5. **Clone the official repo for parity testing.** See `how-to/steps_1_setup.md`.
6. **Port components in parallel via subagents.** See
   `how-to/steps_2_components.md` + `how-to/parity_testing.md`.
7. **Create the PipelineConfig.** See `how-to/steps_3_pipeline.md`.
8. **Build or pick the stages.** See `how-to/architecture.md` (standard
   stages catalog) + `how-to/steps_3_pipeline.md`.
9. **Write the pipeline class.** See `how-to/steps_3_pipeline.md`.
10. **Define presets.** See `how-to/steps_3_pipeline.md`.
11. **Register in `fastvideo/registry.py`.** See `how-to/registry_cheatsheet.md`.
12. **Smoke-test the pipeline.** See `how-to/steps_4_validation.md`.
13. **Full-pipeline parity + example (gated).** See
    `how-to/parity_testing.md` (the pipeline-level section is the
    handoff gate).
14. **Add SSIM regression.** See `how-to/steps_4_validation.md`. For
    audio, see `how-to/audio_workload.md` (no SSIM analog).
15. **Clean up the cloned reference repo.** See `how-to/steps_4_validation.md`.
16. **Ask about tests + perf data.** See `how-to/steps_4_validation.md`.

## Pre-handoff checklist
[New section addressing REVIEW item 16 — bans skip-only parity at handoff]
- [ ] `pytest tests/local_tests/<bucket>/test_<family>_*parity*.py -v` produces non-skip PASS for each non-reused component.
- [ ] `pytest tests/local_tests/pipelines/test_<family>_pipeline_parity.py -v` produces non-skip PASS.
- [ ] `python examples/inference/basic/basic_<family>.py` writes a non-corrupt mp4 (or .wav for audio).
- [ ] Conversion has actually been run (the parity tests skip if not).

## Outputs
[15 lines, unchanged]

## Common pitfalls
See `reference/pitfalls.md` for the full 14-item list. Most-cited:
- #1: `EntryClass` missing → pipeline silently invisible.
- #11: raw `nn.Linear` in DiT/VAE hot paths → use `ReplicatedLinear`.
- #16: skip-only parity → see pre-handoff checklist above.

## See also
- `reference/example_prompt.md` — example user prompt for invoking this skill.
- `reference/links.md` — file/repo references.
- `reference/changelog.md` — change history.
```

## Migration steps

1. **Create the new files** with content extracted from current
   `SKILL.md` (no semantic edits — just relocation + cross-link edits).
2. **Add per-file headers** of the form:

   ```markdown
   # <topic>

   **Part of:** add-model skill
   **When to read:** <one-liner — e.g. "during step 6 / parallel
   component porting">
   **Prerequisites:** <links to other files needed first>

   ---

   <content>
   ```

3. **Rewrite SKILL.md** as the index. Each step gets the new 3-line
   format: title + 2-line summary + link to `how-to/<file>.md`.
4. **Add the pre-handoff checklist** (addresses REVIEW item 16) — the
   single load-bearing addition this split enables.
5. **Add `how-to/component_only_contributions.md`** (addresses REVIEW
   item 23).
6. **Add `how-to/audio_workload.md`** (addresses REVIEW items 25 + 28).
7. **Update REVIEW.md** to mark items 16, 23, 25, 28 as resolved by
   the split.
8. **Re-run a sample skill invocation** (e.g. on a hypothetical new
   port) to validate the split — check that the model only loads
   the index + 2-3 satellite files for a typical port.

## Risks / open questions

- **Breaks existing prompts.** Anyone with an in-flight skill
  invocation may have memorized section anchors that move. Mitigate
  by leaving anchor stubs in SKILL.md for one cycle (forwarding
  comments).
- **Cross-link maintenance.** Each rename / move requires updating
  cross-refs. Mitigate by keeping the file tree shallow (one
  `how-to/` and one `reference/` directory only).
- **Discoverability of new files.** A porter who only reads
  `SKILL.md` may not realize `audio_workload.md` exists. Mitigate by
  having the index's step-3 line for "audio variant of step X" link
  explicitly to the audio doc, and by keeping the "See also" section
  visible.
- **What counts as "idiomatic"?** Anthropic skills tend toward
  ~150-300 line single-file or ~3-5 file splits. 17 files is on the
  large side. We might collapse further if some satellites are
  always-loaded-together (e.g. merge `architecture.md` +
  `registry_cheatsheet.md` if they're never read independently).
  Decide post-prototype.

## Next step

Implement the split as a separate PR (don't bundle with the
will/stable-audio first-class VAE work or the will/magi MagiHuman
port). Sequence:

1. Land the split as-is (mechanical relocation, zero semantic
   change). Verify that running the skill against a known port
   produces equivalent guidance.
2. Then layer in the REVIEW-item edits (16, 23, 25, 28) as content
   changes inside the new structure.
