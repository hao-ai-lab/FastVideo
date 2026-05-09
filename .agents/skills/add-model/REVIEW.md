# add-model skill review backlog (historical)

Updated 2026-04-30 after the phase-based `/add-model` rewrite.

All skill-text items from the prior review were incorporated into the current
split skill stack under `.agents/skills/add-model*`. This file is kept
only for codebase-owner follow-ups that are not blockers for the skill workflow.

### 1. Audit `wan_to_diffusers.py` usage

`SKILL.md` now treats `scripts/checkpoint_conversion/wan_to_diffusers.py` as a
legacy regex-reference file, not a conversion-script template.

Open codebase question: is this module still imported by live code? If yes,
document the caller near the script or in developer docs. If no, delete it in a
separate cleanup PR.

### 2. Decide Whether To Add Audio Workload Enums

The current pipeline skill documents the repository's compatibility workaround:
until `WorkloadType` grows audio values, audio-only pipelines may register as
`T2V` with explicit rationale and minimal video-shaped placeholders when shared
`VideoGenerator` paths require them.

Open codebase question: should `WorkloadType` be extended now with audio and
joint AV variants, or should the first audio pipeline PR own that enum change?
