---
name: reseed-ssim-references
description: Re-seed HF reference videos for a single existing SSIM test on Modal L40S. Always backs up current refs locally first, regenerates on Modal, pauses for the user to eyeball before-vs-after quality, then overwrites the targeted `<model_id>` subtree on `FastVideo/ssim-reference-videos` with `--force`. Use when an intentional code change (model port fix, attention backend swap, kernel upgrade, hyperparameter change) has invalidated existing refs and they need to be regenerated. Pairs with `seed-ssim-references`, which is for first-time seeding only.
---

# Re-seed SSIM Reference Videos

## Purpose

Replace the existing SSIM reference videos for a single `(test_file, model_id)`
pair on the HF dataset (`FastVideo/ssim-reference-videos`). This is **destructive**
on HF — the old refs are overwritten — so the skill always:

1. Confirms intent with a one-liner the user has to type.
2. Downloads the existing refs as a local, timestamped backup.
3. Regenerates on Modal L40S (same code path that CI uses).
4. Pauses for a side-by-side eyeball of backup vs new mp4s.
5. Uploads with `--force`, scoped to the single `--model-id`.
6. Reminds the user to keep the backup until the PR lands.

Pairs with `seed-ssim-references`, which is the inverse (first-time seeding
only, refuses to overwrite). Re-seeding is intentionally a separate, more
ceremonial operation because mistakenly clobbering production refs is much
harder to recover from than failing closed.

## When to use

- An intentional code change (model port fix, kernel upgrade, attention
  backend swap, hyperparameter change in the test itself) has shifted the
  expected SSIM output and the existing refs no longer represent the new
  ground truth.
- A test is failing in CI **for the right reason** (the new code is correct,
  the old refs are stale).

## When not to use

- A test is failing for the **wrong** reason (the port is buggy, not the
  refs). Fix the port; re-seeding hides the bug.
- A brand-new test that has no refs on HF yet. Use `seed-ssim-references`.
- "Just to clean up drift" without a concrete code change to point at. The
  PR description has to justify *why* refs changed; without a concrete
  change, there's nothing to write.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `test_file` | Yes | Path to the SSIM test, e.g. `fastvideo/tests/ssim/test_matrixgame_similarity.py`. Validated against `fastvideo/tests/ssim/test_*_similarity.py`. |
| `model_id` | Yes | Single model id from the test's `*_MODEL_TO_PARAMS`, e.g. `Matrix-Game-2.0-Diffusers-Base`. Re-seed runs are **per model**. For multi-model tests, invoke the skill once per model. |
| `intent_rationale` | Yes | One-line explanation of *why* refs are being regenerated (e.g. "Relax FA-2 head_size whitelist to include 80 — matrix_game now uses FLASH_ATTN instead of TORCH_SDPA"). Recorded in the backup directory and reused in the PR description. |

Hardcoded:

- Modal GPU: **L40S** (matches CI; re-seeding from another SKU produces refs
  that L40S CI cannot match).
- Quality tier: **`default`**. `full_quality` is a separate, deliberate
  operation.
- HF repo: `FastVideo/ssim-reference-videos` (override via
  `FASTVIDEO_SSIM_REFERENCE_HF_REPO`).
- Device folder: `L40S_reference_videos`.

## Prerequisites

The user has confirmed:

- `modal` CLI authenticated.
- `hf` CLI authenticated, **and** `HF_API_KEY` (or `HUGGINGFACE_HUB_TOKEN` /
  `HF_TOKEN`) exported with **write** access to
  `FastVideo/ssim-reference-videos`.
- The current branch's code is the change that motivated the re-seed (i.e.
  `git rev-parse HEAD` is the commit that intentionally invalidated refs).

Fail fast if any of these are missing.

## Steps

### 1. Validate inputs and confirm intent

- Verify `test_file` exists and matches `fastvideo/tests/ssim/test_*_similarity.py`.
- Grep the file for `*_MODEL_TO_PARAMS` and assert `model_id` is one of its
  keys. If the file has only a single hardcoded model, accept that model id
  as the only valid value.
- Print the rationale and ask the user to type **`confirm reseed`** (not just
  `y` — make it deliberate):

  > About to RE-SEED references for model `<model_id>` from test `<test_file>`.
  > This will OVERWRITE existing refs on
  > `FastVideo/ssim-reference-videos/reference_videos/default/L40S_reference_videos/<model_id>/`
  > after backup + Modal regen + eyeball.
  >
  > Reason: `<intent_rationale>`
  > HEAD: `<git rev-parse --short=12 HEAD>`
  >
  > Reply `confirm reseed` to proceed, anything else to abort.

  Stop until the user types exactly `confirm reseed`. Anything else aborts
  with no side effects.

### 2. Back up existing refs

Always required. The backup is the only graceful path back if anything goes
wrong later.

```bash
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "<model_id>" | tr '/' '_')
BACKUP_DIR="ssim_reseed_backup/${TIMESTAMP}_${SHORT_COMMIT}_${MODEL_SAFE}"
mkdir -p "$BACKUP_DIR"

hf download \
    --repo-type dataset FastVideo/ssim-reference-videos \
    --include "reference_videos/default/L40S_reference_videos/<model_id>/**" \
    --local-dir "$BACKUP_DIR"

mp4_count=$(find "$BACKUP_DIR" -name "*.mp4" | wc -l)
echo "Backup mp4 count: $mp4_count"
[ "$mp4_count" -gt 0 ] || {
    echo "ERROR: backup is empty for <model_id>. Either the model id is wrong"
    echo "or there are no existing refs (use seed-ssim-references instead)."
    exit 1
}

# Provenance — used in the PR description
cat > "$BACKUP_DIR/PROVENANCE.txt" <<EOF
test_file: <test_file>
model_id: <model_id>
head_commit: $(git rev-parse HEAD)
timestamp_utc: $(date -u +%FT%TZ)
reason: <intent_rationale>
EOF
```

If the `hf download` produces zero mp4s, abort — the user has either picked a
non-existent `model_id` or there are no refs yet (in which case
`seed-ssim-references` is the right tool).

### 3. Regenerate on Modal L40S

Mirror CI's exact env recipe so the regenerated refs are byte-comparable to
what CI will produce on the same commit. Two differences from CI:

1. **Pass the same env prefix CI uses** (`IMAGE_VERSION`, `BUILDKITE_*`) — see
   `.buildkite/pipeline.yml:1-3` and `.buildkite/scripts/pr_test.sh:62-83`.
   Without this, `ssim_test.py:17-18` resolves a different GHCR image tag
   (default is `latest`, CI is `py3.12-latest`), and `ssim_test.py:38-46`
   bakes different values into the image's frozen env block. **Mismatched
   image or env is the most common source of SSIM drift between reseed and
   CI runs.**
2. **Do not pass `--skip-reference-download`**. Letting the test fetch the
   existing refs and run the full SSIM compare gives "before" SSIM numbers
   for the PR description, and the test still produces the new mp4s
   regardless of whether the comparison passes or fails.

```bash
SUBDIR="${TIMESTAMP}_${SHORT_COMMIT}"

IMAGE_VERSION="py3.12-latest" \
BUILDKITE_REPO="$(git config --get remote.origin.url)" \
BUILDKITE_COMMIT="$(git rev-parse HEAD)" \
BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST:-false}" \
modal run fastvideo/tests/modal/ssim_test.py \
    --git-repo="$(git config --get remote.origin.url)" \
    --git-commit="$(git rev-parse HEAD)" \
    --hf-api-key="$HF_API_KEY" \
    --test-files="<test_file>" \
    --sync-generated-to-volume \
    --generated-volume-subdir="$SUBDIR" \
    --no-fail-fast
```

Capture the printed `modal volume get ...` hint — its `<SUBDIR>` matches
`$SUBDIR` and is needed for step 4. Capture the SSIM numbers from the test
output (or from the JSON next to the generated mp4) for the PR description.

### 4. Download generated videos

```bash
modal volume get --force hf-model-weights \
    ssim_generated_videos/default/"$SUBDIR"/generated_videos \
    ./generated_videos_modal/default
```

After this, the new mp4s live at:

```
./generated_videos_modal/default/generated_videos/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4
```

`--force` is required when `./generated_videos_modal/default` already exists
from a prior run; safe on the first run too.

### 5. PAUSE — user reviews quality side-by-side

Print the diff and the comparison:

```bash
echo "=== File list diff (backup vs new) ==="
diff -u \
    <(find "$BACKUP_DIR/reference_videos/default/L40S_reference_videos/<model_id>" -name "*.mp4" \
        | sed "s|$BACKUP_DIR/reference_videos/default/L40S_reference_videos/||" | sort) \
    <(find ./generated_videos_modal/default/generated_videos/L40S_reference_videos/<model_id> -name "*.mp4" \
        | sed "s|./generated_videos_modal/default/generated_videos/L40S_reference_videos/||" | sort) \
    || true

echo
echo "=== SSIM numbers from this run (paste into PR) ==="
find ./generated_videos_modal/default/generated_videos/L40S_reference_videos/<model_id> -name "*_ssim.json" -exec cat {} \;
```

Then stop and tell the user:

> Old refs backed up to `$BACKUP_DIR`.
> New videos in `./generated_videos_modal/default/generated_videos/L40S_reference_videos/<model_id>/`.
>
> Open both in a video player. Confirm the new videos:
>   1. Look correct (no obvious artifacts, no black/static frames).
>   2. Are *intentionally* different from the backup in the way described
>      in `<intent_rationale>` (e.g. slight numerical drift only, not a
>      different scene / different motion / corrupted output).
>
> Reply **`upload`** to overwrite HF, anything else to abort.
> Aborting leaves the backup and new videos on disk for inspection — nothing
> on HF changes.

Do not proceed until the user types exactly `upload`. If they abort, leave
everything on disk and stop here.

### 6. Copy into the local reference layout

Same as `seed-ssim-references` step 5:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
    --quality-tier default \
    --device-folder L40S_reference_videos \
    --generated-dir ./generated_videos_modal/default/generated_videos/L40S_reference_videos
```

Result: `fastvideo/tests/ssim/reference_videos/default/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4`.

### 7. Upload with `--force`, scoped to `--model-id`

The `--force` flag is what makes this skill different from `seed-ssim-references`.
Always pair it with `--model-id` so a typo cannot accidentally overwrite a
neighboring model's refs.

```bash
python fastvideo/tests/ssim/reference_videos_cli.py upload \
    --quality-tier default \
    --device-folder L40S_reference_videos \
    --model-id "<model_id>" \
    --force
```

The CLI's overwrite guard refuses without `--force`; with `--force` it
overwrites only files under
`reference_videos/default/L40S_reference_videos/<model_id>/`.

### 8. Report success and retention guidance

Print:

- The HF path that was overwritten (`<repo>/reference_videos/default/L40S_reference_videos/<model_id>/`).
- The local backup directory path.
- The new SSIM numbers from step 5.
- This restore command, in case the PR review surfaces a problem after
  upload:

  ```bash
  python fastvideo/tests/ssim/reference_videos_cli.py upload \
      --quality-tier default \
      --device-folder L40S_reference_videos \
      --model-id "<model_id>" \
      --reference-dir "$BACKUP_DIR/reference_videos/default/L40S_reference_videos" \
      --force
  ```

- This PR-description checklist (see `fastvideo/tests/ssim/AGENTS.md` →
  *Updating Reference Videos*):
  1. Source commit that produced the new refs (HEAD at re-seed time).
  2. Test command and GPU SKU (`L40S`).
  3. Before/after SSIM numbers.
  4. The `<intent_rationale>` from step 1.
  5. A note that the backup lives at `$BACKUP_DIR` and should be retained
     until CI on the PR is green.

Do **not** auto-rerun the SSIM test — the user does that as part of the PR.

## Failure modes and how to handle them

- **`HF_API_KEY` unset.** Stop before step 2.
- **Backup is empty (zero mp4s).** Stop before step 3 — the model id is
  wrong or the refs don't exist yet (use `seed-ssim-references`).
- **Modal run fails before generation.** No mp4s on the volume. Don't
  upload. Investigate the failure (test crash, OOM, partition exhaustion),
  fix, then retry from step 3. Backup is still intact.
- **Quality regressed (visual or metric).** User aborts at step 5. Backup
  retained. New videos retained on disk for inspection. Nothing on HF
  changed. Either fix the underlying code change or abandon the re-seed.
- **User confirmed `upload` but later realized the new refs are wrong.**
  Run the restore command from step 8 with the backup `--reference-dir`.
  This is exactly why the backup exists.
- **Multi-model test, only one model is being re-seeded.** Run the skill
  once per model id. The `--model-id` scope on upload guarantees the others
  are untouched.

## Design notes (for future skill maintainers)

- Per-`model_id` scope is mandatory. The dataset houses many model subtrees;
  re-seeding the wrong one is hard to undo without backup.
- `default` tier only; `full_quality` is a separate, deliberate operation
  with different params and ~doubled runtime, and isn't what CI gates on.
- The skill deliberately does **not** pass `--skip-reference-download` to
  Modal so we get pre-reseed SSIM numbers for the PR. The `seed`-skill
  passes it because no refs exist yet; for re-seed, refs do exist and
  exposing the comparison is informative.
- The two-token confirm (`confirm reseed`, then `upload`) is intentional.
  Re-seeding is high-blast-radius and should not be one-keystroke.
- The backup directory is plain mp4s + `PROVENANCE.txt`. No HF metadata is
  preserved; the restore path uses `reference_videos_cli.py upload
  --reference-dir` which doesn't need it.

## References

- `.agents/skills/seed-ssim-references/SKILL.md` — the first-time seed
  skill this one parallels. Read it for the Modal flag rationale shared
  between the two flows.
- `fastvideo/tests/ssim/AGENTS.md` — directory rules, including the PR
  expectations for any reference-video change (rationale, before/after
  SSIM, source commit/model/backend).
- `fastvideo/tests/ssim/reference_videos_cli.py` — `copy-local`, `upload`
  (with `--model-id`, `--force`), `download`. The overwrite guard at
  `upload_reference_videos` is the safety net this skill leans on.
- `fastvideo/tests/modal/ssim_test.py` — Modal orchestrator;
  `--sync-generated-to-volume`, `--generated-volume-subdir`,
  `--skip-reference-download`, `--no-fail-fast`.

## Changelog

| Date | Change |
|------|--------|
| 2026-05-02 | Initial version. Sister skill to `seed-ssim-references`, scoped to single `(test_file, model_id)` re-seeds, with mandatory backup and two-token confirm. |
