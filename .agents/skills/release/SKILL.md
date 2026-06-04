---
name: release
description: Cut a new FastVideo release. Bumps the version across the three authoritative files (fastvideo/version.py, pyproject.toml, pyproject_other.toml), opens a [chore]: release PR, and documents the post-merge tag + GitHub release ritual. Triggers on requests like "release X.Y.Z", "cut a release", "bump version to X.Y.Z", "publish to PyPI".
---

# FastVideo release skill

End-to-end recipe for cutting a FastVideo release. The PyPI publish is automatic — pushing a `pyproject.toml` version change to `main` triggers `.github/workflows/publish-fastvideo.yml`. Your job is to land the version bump cleanly and follow up with a git tag + GitHub Release for the changelog.

## Inputs

- `${NEW}` — the new version (e.g. `0.2.0`). Required.
- `${OLD}` — the current version. Auto-detect with: `grep -oP '__version__ = "\K[^"]+' fastvideo/version.py` from the repo root.

## When to use

Trigger phrases: "release X.Y.Z", "cut a release", "bump version to X.Y.Z", "publish to PyPI", "tag a release".

## Files to update (3 — the authoritative list)

These are the ONLY files that carry the version as a Python/package declaration:

| File | Line | Change |
|---|---|---|
| `fastvideo/version.py` | 1 | `__version__ = "${OLD}"` → `__version__ = "${NEW}"` |
| `pyproject.toml` | 7 | `version = "${OLD}"` → `version = "${NEW}"` |
| `pyproject_other.toml` | 7 | `version = "${OLD}"` → `version = "${NEW}"` |

`fastvideo/__init__.py` re-exports `__version__` from `fastvideo.version`, so no edit needed there.

## Files NOT to touch

- `apps/dreamverse/pyproject.toml` — declares `"fastvideo>=X.Y.Z"` as a floor. A new release usually still satisfies the floor; bumping it is a separate policy call (does dreamverse strictly require the new version?). Leave alone unless explicitly asked.
- `.agents/memory/**/*.md` — historical notes; the version strings in there are snapshots, not declarations.
- `examples/`, `docs/` — version mentions are illustrative; not authoritative.
- `uv.lock` — main does NOT track a `uv.lock`. Do NOT run `uv lock` as part of a release.

## Workflow

### 1. Verify clean state

```bash
# From the primary FastVideo jj workspace
jj git fetch
OLD=$(grep -oP '__version__ = "\K[^"]+' fastvideo/version.py)
echo "current: $OLD  →  target: $NEW"
```

Confirm `$NEW > $OLD` follows semver. Check prior tags for the pattern:
```bash
gh release list --repo hao-ai-lab/FastVideo --limit 5
```

### 2. Create a dedicated jj workspace + bookmark

```bash
WS=/home/william5lin/FastVideo_release_${NEW//./_}
jj workspace add --name release-${NEW//./-} "$WS"
cd "$WS"
jj new main@origin -m "[chore]: release v${NEW}"
jj bookmark create chore/release-${NEW} -r @
```

### 3. Apply the 3-file bump

Use the `edit` tool or `sed -i` with exact context. Example with sed:
```bash
sed -i "s/__version__ = \"${OLD}\"/__version__ = \"${NEW}\"/" fastvideo/version.py
sed -i "0,/version = \"${OLD}\"/s//version = \"${NEW}\"/" pyproject.toml
sed -i "0,/version = \"${OLD}\"/s//version = \"${NEW}\"/" pyproject_other.toml
```
(The `0,/.../s//.../` form replaces only the FIRST match in each `pyproject*.toml`, since `${OLD}` might appear elsewhere as a constraint.)

### 4. Verify

```bash
jj diff --name-only -r @          # MUST be exactly 3 files
jj diff --stat -r @                # MUST be +3 / -3
grep -nE "${OLD//./\\.}" fastvideo/version.py pyproject.toml pyproject_other.toml
# expect NO matches in the three files
```

### 5. Lint

```bash
pre-commit run --files fastvideo/version.py pyproject.toml pyproject_other.toml
```
Must pass. Never `--no-verify`.

### 6. Describe + push

```bash
jj describe -m "[chore]: release v${NEW}

Bumps FastVideo from ${OLD} to ${NEW}.

Files updated:
  fastvideo/version.py
  pyproject.toml
  pyproject_other.toml

Note: pushing this to main triggers .github/workflows/publish-fastvideo.yml,
which detects the pyproject.toml version change and publishes to PyPI.
Tag v${NEW} + GitHub release notes follow merge."

jj git push --bookmark chore/release-${NEW}
```

### 7. Open PR

```bash
gh pr create \
  --repo hao-ai-lab/FastVideo \
  --base main \
  --head chore/release-${NEW} \
  --title "[chore]: release v${NEW}" \
  --body-file - <<EOF
## Summary

Bumps FastVideo from \`${OLD}\` to \`${NEW}\`.

## Files updated (3)

- \`fastvideo/version.py\`
- \`pyproject.toml\`
- \`pyproject_other.toml\`

## Out of scope

\`apps/dreamverse/pyproject.toml\` floor (\`fastvideo>=${OLD}\`) — \`${NEW}\` satisfies it; bumping is a separate policy call.

## After merge

\`.github/workflows/publish-fastvideo.yml\` auto-publishes to PyPI on push-to-main when \`pyproject.toml\` changes.

Manual follow-up:
- Tag the merge commit: \`git tag v${NEW} <merge-sha> && git push origin v${NEW}\`
- Create GitHub Release \`v${NEW}\` matching the prior \`Release X.Y.Z\` pattern.
EOF
```

### 8. Post-merge ritual (do AFTER the PR merges)

1. **Tag the merge commit**:
   ```bash
   git fetch origin
   MERGE_SHA=$(gh pr view <PR-NUMBER> --repo hao-ai-lab/FastVideo --json mergeCommit --jq .mergeCommit.oid)
   git tag v${NEW} ${MERGE_SHA}
   git push origin v${NEW}
   ```
2. **Confirm PyPI publish workflow ran**:
   ```bash
   gh run list --repo hao-ai-lab/FastVideo --workflow publish-fastvideo.yml --limit 3
   ```
3. **Create the GitHub Release**:
   ```bash
   gh release create v${NEW} \
     --repo hao-ai-lab/FastVideo \
     --title "Release ${NEW}" \
     --notes "<changelog highlights — what shipped since v${OLD}>" \
     --target main
   ```
   Use `gh release view v${OLD}` to mirror tone/structure from the prior release.
4. **Cleanup**: after merge + tag + release land, tear down the workspace:
   ```bash
   jj workspace forget release-${NEW//./-}
   rm -rf "$WS"
   jj bookmark delete chore/release-${NEW}
   ```

## Verification gates (must all pass before pushing)

- `jj diff --name-only -r @` returns exactly 3 files
- `jj diff --stat -r @` shows `+3 / -3`
- `grep -E "${OLD//./\\.}" fastvideo/version.py pyproject.toml pyproject_other.toml` returns no matches
- `pre-commit run --files <the-three>` passes
- No `uv.lock` in the change
- No source-code files touched

## Conventions (enforced)

- Commit subject: `[chore]: release v${NEW}` (under 72 chars).
- NEVER add AI co-author trailers (`Co-Authored-By: Claude`, "Generated with…", etc.).
- NEVER `--no-verify`.
- NEVER `uv lock` as part of a release — main doesn't track the lockfile.
- Tag format: `vX.Y.Z` (with leading `v`), matching prior releases.

## Why three files?

`pyproject.toml` and `pyproject_other.toml` are two co-existing project metadata files (the project ships both — the latter is a slimmer variant without dreamverse/job-runner extras). Both carry an authoritative `version = "X.Y.Z"` field and must stay in lock-step. `fastvideo/version.py` is the runtime source of truth re-exported by `fastvideo/__init__.py`.

## Publish workflow contract

`.github/workflows/publish-fastvideo.yml` triggers on `push` to `main` when `pyproject.toml` changes. It compares the new `version` field to the previous commit's `version` field and, if different, builds + publishes to PyPI. The version bump in `pyproject_other.toml` does NOT trigger the workflow (only `pyproject.toml` is in the `paths:` filter), but keeping the two in sync prevents installer surprises for users of the alternate metadata file.

## PyPI publish failure modes

The publish workflow ran on the merge commit but the PyPI upload can still fail at the OIDC trusted-publishing exchange. Always verify the workflow succeeded — do not assume "merge implies published":

```bash
gh run list --repo hao-ai-lab/FastVideo --workflow publish-fastvideo.yml --limit 3
```

Look for the run on the release merge commit. If it shows `failure`, dump the failed log:

```bash
gh run view <run-id> --repo hao-ai-lab/FastVideo --log-failed | tail -80
```

### Known failure: `invalid-publisher` (Trusted Publisher claim mismatch)

The most common failure surfaces as:

```
Trusted publishing exchange failure:
* `invalid-publisher`: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
* environment: MISSING
```

This means the PyPI Trusted Publisher registered for the project expects an `environment` claim (e.g. `pypi`) that the workflow job does not set. Two recovery paths:

**A. Fix the trusted publisher + re-run the workflow** (cleaner long-term):
1. On `pypi.org/manage/project/fastvideo/settings/publishing/`, either remove the `Environment name` field from the registered publisher, OR add `environment: pypi` (matching the existing PyPI config) to the `build-publish-main` job in `.github/workflows/publish-fastvideo.yml`.
2. Re-run the failed workflow:
   ```bash
   gh run rerun <run-id> --repo hao-ai-lab/FastVideo --failed
   ```

**B. Manual one-shot publish** (faster, no infra change):

```bash
git checkout <merge-sha>           # the v${NEW} merge commit on main
uv build                            # builds sdist + wheel into dist/
uv publish --token <PYPI_TOKEN>     # or: twine upload dist/*
```

PyPI is **immutable per version** — if any artifact for `${NEW}` got uploaded (sdist or wheel), you cannot re-upload it. Check before retrying:

```bash
curl -s https://pypi.org/pypi/fastvideo/${NEW}/json | python3 -c "import sys,json; d=json.load(sys.stdin); print('on pypi:', list(d['urls'][0].keys()) if d.get('urls') else 'NOT_PUBLISHED')"
```

If `NOT_PUBLISHED`, either recovery path works. If anything is already up, you have to cut a `${NEW}.postN` patch release instead.

### Tag and GitHub Release are independent

The `git tag v${NEW}` and `gh release create v${NEW}` steps are **independent of PyPI publish success**. If you created the tag + release before noticing the publish failure, that's fine — keep them; just complete the PyPI publish via path A or B above. Do NOT delete and re-create the tag, because doing so will cause confusion in dependents that pin to the tag.
