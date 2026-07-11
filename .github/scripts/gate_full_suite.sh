#!/usr/bin/env bash
# Gate the expensive Buildkite full suite on the cheap GitHub checks.
#
# Polls the workflow runs for the PR head commit and only exits 0 once the
# watched cheap workflows (pre-commit, docs build) have succeeded, so the
# 'ready' label cannot burn ~20 GPU lanes on a head that a cheap check has
# already doomed.
#
# Semantics:
#   - watched run completed with a bad conclusion  -> exit 1 (fail CLOSED:
#     no full suite; the next push re-arms via the 'synchronize' trigger)
#   - watched run cancelled                         -> still pending: the docs
#     workflow's repo-global 'pages' concurrency group cancels runs superseded
#     by unrelated pushes, so 'cancelled' is not a verdict on this PR
#   - watched runs pending                          -> poll until done
#   - docs run absent                               -> not applicable after a
#     short grace period ('Deploy Documentation' is path-filtered on PRs)
#   - pre-commit run absent                         -> keep polling: pre-commit
#     is never path-filtered, so its absence is always anomalous
#   - 'ready' label removed while waiting           -> exit 1 (fail CLOSED:
#     un-labeling is a deliberate maintainer action)
#   - GitHub API unreachable or timeout             -> exit 0 (fail OPEN,
#     loud warning: never brick CI on a GitHub outage)
#
# Required env: PR_SHA (PR head commit), PR_NUMBER, GITHUB_REPOSITORY, GH_TOKEN.
set -euo pipefail

: "${PR_SHA:?PR_SHA (PR head commit) is required}"
: "${PR_NUMBER:?PR_NUMBER (pull request number) is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

# Workflow-level `name:` values that must be green before the full suite
# may start. "Deploy Documentation" is path-filtered on PRs, so its run may
# legitimately never exist; pre-commit always runs, so it must appear.
WATCHED_NAMES='["pre-commit", "Deploy Documentation"]'
WATCHED_REGEX='^(pre-commit|Deploy Documentation)$'
POLL_SECS="${POLL_SECS:-20}"
GRACE_SECS="${GRACE_SECS:-60}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1500}"

# Bound each API call so a hung connection hits the 3-strike fail-open path
# instead of pinning the loop until the job timeout (which would fail closed
# on exactly the GitHub-outage case this script is meant to survive).
if command -v timeout >/dev/null 2>&1; then
  gh_api() { timeout 30 gh api "$@"; }
else
  gh_api() { gh api "$@"; }  # macOS dev boxes; CI always has coreutils timeout
fi

# The workflow checked the label before starting the gate, but the wait can
# last ~25 min: re-check once before any exit 0 and fail closed if 'ready'
# was removed in the meantime. An API error here proceeds (the label was
# present when the gate started; never brick CI on an outage).
recheck_ready_label() {
  local pr_json
  if pr_json=$(gh_api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" 2>/dev/null); then
    if ! jq -e '[.labels[]?.name] | index("ready")' <<<"$pr_json" >/dev/null 2>&1; then
      echo "::error::PR #${PR_NUMBER} no longer has the 'ready' label —" \
        "NOT triggering the Buildkite full suite. Re-add the label to re-arm."
      exit 1
    fi
  else
    echo "::warning::Could not re-check the 'ready' label on PR #${PR_NUMBER}; proceeding (it was present when the gate started)."
  fi
}

start=$(date +%s)
api_fails=0
missing=""

while true; do
  elapsed=$(( $(date +%s) - start ))

  if runs_json=$(gh_api "repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${PR_SHA}&per_page=100" 2>/dev/null) \
     && state=$(jq --arg re "$WATCHED_REGEX" '
          [.workflow_runs[]? | select(.name // "" | test($re))]
          | group_by(.name) | map(max_by(.id))
          | map({name, status, conclusion})' <<<"$runs_json" 2>/dev/null); then
    api_fails=0
    echo "t+${elapsed}s watched checks: $(jq -c . <<<"$state")"

    failed=$(jq -r '[.[] | select(.status == "completed"
          and (.conclusion | IN("success", "skipped", "neutral", "cancelled") | not))]
        | map(.name) | join(", ")' <<<"$state")
    if [ -n "$failed" ]; then
      echo "::error::Cheap check(s) failed on ${PR_SHA}: ${failed}." \
        "NOT triggering the Buildkite full suite. Push a fix (the 'ready'" \
        "label re-arms on every push), or re-run the failed check and then" \
        "re-run this workflow."
      exit 1
    fi

    # 'cancelled' counts as pending: wait for a re-run to reach a real verdict
    # (bounded by MAX_WAIT, then the fail-open below).
    pending=$(jq '[.[] | select(.status != "completed" or .conclusion == "cancelled")] | length' <<<"$state")
    missing=$(jq -r --argjson watched "$WATCHED_NAMES" '($watched - map(.name)) | join(", ")' <<<"$state")
    if [ "$pending" -eq 0 ]; then
      if [ -z "$missing" ]; then
        recheck_ready_label
        echo "All watched cheap checks are green — full suite may proceed."
        exit 0
      fi
      case "$missing" in
        *pre-commit*)
          echo "pre-commit run not found for ${PR_SHA} yet; waiting (pre-commit is never path-filtered, so its absence is anomalous)."
          ;;
        *)
          if [ "$elapsed" -ge "$GRACE_SECS" ]; then
            recheck_ready_label
            echo "::warning::Watched run(s) never appeared for ${PR_SHA}: ${missing} (path-filtered, likely not applicable). Proceeding on the checks that did run."
            exit 0
          fi
          echo "Waiting up to ${GRACE_SECS}s grace for path-filtered run(s) to appear: ${missing}."
          ;;
      esac
    fi
  else
    api_fails=$(( api_fails + 1 ))
    echo "::warning::GitHub API error querying workflow runs for ${PR_SHA} (attempt ${api_fails}/3)."
    if [ "$api_fails" -ge 3 ]; then
      recheck_ready_label
      echo "::warning::FAILING OPEN: cannot query GitHub check status — triggering the full suite WITHOUT the cheap-check gate."
      exit 0
    fi
  fi

  if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    recheck_ready_label
    echo "::warning::FAILING OPEN: watched checks still pending after $(( MAX_WAIT_SECS / 60 )) min${missing:+ (never appeared: ${missing})} — triggering the full suite anyway."
    exit 0
  fi
  sleep "$POLL_SECS"
done
