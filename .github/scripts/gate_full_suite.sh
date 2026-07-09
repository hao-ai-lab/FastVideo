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
#   - watched runs pending                          -> poll until done
#   - watched run absent (path-filtered workflow,   -> treated as not
#     e.g. docs untouched)                             applicable after a
#                                                      short grace period
#   - GitHub API unreachable or timeout             -> exit 0 (fail OPEN,
#     loud warning: never brick CI on a GitHub outage)
#
# Required env: PR_SHA (PR head commit), GITHUB_REPOSITORY, GH_TOKEN.
set -euo pipefail

: "${PR_SHA:?PR_SHA (PR head commit) is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

# Workflow-level `name:` values that must be green before the full suite
# may start. "Deploy Documentation" is path-filtered on PRs, so its run may
# legitimately never exist.
WATCHED_REGEX='^(pre-commit|Deploy Documentation)$'
WATCHED_COUNT=2
POLL_SECS="${POLL_SECS:-20}"
GRACE_SECS="${GRACE_SECS:-60}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1500}"

start=$(date +%s)
api_fails=0

while true; do
  elapsed=$(( $(date +%s) - start ))

  if runs_json=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${PR_SHA}&per_page=100" 2>/dev/null) \
     && state=$(jq --arg re "$WATCHED_REGEX" '
          [.workflow_runs[] | select(.name | test($re))]
          | group_by(.name) | map(max_by(.id))
          | map({name, status, conclusion})' <<<"$runs_json" 2>/dev/null); then
    api_fails=0
    echo "t+${elapsed}s watched checks: $(jq -c . <<<"$state")"

    failed=$(jq -r '[.[] | select(.status == "completed"
          and (.conclusion | IN("success", "skipped", "neutral") | not))]
        | map(.name) | join(", ")' <<<"$state")
    if [ -n "$failed" ]; then
      echo "::error::Cheap check(s) failed on ${PR_SHA}: ${failed}." \
        "NOT triggering the Buildkite full suite. Push a fix (the 'ready'" \
        "label re-arms on every push), or re-run the failed check and then" \
        "re-run this workflow."
      exit 1
    fi

    pending=$(jq '[.[] | select(.status != "completed")] | length' <<<"$state")
    found=$(jq 'length' <<<"$state")
    if [ "$pending" -eq 0 ]; then
      if [ "$found" -eq "$WATCHED_COUNT" ] || [ "$elapsed" -ge "$GRACE_SECS" ]; then
        echo "All applicable cheap checks are green (${found}/${WATCHED_COUNT} ran) — full suite may proceed."
        exit 0
      fi
      echo "Only ${found}/${WATCHED_COUNT} watched runs exist; allowing ${GRACE_SECS}s grace for the rest to appear (path-filtered workflows may never run)."
    fi
  else
    api_fails=$(( api_fails + 1 ))
    echo "::warning::GitHub API error querying workflow runs for ${PR_SHA} (attempt ${api_fails}/3)."
    if [ "$api_fails" -ge 3 ]; then
      echo "::warning::FAILING OPEN: cannot query GitHub check status — triggering the full suite WITHOUT the cheap-check gate."
      exit 0
    fi
  fi

  if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    echo "::warning::FAILING OPEN: watched checks still pending after $(( MAX_WAIT_SECS / 60 )) min — triggering the full suite anyway."
    exit 0
  fi
  sleep "$POLL_SECS"
done
