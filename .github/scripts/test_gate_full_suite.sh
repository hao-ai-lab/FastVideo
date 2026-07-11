#!/usr/bin/env bash
# Self-test for gate_full_suite.sh using a mocked `gh`. No network, runs on
# any dev box: bash .github/scripts/test_gate_full_suite.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Mock gh. Asserts the exact endpoint (including head_sha) it is called
# with — an endpoint typo in the gate script fails the test rather than
# silently serving canned data. On the runs endpoint it serves
# $MOCK_DIR/response_<call#>.json, sticking on the highest existing file,
# and exits 1 if none exist (simulates a GitHub API outage). On the pulls
# endpoint it serves $MOCK_DIR/pr.json, defaulting to a 'ready'-labeled PR.
cat > "$tmp/gh" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" != "api" ]; then
  echo "unexpected gh invocation: $*" >> "$MOCK_DIR/endpoint_error"
  exit 2
fi
case "${2:-}" in
  "repos/o/r/actions/runs?head_sha=deadbeef&per_page=100")
    n=$(( $(cat "$MOCK_DIR/count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$MOCK_DIR/count"
    while [ "$n" -gt 0 ]; do
      if [ -f "$MOCK_DIR/response_$n.json" ]; then
        cat "$MOCK_DIR/response_$n.json"
        exit 0
      fi
      n=$(( n - 1 ))
    done
    echo "api outage" >&2
    exit 1
    ;;
  "repos/o/r/pulls/42")
    if [ -f "$MOCK_DIR/pr.json" ]; then
      cat "$MOCK_DIR/pr.json"
    else
      echo '{"labels": [{"name": "ready"}]}'
    fi
    ;;
  *)
    echo "unexpected gh endpoint: $2" >> "$MOCK_DIR/endpoint_error"
    exit 2
    ;;
esac
EOF
chmod +x "$tmp/gh"

PC_OK='{"name": "pre-commit", "id": 1, "status": "completed", "conclusion": "success"}'
PC_BAD='{"name": "pre-commit", "id": 1, "status": "completed", "conclusion": "failure"}'
PC_PENDING='{"name": "pre-commit", "id": 1, "status": "in_progress", "conclusion": null}'
DOCS_OK='{"name": "Deploy Documentation", "id": 2, "status": "completed", "conclusion": "success"}'
DOCS_BAD='{"name": "Deploy Documentation", "id": 2, "status": "completed", "conclusion": "failure"}'
DOCS_CANCELLED='{"name": "Deploy Documentation", "id": 2, "status": "completed", "conclusion": "cancelled"}'
OTHER='{"name": "Trigger Full Suite", "id": 3, "status": "in_progress", "conclusion": null}'
NULL_NAME='{"name": null, "id": 4, "status": "completed", "conclusion": "failure"}'
PC_OK_RERUN='{"name": "pre-commit", "id": 5, "status": "completed", "conclusion": "success"}'

fails=0
want_log=""  # optional: expect() also greps out.log for this regex, then resets
pr_json=""   # optional: served for the pulls (label re-check) endpoint, then resets
raw_body=""  # optional: serve responses verbatim instead of wrapping in workflow_runs
expect() { # <name> <expected-exit> <response json>...
  local name=$1 want=$2 dir i=1
  shift 2
  dir=$(mktemp -d "$tmp/test_XXXXXX")
  for body in "$@"; do
    if [ -n "$raw_body" ]; then
      printf '%s' "$body" > "$dir/response_$i.json"
    else
      printf '{"workflow_runs": [%s]}' "$body" > "$dir/response_$i.json"
    fi
    i=$(( i + 1 ))
  done
  [ -n "$pr_json" ] && printf '%s' "$pr_json" > "$dir/pr.json"
  ( export PATH="$tmp:$PATH" MOCK_DIR="$dir" PR_SHA=deadbeef PR_NUMBER=42 \
      GITHUB_REPOSITORY=o/r POLL_SECS=0 GRACE_SECS=1 MAX_WAIT_SECS=3
    bash "$here/gate_full_suite.sh" > "$dir/out.log" 2>&1 )
  local rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL: $name (exit $rc, want $want)"
    cat "$dir/out.log"
    fails=1
  elif [ -f "$dir/endpoint_error" ]; then
    echo "FAIL: $name (mock gh got an unexpected call)"
    cat "$dir/endpoint_error"
    fails=1
  elif [ -n "$want_log" ] && ! grep -Eq "$want_log" "$dir/out.log"; then
    echo "FAIL: $name (log does not match: $want_log)"
    cat "$dir/out.log"
    fails=1
  else
    echo "ok: $name"
  fi
  want_log="" pr_json="" raw_body=""
}

expect "both green -> proceed" 0 "$PC_OK, $DOCS_OK, $OTHER, $NULL_NAME"
expect "docs build failed -> blocked" 1 "$PC_OK, $DOCS_BAD"
expect "pre-commit failed -> blocked" 1 "$PC_BAD"
expect "pending then green -> proceed" 0 "$PC_PENDING" "$PC_OK, $DOCS_OK"
want_log="never appeared.*Deploy Documentation"
expect "docs run absent (path-filtered) -> proceed after grace" 0 "$PC_OK"
expect "API outage -> fail open" 0
want_log="FAILING OPEN"
expect "pending past MAX_WAIT -> fail open" 0 "$PC_PENDING"
want_log="FAILING OPEN"
expect "unrelated runs only -> no grace, fail open at MAX_WAIT" 0 "$OTHER"
expect "cancelled docs then green -> proceed" 0 \
  "$PC_OK, $DOCS_CANCELLED" "$PC_OK, $DOCS_OK"
want_log="FAILING OPEN"
expect "cancelled docs forever -> fail open at MAX_WAIT" 0 "$PC_OK, $DOCS_CANCELLED"
want_log="FAILING OPEN"
expect "pre-commit absent -> no grace, fail open at MAX_WAIT" 0 "$DOCS_OK"
expect "duplicate run names -> latest wins" 0 "$PC_BAD, $PC_OK_RERUN, $DOCS_OK"
raw_body=1
expect "garbage response body -> fail open" 0 "this is not json"
pr_json='{"labels": [{"name": "other"}]}'
expect "ready label removed mid-gate -> blocked" 1 "$PC_OK, $DOCS_OK"

exit "$fails"
