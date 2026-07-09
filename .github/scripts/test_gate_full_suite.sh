#!/usr/bin/env bash
# Self-test for gate_full_suite.sh using a mocked `gh`. No network, runs on
# any dev box: bash .github/scripts/test_gate_full_suite.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Mock gh: serves $MOCK_DIR/response_<call#>.json, sticking on the highest
# existing file; exits 1 if none exist (simulates a GitHub API outage).
cat > "$tmp/gh" <<'EOF'
#!/usr/bin/env bash
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
EOF
chmod +x "$tmp/gh"

PC_OK='{"name": "pre-commit", "id": 1, "status": "completed", "conclusion": "success"}'
PC_PENDING='{"name": "pre-commit", "id": 1, "status": "in_progress", "conclusion": null}'
DOCS_OK='{"name": "Deploy Documentation", "id": 2, "status": "completed", "conclusion": "success"}'
DOCS_BAD='{"name": "Deploy Documentation", "id": 2, "status": "completed", "conclusion": "failure"}'
OTHER='{"name": "Trigger Full Suite", "id": 3, "status": "in_progress", "conclusion": null}'

fails=0
expect() { # <name> <expected-exit> <response json>...
  local name=$1 want=$2 dir i=1
  shift 2
  dir=$(mktemp -d "$tmp/test_XXXXXX")
  for body in "$@"; do
    printf '{"workflow_runs": [%s]}' "$body" > "$dir/response_$i.json"
    i=$(( i + 1 ))
  done
  ( export PATH="$tmp:$PATH" MOCK_DIR="$dir" PR_SHA=deadbeef \
      GITHUB_REPOSITORY=o/r POLL_SECS=0 GRACE_SECS=1 MAX_WAIT_SECS=3
    bash "$here/gate_full_suite.sh" > "$dir/out.log" 2>&1 )
  local rc=$?
  if [ "$rc" -eq "$want" ]; then
    echo "ok: $name"
  else
    echo "FAIL: $name (exit $rc, want $want)"
    cat "$dir/out.log"
    fails=1
  fi
}

expect "both green -> proceed" 0 "$PC_OK, $DOCS_OK, $OTHER"
expect "docs build failed -> blocked" 1 "$PC_OK, $DOCS_BAD"
expect "pre-commit failed -> blocked" 1 \
  '{"name": "pre-commit", "id": 1, "status": "completed", "conclusion": "failure"}'
expect "pending then green -> proceed" 0 "$PC_PENDING" "$PC_OK, $DOCS_OK"
expect "docs run absent (path-filtered) -> proceed after grace" 0 "$PC_OK"
expect "API outage -> fail open" 0
expect "pending past MAX_WAIT -> fail open" 0 "$PC_PENDING"
expect "unrelated runs only -> proceed after grace" 0 "$OTHER"

exit "$fails"
