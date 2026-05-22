#!/usr/bin/env bash
set -u
set -o pipefail

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
TRAIN_CMD="MODAL_PROFILE=hao-ai-lab modal run modal_train_genrl.py"

mkdir -p logs

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "========================================================"
  echo "Training attempt $i / $MAX_ATTEMPTS"
  echo "Command: $TRAIN_CMD"
  echo "========================================================"

  bash -lc "$TRAIN_CMD" 2>&1 | tee "logs/train_attempt_${i}.log"
  status=${PIPESTATUS[0]}

  if [ "$status" -eq 0 ]; then
    echo "Training succeeded on attempt $i."
    exit 0
  fi

  echo "Training crashed with exit code $status."

  tail -n 500 "logs/train_attempt_${i}.log" > logs/last_crash.log

  echo "Asking Codex to inspect logs and patch..."
  codex exec \
    --cd . \
    --sandbox workspace-write \
    --ask-for-approval never \
    --output-last-message "logs/codex_attempt_${i}.md" \
    "
The Modal training command crashed.

Command:
MODAL_PROFILE=hao-ai-lab modal run modal_train_genrl.py

Read:
logs/last_crash.log

Task:
1. Diagnose the root cause from the crash log.
2. Make the minimal necessary code/config fix.
3. Run only cheap validation commands, such as:
   python -m py_compile modal_train_genrl.py
   python -m compileall .
   targeted import checks
4. Do NOT run the full Modal training command yourself.
5. Write a concise summary of the fix.
"

  codex_status=$?
  if [ "$codex_status" -ne 0 ]; then
    echo "Codex failed with exit code $codex_status."
    exit "$codex_status"
  fi

  echo "Codex patch done. Retrying training..."
done

echo "Failed after $MAX_ATTEMPTS attempts."
exit 1
