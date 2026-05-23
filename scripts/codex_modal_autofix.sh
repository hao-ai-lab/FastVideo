#!/usr/bin/env bash
set -u
set -o pipefail

MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
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
  tail -n 1000 "logs/train_attempt_${i}.log" > logs/last_crash.log

  echo "Asking Codex to inspect logs and patch..."

  codex exec \
    -m gpt-5.5 \
    -C . \
    -c 'model_reasoning_effort="high"' \
    -c 'sandbox_mode="workspace-write"' \
    -c 'approval_policy="never"' \
    -o "logs/codex_attempt_${i}.md" \
    "
The Modal training command crashed.

Command:
MODAL_PROFILE=hao-ai-lab modal run modal_train_genrl.py

Crash log:
logs/last_crash.log

Full log:
logs/train_attempt_${i}.log

Task:
1. Read logs/last_crash.log.
2. If the tail only shows shutdown/wrapper noise, inspect the full log and
   find the first real Python exception, CUDA OOM, RuntimeError, ImportError,
   AttributeError, or rank traceback.
3. Diagnose the actual root cause.
4. Patch the minimal code/config needed.
5. Do NOT run the full Modal training command yourself.
6. Run only cheap local validation:
   - python -m py_compile modal_train_genrl.py
   - python -m compileall fastvideo modal_train_genrl.py
   - any targeted import/config checks that do not launch full training.
7. Write a concise summary of what you changed.
"

  codex_status=$?

  if [ "$codex_status" -ne 0 ]; then
    echo "Codex failed with exit code $codex_status."
    echo "Run this to inspect your installed Codex flags:"
    echo "  codex exec --help"
    exit "$codex_status"
  fi

  echo "Codex patch finished. Retrying training..."
done

echo "Failed after $MAX_ATTEMPTS attempts."
exit 1
