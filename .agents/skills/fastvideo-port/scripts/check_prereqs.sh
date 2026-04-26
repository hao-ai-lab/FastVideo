#!/usr/bin/env bash
# check_prereqs.sh — run this before starting a fastvideo-port.
# Exits non-zero on any hard failure; prints warnings for soft issues.
#
# Usage:
#   bash .agents/skills/fastvideo-port/scripts/check_prereqs.sh \
#       --model davinci-magihuman \
#       --hf_ids "google/t5gemma-9b stabilityai/stable-audio-open-1.0"
#
# Hard failures (exit 1): missing GPU, uv not installed, FastVideo not installed
# Warnings (continue):    missing tokens, weights not downloaded yet

set -euo pipefail

MODEL_NAME=""
HF_IDS=""
PASS=0
WARN=0
FAIL=0

RED='\033[0;31m'
YEL='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'

ok()   { echo -e "${GRN}[OK]${NC}    $1"; ((PASS++)); }
warn() { echo -e "${YEL}[WARN]${NC}  $1"; ((WARN++)); }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; ((FAIL++)); }

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_NAME="$2"; shift 2 ;;
    --hf_ids) HF_IDS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "================================================="
echo " FastVideo Port — Prerequisites Check"
[[ -n "$MODEL_NAME" ]] && echo " Model: $MODEL_NAME"
echo "================================================="
echo ""

# ── 1. FastVideo installed ────────────────────────────────────────────────────
echo "[ Environment ]"
if python -c "import fastvideo" 2>/dev/null; then
  ok "fastvideo importable"
else
  fail "fastvideo not importable — run: uv pip install -e .[dev]"
fi

if command -v uv &>/dev/null; then
  ok "uv available"
else
  warn "uv not found — pip install uv  (needed for editable install)"
fi

# ── 2. GPU ─────────────────────────────────────────────────────────────────────
echo ""
echo "[ GPU ]"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
  VRAM=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1))")
  ok "GPU: $GPU_NAME — ${VRAM}GB VRAM"
  if python -c "import torch; assert torch.cuda.get_device_properties(0).total_memory > 20e9" 2>/dev/null; then
    ok "VRAM >= 20GB (sufficient for dual-model alignment tests)"
  else
    warn "VRAM < 20GB — alignment tests may OOM when loading both official + FastVideo models simultaneously"
  fi
else
  fail "No CUDA GPU detected — alignment tests and training require GPU"
fi

# ── 3. Attention backend ───────────────────────────────────────────────────────
echo ""
echo "[ Attention Backend ]"
if [[ "${FASTVIDEO_ATTENTION_BACKEND:-}" == "TORCH_SDPA" ]]; then
  ok "FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA (correct for alignment tests)"
else
  warn "FASTVIDEO_ATTENTION_BACKEND not set to TORCH_SDPA — set it before running alignment tests to avoid backend-specific numerical divergence"
  echo "       export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA"
fi

# ── 4. Credentials ────────────────────────────────────────────────────────────
echo ""
echo "[ Credentials ]"
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN is set"
else
  warn "HF_TOKEN not set — required for gated models (Llama, Gemma, t5gemma, etc.)"
  echo "       huggingface-cli login  OR  export HF_TOKEN=hf_..."
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  ok "GITHUB_TOKEN is set (recon.py will use 5000 req/hr limit)"
else
  warn "GITHUB_TOKEN not set — recon.py limited to 60 GitHub API req/hr"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ok "WANDB_API_KEY is set"
else
  warn "WANDB_API_KEY not set — validation logging disabled (not required for porting)"
fi

# ── 5. HF gating check ────────────────────────────────────────────────────────
if [[ -n "$HF_IDS" ]]; then
  echo ""
  echo "[ HF Model Access ]"
  for hf_id in $HF_IDS; do
    GATED=$(HF_ID="$hf_id" python - <<'PY' 2>/dev/null || echo "error"
import os
from huggingface_hub import model_info
try:
    hf_id = os.environ.get("HF_ID")
    token = os.environ.get("HF_TOKEN")
    info = model_info(hf_id, token=token)
    gated = getattr(info, "gated", False)
    print("gated" if gated else "open")
except Exception as e:
    print(f"error: {e}")
PY
)
    case "$GATED" in
      open)  ok "$hf_id — open access" ;;
      gated) warn "$hf_id — GATED: request access at https://huggingface.co/$hf_id" ;;
      error*) warn "$hf_id — could not check (token missing or network error): $GATED" ;;
    esac
  done
fi

# ── 6. Weights on disk ────────────────────────────────────────────────────────
if [[ -n "$MODEL_NAME" ]]; then
  echo ""
  echo "[ Weights ]"
  WEIGHTS_DIR="official_weights/$MODEL_NAME"
  if [[ -d "$WEIGHTS_DIR" ]]; then
    FILE_COUNT=$(find "$WEIGHTS_DIR" -type f | wc -l | tr -d ' ')
    ok "$WEIGHTS_DIR exists ($FILE_COUNT files)"
  else
    warn "$WEIGHTS_DIR not found — download weights manually before alignment tests"
    echo "       mkdir -p $WEIGHTS_DIR && <download command>"
  fi
fi

# ── 7. Pre-commit ─────────────────────────────────────────────────────────────
echo ""
echo "[ Code Quality ]"
if command -v pre-commit &>/dev/null && [[ -f ".pre-commit-config.yaml" ]]; then
  ok "pre-commit available"
else
  warn "pre-commit not installed or config missing — run: pre-commit install"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo -e " ${GRN}PASS: $PASS${NC}  ${YEL}WARN: $WARN${NC}  ${RED}FAIL: $FAIL${NC}"
echo "================================================="

if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "Hard failures detected — fix before proceeding."
  exit 1
elif [[ $WARN -gt 0 ]]; then
  echo ""
  echo "Warnings present — review above before alignment tests."
  exit 0
else
  echo ""
  echo "All checks passed. Ready to port."
  exit 0
fi