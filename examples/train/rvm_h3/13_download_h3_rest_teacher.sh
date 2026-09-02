#!/usr/bin/env bash
set -euo pipefail

# Download only the full-H3 components used by REST cache generation. Text is
# already preprocessed, and reward alignment is visual, so the 32B text-encoder
# weights and audio decoder are intentionally excluded.
source "$(dirname "$0")/common.sh"
activate_rvm_env

command -v hf >/dev/null 2>&1 || {
    echo "The Hugging Face 'hf' CLI is required (pip install huggingface_hub)." >&2
    exit 1
}

mkdir -p "${H3_TEACHER_MODEL_DIR}"
source_manifest="${H3_TEACHER_MODEL_DIR}/FASTVIDEO_SOURCE.json"
if [[ -f "${source_manifest}" ]]; then
    python - "${source_manifest}" "${H3_TEACHER_REPO}" "${H3_TEACHER_REVISION}" <<'PY'
import json
import sys
path, repo, revision = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("repository") != repo or payload.get("revision") != revision:
    raise SystemExit(
        "Existing H3 teacher directory was created from a different source: "
        f"{payload}. Remove it or set H3_TEACHER_MODEL_DIR to a new path."
    )
PY
fi

hf download "${H3_TEACHER_REPO}" \
    --revision "${H3_TEACHER_REVISION}" \
    --local-dir "${H3_TEACHER_MODEL_DIR}" \
    --include \
        "model_index.json" \
        "modular_model_index.json" \
        "README.md" \
        "LICENSE*" \
        "transformer/*" \
        "vae/*" \
        "scheduler/*" \
        "audio_scheduler/*" \
        "text_encoder/config.json" \
        "tokenizer/*" \
        "processor/*"

python - "${H3_TEACHER_MODEL_DIR}" "${H3_TEACHER_REPO}" "${H3_TEACHER_REVISION}" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = sys.argv[2]
revision = sys.argv[3]
manifest_candidates = [root / "model_index.json", root / "modular_model_index.json"]
if not any(path.is_file() for path in manifest_candidates):
    raise SystemExit(f"No Diffusers model manifest found under {root}")
for component in ("transformer", "vae"):
    directory = root / component
    if not (directory / "config.json").is_file():
        raise SystemExit(f"Missing {component}/config.json under {root}")
    if not any(directory.glob("*.safetensors")):
        raise SystemExit(f"No {component} safetensors found under {root}")
try:
    fastvideo_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
    ).strip()
except Exception:
    fastvideo_head = None
payload = {
    "repository": repo,
    "revision": revision,
    "fastvideo_head_at_download": fastvideo_head,
    "selective_components": [
        "transformer",
        "vae",
        "scheduler",
        "audio_scheduler",
        "text_encoder/config.json",
        "tokenizer",
        "processor",
    ],
    "excluded_large_components": ["text_encoder weights", "audio_vae weights"],
}
(root / "FASTVIDEO_SOURCE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Pinned full-H3 REST teacher is ready at ${H3_TEACHER_MODEL_DIR}"
