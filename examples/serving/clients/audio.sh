#!/usr/bin/env bash
# Requires curl and jq. Run after starting the FastVideo audio server.
set -euo pipefail

base_url="${FASTVIDEO_BASE_URL:-http://127.0.0.1:8000/v1}"
api_key="${FASTVIDEO_API_KEY:-local}"
model="${FASTVIDEO_MODEL:-stable-audio}"
prompt="${FASTVIDEO_PROMPT:-Rain on a metal roof, distant thunder.}"
seconds="${FASTVIDEO_SECONDS:-6}"
payload=$(jq -n --arg model "$model" --arg prompt "$prompt" --argjson seconds "$seconds" \
  '{model: $model, prompt: $prompt, seconds: $seconds}')
if [[ -n "${FASTVIDEO_VIDEO_REFERENCE:-}" ]]; then
  payload=$(jq --arg video "$FASTVIDEO_VIDEO_REFERENCE" '. + {video_reference: {video_url: $video}}' <<< "$payload")
fi

job=$(curl --fail-with-body --silent --show-error --max-time 60 \
  "$base_url/videos" \
  -H "Authorization: Bearer $api_key" \
  -H 'Content-Type: application/json' \
  -d "$payload")
job_id=$(jq -er '.id' <<< "$job")
printf 'Submitted %s\n' "$job_id"
deadline=$((SECONDS + 1800))

while true; do
  status=$(jq -er '.status' <<< "$job")
  case "$status" in
    completed) break ;;
    failed) jq -r '.error.message' <<< "$job" >&2; exit 1 ;;
    queued|in_progress) ;;
    *) printf 'Unexpected job status: %s\n' "$status" >&2; exit 1 ;;
  esac
  remaining=$((deadline - SECONDS))
  if (( remaining <= 0 )); then
    printf 'Polling timed out; job %s may still be running\n' "$job_id" >&2
    exit 1
  fi
  if (( remaining > 2 )); then sleep 2; else sleep "$remaining"; fi
  remaining=$((deadline - SECONDS))
  if (( remaining <= 0 )); then continue; fi
  if (( remaining > 60 )); then remaining=60; fi
  job=$(curl --fail-with-body --silent --show-error --max-time "$remaining" \
    -H "Authorization: Bearer $api_key" "$base_url/videos/$job_id")
done

media_type=$(jq -er '.media_type' <<< "$job")
if [[ "$media_type" != "audio/wav" ]]; then
  printf 'Expected audio/wav, got %s\n' "$media_type" >&2
  exit 1
fi
curl --fail-with-body --silent --show-error --max-time 300 \
  -H "Authorization: Bearer $api_key" \
  "$base_url/videos/$job_id/content" --output "$job_id.wav"
printf 'Saved %s.wav\n' "$job_id"
