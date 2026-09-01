#!/usr/bin/env bash
# Requires curl and jq. Run after starting the FastVideo server.
set -euo pipefail

base_url="${FASTVIDEO_BASE_URL:-http://127.0.0.1:8000/v1}"
api_key="${FASTVIDEO_API_KEY:-local}"
model="${FASTVIDEO_MODEL:-fasth3}"
payload=$(jq -n --arg model "$model" \
  '{model: $model, prompt: "A fox runs through fresh snow."}')

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

curl --fail-with-body --silent --show-error --max-time 300 \
  -H "Authorization: Bearer $api_key" \
  "$base_url/videos/$job_id/content" --output "$job_id.mp4"
printf 'Saved %s.mp4\n' "$job_id"
