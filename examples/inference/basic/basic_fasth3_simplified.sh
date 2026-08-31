#!/usr/bin/env bash

PROMPT="integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds."

# Options: dense-datafree, vsa-datafree, vsa-synthetic-step1300, vsa-synthetic-step1900
VARIANT="vsa-datafree"

exec python "$(dirname "$0")/basic_fasth3_simplified.py" \
  "$VARIANT" \
  --prompt "$PROMPT"
