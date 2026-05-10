# Prompt Files

This directory contains the editable system prompts used by the
`PromptEnhancer` in `server/prompt_enhancer.py`.

## `next_segment_system_prompt.md`

Use this prompt for guided continuation.

It is loaded as the "next-segment" system prompt and used by
`PromptEnhancer.enhance_prompt()` for the normal continuation flow when the
request includes:

- locked prior segments
- a new user conditioning prompt describing what should happen next

In that path, the model is asked to write exactly one new segment prompt that
continues from the locked history while satisfying the user's requested next
beat.

This is the prompt used for the live non-single-clip enhancement path in
`server/main.py`.

## `auto_extension_system_prompt.md`

Use this prompt for autonomous prompt expansion.

It is loaded as the "auto-extension" system prompt and used in two different
paths:

1. `PromptEnhancer.generate_auto_prompt()`

This is the background auto-extension flow. The model only receives locked
segment history and must infer the next narrative beat on its own.

1. `PromptEnhancer.enhance_prompt()` in single-clip mode

This is the single-clip expansion flow. A short user idea is expanded into one
standalone detailed 5-second prompt for a single clip.

## `rewrite_user_system_prompt.md`

Use this prompt for new-rollout generation from a user's initial rollout
instruction.

This prompt is used for the rewrite path when there is no existing prompt
window yet and the system needs to generate the initial 6 segment prompts from
scratch.

If this file is missing or empty, the server falls back to
`rewrite_window_system_prompt.md` so startup does not break while the file is
still being drafted.

## Practical difference

- `next_segment_system_prompt.md` is for "continue based on the user's new
  instruction."
- `auto_extension_system_prompt.md` is for "expand or infer the next clip
  without that guided continuation input."
