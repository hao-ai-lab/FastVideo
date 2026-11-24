# Dreamverse H3 prompt research

Research date: 2026-09-15. Scope: MiniMax-H3 full-checkpoint prompting, not every
Hailuo service or the FastH3 Preview LoRA. This is a targeted sample of public
discussions, not a survey or a reproducible quality benchmark.

The proposed definitions are in [H3 system prompt v1](dreamverse-h3-system-prompt-v1.md).
Neither document activates new runtime prompts or demonstrates their generation quality.

## Two different prompts

A **system prompt** instructs the rewriting LLM. The resulting **video prompt**
instructs H3. The intended flow is creative request and verified media context,
then LLM rewriting, application validation, and finally video generation.
Do not send the rewriting instructions themselves to the video model.

## Evidence hierarchy

Use the official [base guide][base], [reference guide][ref], and
[prompt-writing specification][official-skill] for model-facing syntax.
First-person reports identify practical failure cases; third-party tool releases
describe workflows, not independent evidence of universal quality improvements.
Dates below are original post dates; comments may have been added later.

## Seven representative Reddit discussions

### 1. Prompt Guide — 2026-08-03

[Original discussion][reddit-guide]: users pass both official guides to an LLM.
One commenter reports off-screen delivery instead of visible character speech,
then reports improvement after binding dialogue to the character.
Useful rule: specify the speaker, language, and exact words together.
This is anecdotal troubleshooting, not a measured success rate.

### 2. Community system instructions — 2026-08-05

[Original discussion][reddit-template]: an Ollama/ComfyUI user shares a template
that separates action, ambience, and music, and identifies what should stay fixed.
Retain the distinction between preserved and changing content. Do not copy its
universal first-image anchor, camera-gaze ban, compulsory action lead-ins, or
plain quotation marks in place of dialogue tags. Comments challenge that syntax;
the template describes one image-to-video workflow, not all three modes.

### 3. Official team AMA — 2026-08-06

[Original discussion][reddit-ama]: identified H3 researchers direct users to the
official guides and prompt-writing specification. In a [direct reply][ama-quality],
Nero acknowledges different visual-quality tendencies between FL2VA and Ref2VA
and recommends high-quality reference inputs. This is official guidance, but not
an independent comparison. Better prompts cannot fix every conditioning or model
limitation, and a local template is not the hosted Context-IR system.

### 4. The H3 Gibberish Problem Solved! — 2026-08-08

[Original discussion][reddit-gibberish]: the author reports improvement after
explicitly disabling score. A commenter still reports unwanted music about 20%
of the time; that number is another personal estimate, not a benchmark.
Use explicit audio intent, but do not promise that one field eliminates gibberish.
Plugin and LoRA differences also limit comparisons. Do not move voice-over into
the music field or replace official dialogue tags based on isolated comments.

### 5. Proper prompt structure — 2026-08-20

[Original discussion][reddit-structure]: the author reports better dialogue over
a long sequence after using the reference guide. Comments recommend retaining
the base guide's camera rules; others describe enhancers adding, omitting, or
reversing requested details. Adopt shared rules plus mode-specific structure,
with fidelity to the user's intent. The claimed 30-by-15-second success and
cross-mode reuse of reference syntax are not independently reproduced here.

### 6. H3 Prompt Writer — 2026-08-30

[Original discussion][reddit-writer]: a third-party tool announcement, with useful
user reports about image/video/audio workflows. Users describe invented timing,
ambiguous character references, and missing sound sections or dialogue tags.
The author recommends preserving source-video timing for edits; explicit subject
labels reportedly help a multi-person case. Add validation after rewriting.
Built-in guides and a preferred LLM do not guarantee complete or correct output.

### 7. Precise choreography discussion — 2026-09-04

[Original discussion][reddit-choreography]: a user already following the official
guide still needs repeated revisions for attacks, dodges, movement directions,
and final positions. Describe actors, targets, spatial relations, and sequence
explicitly; reduce action density when appropriate. This is failure evidence,
not proof of a universal remedy. Do not treat timestamps as frame-exact control
or a commenter's fixed overlap-frame count as a Dreamverse requirement.

## Model conventions versus product choices

| Concern | Proposed treatment | Basis |
| --- | --- | --- |
| Base modes | Three core fields; image alignment only for keyframe tasks | Official base guide |
| FL2VA | Describe the observable path between actual opening and ending frames | Official base guide |
| Ref2VA | Six ordered sections; stable references and explicit retention roles | Official reference guide |
| Language | English descriptions; retain original dialogue, lyrics, and visible text | Official guides |
| Speech and shots | Stable speaker IDs, dialogue tags, and ordered cuts within duration | Official guides |
| Creative defaults | No invented dialogue, score, or plot; prefer a readable short action | Dreamverse product choice |
| Reliability | Validate fields, references, segment count, and verbatim content | Application design, not model syntax |

For first-frame-only conditioning, use I2VA semantics rather than inventing a
last image. Reference subjects and speaker IDs are separate namespaces. A motion
reference is not automatically a video-editing task. The reference guide also
distinguishes copying audio from using its timbre or other properties as guidance.
These distinctions must survive rewriting; schema validity alone is insufficient.

## Integration boundary

The existing [Dreamverse README](../../apps/dreamverse/README.md) documents a
FastH3 Preview profile. Full three-mode application plumbing is proposed separately
in [PR #1835](https://github.com/hao-ai-lab/FastVideo/pull/1835); it is not assumed
to be available merely because these prompt definitions exist.

- Existing prompt files contain LTX-specific assumptions. Route model-specific
  definitions explicitly instead of globally replacing another model's prompts.
- Single-clip enhancement, continuation, and rollout rewriting have different
  outer response contracts. A rollout JSON wrapper cannot replace them all.
- The caller must supply validated duration, conditioning, reference ordering,
  content observations, and edit scope. Asset names alone are not media understanding.
- Derive media labels from the actual pipeline presentation, including enabled
  video-audio tracks. Never guess audio indices from upload order or filenames.
- Budget output for all requested sections and segments; detect truncation and
  missing content. Several detailed Ref2VA segments may require separate calls.

The [companion proposal](dreamverse-h3-system-prompt-v1.md) defines the candidate
contract. Runtime wiring, provider behavior, and quality acceptance are follow-up
work; prior mode-switching smoke tests do not validate this new system prompt.

## Original T2VA illustration

Creative request: an orange cat watches rain through a window; slowly move the
camera closer, with no speech or music. This original illustration is the inner
video prompt, not the system prompt, and has not had GPU quality validation.

```text
integrated_multimodal_description: [Shot 1] A naturalistic medium shot frames an orange cat on a wooden windowsill beside rain-streaked glass. The camera pushes in with small amplitude at slow speed as the cat turns toward the falling rain, then rests in profile. Soft overcast light reveals its fur and the droplets. Its mouth remains closed; there is no speech or narration.

overall_soundscape: Gentle rain taps against the glass above quiet indoor room tone.

non_diegetic_music: N/A
```

FL2VA needs real frame anchors and a transition between them. Ref2VA needs a
verified account of which supplied assets establish appearance, action, or setting.
Do not fabricate reference labels to adapt this text-only example.

## Proposed A/B validation

1. Cover quiet scenes, one Chinese-speaking character, alternating speakers,
   first/last-frame transitions, first-frame-only input, and mixed references
   containing an image, a video with enabled audio, and a separate audio asset.
2. Compare the raw request with v1 expansion. An existing LTX template may be an
   additional diagnostic baseline, not the definition of correct H3 formatting.
3. Hold checkpoint, asset ordering, resolution, frame count, sampling settings,
   acceleration configuration, and a predeclared seed set constant between arms.
4. Check schema, section order, references, exact dialogue, and completion before
   inference. Reject failures explicitly; record retries and rewriting latency.
5. Blind-review intent, identity/reference fidelity, actions, speaker assignment,
   unexpected audio, camera behavior, and timing. Record prompt length and failures.
6. Compare concise and officially detailed Ref2VA descriptions separately rather
   than assuming that longer or shorter prompts always perform better.

Publish the inputs, configuration, rubric, and failures before claiming improvement.
No A/B result or successful generation with these new definitions is claimed here.

[base]: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
[ref]: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
[official-skill]: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md
[reddit-guide]: https://www.reddit.com/r/StableDiffusion/comments/1veauqw/minimax_h3_prompt_guide/
[reddit-template]: https://www.reddit.com/r/StableDiffusion/comments/1vgjqc1/here_is_a_set_of_instructions_to_use_with_you_llm/
[reddit-ama]: https://www.reddit.com/r/StableDiffusion/comments/1vh9rtw/ama_minimax_h3_team_ask_us_anything_about_our/
[ama-quality]: https://www.reddit.com/r/StableDiffusion/comments/1vh9rtw/comment/p29qqaa/
[reddit-gibberish]: https://www.reddit.com/r/StableDiffusion/comments/1viui81/the_h3_gibberish_problem_solved/
[reddit-structure]: https://www.reddit.com/r/StableDiffusion/comments/1vtnhvj/psa_proper_prompt_structure_really_matters_in_h3/
[reddit-writer]: https://www.reddit.com/r/StableDiffusion/comments/1w2mvtv/minimax_h3_prompt_writer_v043_windows_standalone/
[reddit-choreography]: https://www.reddit.com/r/StableDiffusion/comments/1w7d4go/minimax_h3_prompting_guide_discussion_best/
