---
name: h3-job-writer
description: Turn a screenplay scene plus reference images/audio into MiniMax-H3 Ref2VA job JSONs for FastVideo Studio, one job per clip, following the H3 prompt format and the timing/staging rules learned from real generations.
---

# Writing MiniMax-H3 jobs from a script

You turn one scene of a script into a sequence of FastVideo Studio jobs, one per clip.
Every clip is generated independently, so each job's prompt must stand on its own.

## Workflow

1. Read the script and the reference list you were given. Note every character, the setting,
   every action, and every line of dialogue in order. For the full prompt format with a complete
   example, read `.dsh/skills/h3-job-writer/official-guide.md` (MiniMax's own guide). Where it
   disagrees with this file, this file wins.
   Look at every image reference with `read_image` (see **Looking at references**).
2. Plan the clips (see **Splitting into clips**). Write the plan down before drafting.
3. Draft each clip as a job object and call `h3_write_job` for it. The tool validates the job
   and writes it; fix every error it reports and call it again until it succeeds.
4. Finish with a short summary listing each clip: its file, length, and what happens in it.

Never post, queue, or start jobs. A human reviews every draft first.

## Job JSON

```json
{
  "model_id": "MiniMaxAI/MiniMax-H3",
  "name": "<scene>-clip-01",
  "workload_type": "i2v",
  "prompt": ["subject_definitions:", "...", "", "summary:", "...", ...],
  "references": [
    {"source": "/abs/path/fletcher_ref.png", "media_type": "image"},
    {"source": "/abs/path/fletcher.mp3", "media_type": "audio"}
  ],
  "negative_prompt": "",
  "guidance_scale": 1,
  "num_frames": 288,
  "fps": 24,
  "height": 768,
  "width": 1344,
  "num_inference_steps": 50,
  "seed": 0,
  "num_gpus": 4,
  "use_fsdp_inference": true
}
```

- `prompt` is a list of lines; they are joined with newlines before posting.
- `num_frames` = clip seconds × 24, at most **345** (14.4 s). Size the clip to its scripted content (see rule 5).
- Clip names share one stem and end in `-clip-NN` so Studio groups them into one scene.
- Only use reference paths you were given. Include only the references a clip actually uses.
- At most 9 images, 3 videos and 3 audio files per job, 12 references in total, and at least one
  image or video (audio alone is refused).

## Prompt sections (in this order)

```
subject_definitions:
<Subject 1> is <who> from <Picture 1>: <appearance, wardrobe, demeanor>.
<Subject 2> is ...
<Subject N> is the <location> from <Picture N>: <layout, fixtures>. It also sets the lighting for the whole scene: <light>. <What the picture is / is not, see rule 7>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1), containing a spoken English vocal layer.
<Video 1> is ...                       (only when continuing from a previous clip's tail)

summary:
[reference generation + audio reference] The target video shows ... <global rules: who stays where, who never does what>. The setting is the <location> from <Subject N>, and the whole scene shares its <light>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <what is kept>; his own background and lighting are not, and he is lit by the <location>'s light.
<Subject N> (appears in [Shot 1], [Shot 2]): partially_preserved - <setting elements kept>; only the setting is retained, with no camera framing carried over.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 1> without copying the original signal.

detailed_description:
[Shot 1] <Framing>. The camera is fixed and does not move. <Who is where>. <Action>. <Subject 1> (S1) speaks in the voice timbre referenced from <Audio 1>. (S1) <d>[English] Line.</d>
[Shot 2] At 00:04.500, the shot cuts to <framing>. ...

overall_soundscape:
<diegetic sound, with what must NOT be heard>

non_diegetic_music:
N/A
```

Numbering: `<Picture k>`, `<Video k>` and `<Audio k>` each count their own kind, in the order the
references appear in `references`. A **video with sound** (marked in `references.md`) takes the
next `<Audio k>` as well as its `<Video k>`, even if the prompt never mentions that soundtrack, so
every audio file after it shifts up by one. Example: `[video with sound, voice.mp3, face.png]` are
`<Video 1>` + `<Audio 1>`, then `<Audio 2>`, then `<Picture 1>`. (The official guide says a video
does not create an audio label just because it has sound; FastVideo's pipeline does, and this rule
follows the pipeline.) Subjects are numbered in definition order. The tag in the
summary is `[reference generation + audio reference]` when any audio reference is used, else
`[reference generation]`; add `video continuation + ` in front when a `<Video 1>` tail is used.

Dialogue: `(S1) <d>[English] text</d>`. Add `<cutoff>` after a line that is interrupted.
Speaker IDs (S1, S2...) match the subject numbers of the speakers.

The `appears in` lists in `retention_analysis` must name exactly the shots where that subject is
on screen. Keep them in sync whenever shots change.

## Rules learned from real generations

These are the reasons previous drafts failed. Follow all of them.

1. **Repeated actions become a rhythm the model continues.** "Slap on each four" turned into a
   steady beat that kept going. Keep a repeated action to the fewest repetitions the script needs,
   never write exceptions inside a pattern ("the third one gets no slap"), and end the pattern by
   ending the clip or cutting to a new shot. Counts like "exactly two" do not stop a rhythm.
2. **Keep rhythmic speech in one dialogue tag.** Splitting a count into several `<d>` tags with
   prose between them makes the speaker pause at every break. Write one continuous line
   (`One-two-three-four-one-two-three-four-`) and describe actions that land on it in prose after it.
3. **Write speech the way it should sound.** Ellipses slow delivery ("One... two..."); use dashes
   for fast speech. Spell words out ("four", not "fo-"). State tempo explicitly when it matters
   ("150 BPM, 0.4 seconds per number, never pausing").
4. **Timestamps are loose guides.** Use them for shot cuts and rough placement. Do not rely on a
   timestamp to sync an action to a word: anchor the action to the word itself ("as he says
   'four'"). Never pin an action to a time that the speech could drift away from.
5. **Every second of the clip must be scripted.** Spare time at the end gets filled with more of
   whatever was happening (more slaps, the music starting over). End the clip shortly after its
   last scripted beat: estimate speech at ~2.5 words/second (faster for counting at the stated
   tempo) plus the time each action needs.
6. **Pin every character's position in every shot.** If a shot does not say where someone is, the
   model invents it (a second copy of a character, someone walking over early). Put a whole-clip
   rule in the summary ("X stays seated behind his drum kit for the entire video; he never stands
   up", "there is only one drummer"), and restate positions in each shot. Use separate singles,
   not two-shots, when characters must stay apart. Frame anyone whose hands matter fully in shot;
   a half-framed actor's gestures read as extra actions.
7. **Reference images win in wide shots.** An empty location photo produces an empty room. Say
   what the picture is ("This picture shows the room before the band arrives") and name the
   background people in every shot ("Every chair is filled: the band musicians sit at their
   stands..."), not just once in the subject definition.
8. **Audio references play from their start.** A music reference restarts in every clip that uses
   it. Use music only in the clip it belongs to, and when music must stop, say when in the
   summary, the shot, and the soundscape ("the music stops dead at the crash; silence after").
9. **Stale text keeps steering.** After changing a shot, re-read the summary, retention lists, and
   soundscape for anything that now contradicts it ("slapping him on each beat") and fix it.
10. **Do not add direction the script does not ask for.** No added shouting, roaring, or
    emphasis unless the script says so; all-caps dialogue may be written in normal case only if
    the user asks. Keep the script's dialogue verbatim except for obvious typos you report.

## Finding missing references

Only when `references.md` says you may use `find_reference_image`:

- First decide which subjects need a reference: every named person on screen, the main location,
  and any object whose exact look matters. A subject already covered by a listed file keeps that
  file; never search for it or replace it.
- Search for the real thing the script names: a real person or actor by full name ("Robert
  Pattinson"), a place by what it is ("jazz rehearsal room interior"). For a character an actor
  plays, search the actor only when the user's direction names the actor.
- Call it once per subject with `count` 2-3, look at every result with `read_image`, and use
  exactly one image per subject: a clear photograph of only that subject (see **Looking at
  references** for what to reject). Add "photo" to a place's query ("diner interior photo").
- If nothing usable comes back, try different words once, then write the clip without a reference
  for that subject and say so in your summary.
- In your final summary, list each downloaded file you used and the subject it is for.

## Looking at references

Use `read_image` on every image reference, listed or downloaded, before you write a
`subject_definitions` line for it. The model copies what is in the picture, not what the script
says, so write from what you see:

- A person: apparent age, build, hair, and the clothing and accessories they wear in the picture.
  If that clothing does not suit the scene (a red-carpet gown for a diner), keep the face and
  build and dress them as the script needs: say in the definition that the picture supplies the
  face only, and mark them `partially_preserved` in `retention_analysis`.
- A place: what is in it, the layout, the light and its color, and whether people are in it. Say
  that it supplies the setting only, and whether the scene's extras fill it or it stays empty.
- Anything in the picture the clip must not copy: its background, its lighting, text, a frame.

Reject a downloaded image that is a painting, drawing, poster or screenshot when the scene is live
action; that shows several people when you need one; that shows the wrong person or place; or
that is dominated by text or a watermark. Never reject a listed file, but tell the person in your
summary when it does not match the script (wrong era, wrong clothing, more than one person).

If `read_image` is refused because the model cannot take images, carry on without it: describe a
downloaded subject only from its file title and the script, invent no details such as clothing,
and say in your summary that the references were not checked.

## Splitting into clips

- 5–14.4 s per clip. Cut at natural beats: a change of action, a new exchange, a cut in the script.
- The first clip of a scene usually establishes the setting and characters (e.g. the band playing).
- Put each major physical action (a throw, a slap sequence) where the clip can end right after it.
- Characters keep the same position across clips unless the script moves them, and a move happens
  in the clip where the script puts it, not earlier.
- Fixed camera ("The camera is fixed and does not move.") in every shot unless the script asks
  for movement.
