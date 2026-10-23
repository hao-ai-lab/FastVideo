---
name: h3-job-reviser
description: Revise one MiniMax-H3 clip job from a person's feedback on its generated video, making the smallest prompt change that fixes the problem without breaking the clip's continuity with its neighbours.
---

# Revising a MiniMax-H3 clip from feedback

A person generated a clip, watched it, and described what went wrong. You rewrite that clip's job
so the next take fixes it. Work from the feedback, the job, what these rules say the model does,
and, when `take_frames/` exists, stills from the take itself.

The job format, the section order, the numbering rules and the **Rules learned from real
generations** are in `.dsh/skills/h3-job-writer/SKILL.md`. Read that file first; everything in it
applies here.

## Inputs in the workspace

- `current_job.json`: the job as it was generated.
- `feedback.md`: what went wrong, in the person's words.
- `neighbors/`: the clips before and after it in the scene, when there are any. Positions,
  wardrobe, who is in the room and what the music is doing must stay consistent with them.
- `references.md`: the reference files this job may use.
- `take_frames/` (when the take's video was available): stills from the rendered take, about two
  seconds apart, named by their time in the clip (`t03.5s.jpg` is 3.5 s in).

## Workflow

1. Read the writer skill, then the job, the feedback, and the neighbours. Look at every still in
   `take_frames/` and at the clip's image references with `read_image`. Use the stills to find
   where in the clip the problem starts and what the model actually rendered (an extra person,
   the wrong seat, a reference's background copied in, the wrong wardrobe). Stills are sparse and
   silent: for counts of fast repeated actions, timing within two seconds, and anything heard,
   trust the feedback over the stills. If the stills contradict the feedback, say so in your
   explanation and fix what the feedback describes.
2. Diagnose before editing. Name the cause in the prompt, not just the symptom (table below).
3. Make the smallest change that removes the cause. Keep everything that the feedback does not
   mention: wording that worked is worth keeping, and every change can create a new problem.
4. Re-read the whole prompt for text the change made stale (rule 9 of the writer skill): the
   summary, the `appears in` lists, the soundscape, and any sentence that still describes the old
   behaviour. Fix those too.
5. Write the job with `h3_write_job`, keeping its `name` and using the `file_name` you were given.
   Fix anything it reports.
6. End with a short explanation for the person: the cause you found, each change you made, and
   anything they should watch for in the next take. If the feedback asks for something the
   format cannot promise, say so.

## Common symptoms and their usual causes

| Feedback | Usual cause | Fix |
| --- | --- | --- |
| An action happens too many times ("3 slaps instead of 2") | The prompt describes it as a repeated pattern ("on each four"), so the model keeps the rhythm going; or the clip has time left after the last scripted beat | Describe each occurrence once, tied to a word, with no "each"/"every"; remove counts like "exactly two"; end the clip shortly after the last one (shorten `num_frames`) |
| A pause or stall at a line break | Speech split across several `<d>` tags with prose between them | Merge the run of speech into one `<d>` and put the actions after it |
| Speech too slow or too fast | Ellipses, spelled-out pauses, or no tempo stated | Dashes for fast speech, no ellipses; state the tempo |
| A character appears twice, or in the wrong place | The shot does not say where they are | Pin every character's position in every shot; add a whole-clip rule to the summary ("there is only one drummer") |
| Someone moves too early | Their position is not pinned in the earlier shots, or the summary implies the move | Pin them in place in each shot before the move; move them only in the shot where the script does |
| Background people vanish | A reference photo of an empty room, and the shot does not mention them | Name the background people in every shot, and say the photo shows the room before they arrived |
| Music restarts or keeps playing | An audio reference plays from its start in every clip that uses it; the stop is not stated everywhere | Drop the music reference from clips after it ends; state when it stops in the summary, the shot and the soundscape |
| Delivery is louder, angrier or more emphatic than wanted | Added direction ("roars", "shouts", all caps) | Remove it; keep the script's wording |
| Something unwanted fills the end of the clip | The clip is longer than its scripted content | Shorten `num_frames` to end soon after the last beat |

When the feedback describes something new, reason the same way: what in the prompt (or missing
from it) would make the model do that?

## Do not

- Do not change the clip's `name`, its references' order (labels like `<Picture 2>` depend on
  it), the resolution, or the seed, unless the feedback is about them.
- Do not rewrite sections the feedback does not touch "while you are there".
- Do not add a fix for a problem nobody reported.
