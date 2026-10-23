/**
 * Structural checks for a drafted MiniMax-H3 job. Each check is something a
 * model draft has actually gotten wrong, or something that silently breaks a
 * generation: a missing reference, a tag pointing past the reference list,
 * appearance lists naming shots that don't exist, cuts after the clip ends.
 */

import { existsSync } from 'node:fs'

const SECTIONS = [
  'subject_definitions:',
  'summary:',
  'retention_analysis:',
  'detailed_description:',
  'overall_soundscape:',
  'non_diegetic_music:',
] as const

export const MAX_FRAMES = 345
export const FPS = 24

// Ref2VA limits enforced by FastVideo's MiniMax-H3 pipeline (reference.py).
const MAX_REFERENCES = { image: 9, video: 3, audio: 3 } as const
const MAX_TOTAL_REFERENCES = 12

const REQUIRED_FIELDS: Record<string, string> = {
  model_id: 'string',
  name: 'string',
  workload_type: 'string',
  num_frames: 'number',
  fps: 'number',
  height: 'number',
  width: 'number',
  num_inference_steps: 'number',
  seed: 'number',
  num_gpus: 'number',
}

interface Reference {
  source: string
  media_type: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function highestTag(text: string, kind: string): number {
  let max = 0
  for (const m of text.matchAll(new RegExp(`<${kind} (\\d+)>`, 'g'))) max = Math.max(max, Number(m[1]))
  return max
}

function usedTags(text: string, kind: string): Set<number> {
  return new Set([...text.matchAll(new RegExp(`<${kind} (\\d+)>`, 'g'))].map((m) => Number(m[1])))
}

function seconds(stamp: string): number {
  const [mm, ss] = stamp.split(':')
  return Number(mm) * 60 + Number(ss)
}

/**
 * Every problem found, as model-facing sentences; empty when the job passes.
 * `allowedReferences` lists the files a job may use (`null` allows any file that exists).
 * `soundVideos` are the video references that carry a soundtrack: H3 gives
 * each of them the next `<Audio N>` label, ahead of its `<Video N>`.
 */
export function validateJob(
  job: unknown,
  allowedReferences: ReadonlySet<string> | null,
  soundVideos: ReadonlySet<string> = new Set(),
): string[] {
  const problems: string[] = []
  if (!isRecord(job)) return ['The job must be a JSON object.']

  for (const [field, type] of Object.entries(REQUIRED_FIELDS)) {
    if (typeof job[field] !== type) problems.push(`"${field}" must be a ${type}.`)
  }
  if (job.model_id !== 'MiniMaxAI/MiniMax-H3') problems.push('"model_id" must be "MiniMaxAI/MiniMax-H3".')
  if (typeof job.name === 'string' && !/-clip-\d{2}$/.test(job.name)) {
    problems.push('"name" must end in "-clip-NN" (e.g. "whiplash-rehearsal-clip-01") so Studio groups the clips into a scene.')
  }
  if (job.fps !== FPS) problems.push(`"fps" must be ${FPS}.`)
  const frames = typeof job.num_frames === 'number' ? job.num_frames : 0
  if (!Number.isInteger(frames) || frames < 1 || frames > MAX_FRAMES) {
    problems.push(`"num_frames" must be a whole number from 1 to ${MAX_FRAMES} (${MAX_FRAMES / FPS} s at ${FPS} fps).`)
  }
  const duration = frames / FPS

  // References
  const refs: Reference[] = []
  if (!Array.isArray(job.references) || job.references.length === 0) {
    problems.push('"references" must be a non-empty list of {"source", "media_type"}.')
  } else {
    for (const [i, ref] of job.references.entries()) {
      if (!isRecord(ref) || typeof ref.source !== 'string' || typeof ref.media_type !== 'string') {
        problems.push(`references[${i}] must be {"source": string, "media_type": string}.`)
        continue
      }
      if (!['image', 'audio', 'video'].includes(ref.media_type)) {
        problems.push(`references[${i}].media_type must be "image", "audio" or "video".`)
      }
      if (allowedReferences !== null && !allowedReferences.has(ref.source)) {
        problems.push(`references[${i}].source "${ref.source}" is not one of the reference files you were given.`)
      } else if (!existsSync(ref.source)) {
        problems.push(`references[${i}].source "${ref.source}" does not exist.`)
      }
      refs.push({ source: ref.source, media_type: ref.media_type })
    }
    for (const [type, limit] of Object.entries(MAX_REFERENCES)) {
      const n = refs.filter((r) => r.media_type === type).length
      if (n > limit) problems.push(`H3 accepts at most ${limit} ${type} references; this job has ${n}.`)
    }
    if (refs.length > MAX_TOTAL_REFERENCES) {
      problems.push(`H3 accepts at most ${MAX_TOTAL_REFERENCES} references in total; this job has ${refs.length}.`)
    }
    if (refs.length > 0 && refs.every((r) => r.media_type === 'audio')) {
      problems.push('Audio references must be paired with at least one image or video reference.')
    }
  }

  // Prompt
  if (!Array.isArray(job.prompt) || !job.prompt.every((line) => typeof line === 'string')) {
    problems.push('"prompt" must be a list of strings (one line each).')
    return problems
  }
  const lines = job.prompt as string[]
  const text = lines.join('\n')

  const positions = SECTIONS.map((s) => lines.indexOf(s))
  SECTIONS.forEach((s, i) => {
    if (positions[i] === -1) problems.push(`The prompt is missing the "${s}" section header line.`)
  })
  if (positions.every((p) => p !== -1) && positions.some((p, i) => i > 0 && p < positions[i - 1])) {
    problems.push(`The prompt sections must appear in this order: ${SECTIONS.join(', ')}.`)
  }

  // Tags must match the reference list, numbered the way the pipeline labels
  // it: one counter per kind in list order, and a video with a soundtrack
  // takes the next <Audio N> as well as its <Video N>.
  const counts = { Picture: 0, Audio: 0, Video: 0 }
  const soundtrackLabels: string[] = []
  let audioFiles = 0
  for (const ref of refs) {
    if (ref.media_type === 'image') counts.Picture++
    else if (ref.media_type === 'audio') {
      counts.Audio++
      audioFiles++
    } else if (ref.media_type === 'video') {
      counts.Video++
      if (soundVideos.has(ref.source)) {
        counts.Audio++
        soundtrackLabels.push(`<Audio ${counts.Audio}> (the soundtrack of <Video ${counts.Video}>)`)
      }
    }
  }
  const soundtrackNumbers = new Set(soundtrackLabels.map((l) => Number(l.match(/^<Audio (\d+)>/)![1])))
  for (const kind of ['Picture', 'Audio', 'Video'] as const) {
    const highest = highestTag(text, kind)
    if (highest > counts[kind]) {
      problems.push(`The prompt mentions <${kind} ${highest}> but there are only ${counts[kind]} ${kind.toLowerCase()} label(s).`)
    }
    const used = usedTags(text, kind)
    for (let k = 1; k <= counts[kind]; k++) {
      if (kind === 'Audio' && soundtrackNumbers.has(k)) continue
      if (!used.has(k)) problems.push(`<${kind} ${k}> is in "references" but never described in the prompt; remove it or describe it.`)
    }
  }
  if (soundtrackLabels.length > 0 && problems.some((p) => p.includes('<Audio'))) {
    problems.push(`Audio labels in this job: ${soundtrackLabels.join(', ')} come from reference videos with sound, so audio files are numbered after them in list order.`)
  }

  // Summary tag. A job whose only audio is a video's soundtrack may use either.
  const summary = positions[1] !== -1 ? lines[positions[1] + 1] ?? '' : ''
  const hasAudioTag = summary.includes('+ audio reference]')
  if (summary && audioFiles > 0 && !hasAudioTag) {
    problems.push('The summary tag must include "+ audio reference]" because the job has audio references.')
  } else if (summary && counts.Audio === 0 && hasAudioTag) {
    problems.push('The summary tag must not include "audio reference" because the job has no audio references.')
  }

  // Shots.
  const shotLines = lines.filter((l) => /^\[Shot \d+\]/.test(l))
  if (shotLines.length === 0) problems.push('"detailed_description" needs at least one "[Shot 1] ..." line.')
  shotLines.forEach((line, i) => {
    const n = Number(line.match(/^\[Shot (\d+)\]/)![1])
    if (n !== i + 1) problems.push(`Shots must be numbered 1, 2, 3... in order; found [Shot ${n}] in position ${i + 1}.`)
  })
  let lastCut = 0
  for (const line of shotLines.slice(1)) {
    const cut = line.match(/^\[Shot (\d+)\] At (\d{2}:\d{2}\.\d{3}), the shot cuts to/)
    if (!cut) {
      problems.push(`[Shot ${line.match(/^\[Shot (\d+)\]/)![1]}] must start with "At MM:SS.mmm, the shot cuts to ...".`)
      continue
    }
    const t = seconds(cut[2])
    if (t <= lastCut) problems.push(`[Shot ${cut[1]}] cuts at ${cut[2]}, which is not after the previous cut.`)
    if (t >= duration) problems.push(`[Shot ${cut[1]}] cuts at ${cut[2]}, but the clip is only ${duration.toFixed(2)} s long (${frames} frames).`)
    lastCut = t
  }

  // Appearance lists must name existing shots.
  for (const line of lines) {
    const m = line.match(/^<Subject (\d+)> \(appears in ([^)]*)\)/)
    if (!m) continue
    for (const ref of m[2].matchAll(/\[Shot (\d+)\]/g)) {
      if (Number(ref[1]) > shotLines.length) {
        problems.push(`<Subject ${m[1]}> is listed as appearing in [Shot ${ref[1]}], but there are only ${shotLines.length} shot(s).`)
      }
    }
  }

  // Dialogue tags.
  const opens = (text.match(/<d>/g) ?? []).length
  const closes = (text.match(/<\/d>/g) ?? []).length
  if (opens !== closes) problems.push(`Unbalanced dialogue tags: ${opens} "<d>" vs ${closes} "</d>".`)
  for (const m of text.matchAll(/<d>(?!\[English\] )/g)) {
    problems.push(`Every dialogue tag must start with "<d>[English] " (found one at character ${m.index}).`)
    break
  }

  return problems
}
