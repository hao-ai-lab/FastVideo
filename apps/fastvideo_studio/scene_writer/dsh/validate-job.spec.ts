import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { validateJob } from './validate-job.ts'

const dir = mkdtempSync(path.join(tmpdir(), 'h3-validate-'))
const picture = path.join(dir, 'a.png')
const voice = path.join(dir, 'a.mp3')
const clip = path.join(dir, 'tail.mp4')
writeFileSync(picture, '')
writeFileSync(voice, '')
writeFileSync(clip, '')

function goodJob() {
  return {
    model_id: 'MiniMaxAI/MiniMax-H3',
    name: 'scene-clip-01',
    workload_type: 'i2v',
    prompt: [
      'subject_definitions:',
      '<Subject 1> is Ada from <Picture 1>.',
      '<Audio 1> is the voice-timbre reference for <Subject 1> (S1), containing a spoken English vocal layer.',
      '',
      'summary:',
      '[reference generation + audio reference] Ada talks.',
      '',
      'retention_analysis:',
      '<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her face.',
      '',
      'detailed_description:',
      '[Shot 1] Medium shot. (S1) <d>[English] Hello.</d>',
      '[Shot 2] At 00:03.000, the shot cuts to a close-up shot. (S1) <d>[English] Bye.</d> <cutoff>',
      '',
      'overall_soundscape:',
      'Room tone.',
      '',
      'non_diegetic_music:',
      'N/A',
    ],
    references: [
      { source: picture, media_type: 'image' },
      { source: voice, media_type: 'audio' },
    ],
    negative_prompt: '',
    guidance_scale: 1,
    num_frames: 144,
    fps: 24,
    height: 768,
    width: 1344,
    num_inference_steps: 50,
    seed: 0,
    num_gpus: 4,
    use_fsdp_inference: true,
  }
}

const allowed = new Set([picture, voice, clip])

describe('validateJob', () => {
  it('passes a well-formed job', () => {
    expect(validateJob(goodJob(), allowed)).toEqual([])
  })

  it('rejects clips longer than one generation', () => {
    expect(validateJob({ ...goodJob(), num_frames: 400 }, allowed).join()).toMatch(/num_frames/)
  })

  it('rejects a tag that points past the reference list', () => {
    const job = goodJob()
    job.prompt[1] = '<Subject 1> is Ada from <Picture 2>.'
    expect(validateJob(job, allowed).join()).toMatch(/<Picture 2> but there are only 1/)
  })

  it('rejects references that were not provided', () => {
    const job = goodJob()
    job.references[0] = { source: '/elsewhere/b.png', media_type: 'image' }
    expect(validateJob(job, allowed).join()).toMatch(/not one of the reference files/)
  })

  it('rejects a missing section and cuts past the end of the clip', () => {
    const job = goodJob()
    job.prompt = job.prompt.filter((l) => l !== 'overall_soundscape:')
    job.num_frames = 48
    const problems = validateJob(job, allowed).join('\n')
    expect(problems).toMatch(/missing the "overall_soundscape:"/)
    expect(problems).toMatch(/cuts at 00:03.000, but the clip is only 2.00 s/)
  })

  it('rejects appearance lists naming shots that do not exist', () => {
    const job = goodJob()
    job.prompt[8] = '<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - her face.'
    expect(validateJob(job, allowed).join()).toMatch(/\[Shot 3\], but there are only 2/)
  })

  it('requires the audio summary tag to match the references', () => {
    const job = goodJob()
    job.prompt[5] = '[reference generation] Ada talks.'
    expect(validateJob(job, allowed).join()).toMatch(/must include "\+ audio reference\]"/)
  })

  it('numbers a sound video\'s soundtrack before later audio files', () => {
    const job = goodJob()
    job.references = [{ source: clip, media_type: 'video' }, ...job.references]
    job.prompt[1] = '<Subject 1> is Ada from <Picture 1>, continuing from <Video 1>.'
    // The video's soundtrack is <Audio 1>, so the voice file is <Audio 2>.
    expect(validateJob(job, allowed, new Set([clip])).join()).toMatch(/<Audio 2> is in "references" but never described/)
    job.prompt = job.prompt.map((l) => l.replaceAll('<Audio 1>', '<Audio 2>'))
    expect(validateJob(job, allowed, new Set([clip]))).toEqual([])
    // Without a soundtrack the voice file stays <Audio 1>.
    expect(validateJob(job, allowed).join()).toMatch(/<Audio 2> but there are only 1/)
  })

  it('enforces the pipeline\'s reference limits', () => {
    const job = goodJob()
    job.references = [{ source: voice, media_type: 'audio' }]
    expect(validateJob(job, allowed).join()).toMatch(/paired with at least one image or video/)
    job.references = Array.from({ length: 10 }, () => ({ source: picture, media_type: 'image' }))
    expect(validateJob(job, allowed).join()).toMatch(/at most 9 image references/)
  })

  it('allows only listed references, and none when the list is empty', () => {
    expect(validateJob(goodJob(), new Set()).join()).toMatch(/not one of the reference files/)
    expect(validateJob(goodJob(), null)).toEqual([])
  })
})
