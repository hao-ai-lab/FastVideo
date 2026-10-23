/**
 * DSH plugin for FastVideo Studio's scene writer. `h3_write_job` validates a
 * drafted MiniMax-H3 job against the prompt format and writes it into the
 * run's output directory; the agent learns the format from the
 * `h3-job-writer` skill, and this tool is the hard check. When the run has a
 * reference directory, `find_reference_image` also lets the agent download
 * photos from Wikimedia Commons into it for subjects nobody gave a reference
 * for; those files then count as allowed references.
 */

import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import Schema from '@deepseek-ai/schemastery'

import { fetchReferenceImages } from './reference-images.ts'
import { validateJob } from './validate-job.ts'

export const name = 'h3-scene-writer'
export const inject = ['tools']

export interface Config {
  /** Directory the validated job JSONs are written to; set per run. */
  outDir: string
  /** Absolute paths of the reference files a job may use. */
  references: string[]
  /** The video references among them that carry a soundtrack (each takes an <Audio N> label too). */
  soundVideos: string[]
  /** Where `find_reference_image` saves downloads; empty leaves the tool out. */
  refsDir: string
  /** When set (revising one clip), every written job must keep this name. */
  expectedName: string
}

export const Config: Schema<Config> = Schema.object({
  outDir: Schema.string().default(''),
  references: Schema.array(Schema.string()).default([]),
  soundVideos: Schema.array(Schema.string()).default([]),
  refsDir: Schema.string().default(''),
  expectedName: Schema.string().default(''),
})

export function apply(ctx: Context, config: Config) {
  const allowed = new Set(config.references)

  ctx.tools.register(defineTool({
    name: 'h3_write_job',
    description:
      'Validate one MiniMax-H3 job (the full job JSON as a string) and, if it passes, write it as <file_name>.json in the output directory. '
      + 'Returns the list of problems when it does not pass; fix them and call again.',
    parameters: {
      file_name: { type: 'string', required: true, description: 'File stem, e.g. "whiplash_rehearsal_clip_01"' },
      job_json: { type: 'string', required: true, description: 'The complete job object as JSON text' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      if (!config.outDir) {
        return 'Error: no output directory is configured for this run; nothing can be written.'
      }
      if (!/^[A-Za-z0-9_-]{1,100}$/.test(args.file_name)) {
        return 'Error: file_name must be letters, digits, "_" or "-" only.'
      }
      let job: unknown
      try {
        job = JSON.parse(args.job_json)
      } catch (error) {
        return `Error: job_json is not valid JSON: ${(error as Error).message}`
      }
      const problems = validateJob(job, allowed, new Set(config.soundVideos))
      if (config.expectedName && (job as { name?: unknown }).name !== config.expectedName) {
        problems.push(`"name" must stay "${config.expectedName}": a revision is a new take of the same clip.`)
      }
      if (problems.length > 0) {
        return `Not written. Fix these problems and call h3_write_job again:\n- ${problems.join('\n- ')}`
      }
      await mkdir(config.outDir, { recursive: true })
      const file = path.join(config.outDir, `${args.file_name}.json`)
      await writeFile(file, `${JSON.stringify(job, null, 2)}\n`)
      return `Wrote ${file}.`
    },
  }))

  if (!config.refsDir) return
  ctx.tools.register(defineTool({
    name: 'find_reference_image',
    description:
      'Search Wikimedia Commons for photos of a person, place, or object that has no reference file, and download up to '
      + '`count` candidates as image references. Returns each file\'s path with its title, size, and license. '
      + 'Look at each candidate with read_image before using it (Commons results include paintings and group photos). '
      + 'Use the chosen paths in "references" with media_type "image".',
    parameters: {
      subject: { type: 'string', required: true, description: 'What the reference is for, e.g. "Robert Pattinson" or "jazz rehearsal room"' },
      query: { type: 'string', required: true, description: 'Search words, e.g. the person\'s full name, or "rehearsal room interior"' },
      count: { type: 'number', description: 'Candidates to download, 1 to 4 (default 3)' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      const count = Math.min(4, Math.max(1, Math.round(args.count ?? 3)))
      let images
      try {
        await mkdir(config.refsDir, { recursive: true })
        images = await fetchReferenceImages(config.refsDir, args.subject, args.query, count)
      } catch (error) {
        return `Error: ${(error as Error).message}`
      }
      if (images.length === 0) {
        return `No usable photos found on Wikimedia Commons for "${args.query}". Try other words (a full name, a simpler description), or write the job without a reference for this subject.`
      }
      for (const image of images) allowed.add(image.path)
      return [
        `Downloaded ${images.length} candidate(s) for ${args.subject}:`,
        ...images.map((i) => `- ${i.path} - "${i.title}", ${i.width}x${i.height}, ${i.license}`),
        'Use one per subject: prefer a clear, recent, front-facing photo of just that subject (titles mentioning "cropped" or "portrait" are often best).',
      ].join('\n')
    },
  }))
}
