/**
 * Reference image search for the scene writer: finds photos on Wikimedia
 * Commons, downloads them into the run's reference directory, and records
 * where each came from (page, license, author) in `sources.json` beside them
 * so the person reviewing the draft can check and credit them.
 */

import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'

const COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
// Wikimedia asks API clients to identify themselves.
const USER_AGENT = 'FastVideoStudio-SceneWriter/0.1 (reference image search)'
const DOWNLOAD_WIDTH = 1024
const MIN_SIDE = 400
const TIMEOUT_MS = 20_000

export interface FetchedImage {
  path: string
  subject: string
  query: string
  title: string
  page: string
  license: string
  artist: string
  width: number
  height: number
}

interface CommonsPage {
  index: number
  title: string
  imageinfo?: {
    mime: string
    width: number
    height: number
    thumburl?: string
    thumbwidth?: number
    thumbheight?: number
    descriptionurl: string
    extmetadata?: Record<string, { value?: string } | undefined>
  }[]
}

function plainText(html: string | undefined): string {
  return (html ?? '').replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()
}

export function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40) || 'reference'
}

/** Commons photos (JPEG/PNG, at least MIN_SIDE px on the short side) for a search, best match first. */
export async function searchCommons(query: string, limit: number): Promise<CommonsPage[]> {
  const params = new URLSearchParams({
    action: 'query',
    format: 'json',
    generator: 'search',
    gsrsearch: `${query} filetype:bitmap`,
    gsrnamespace: '6',
    gsrlimit: '20',
    prop: 'imageinfo',
    iiprop: 'url|size|mime|extmetadata',
    iiurlwidth: String(DOWNLOAD_WIDTH),
  })
  const response = await fetch(`${COMMONS_API}?${params}`, {
    headers: { 'user-agent': USER_AGENT },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (!response.ok) throw new Error(`Wikimedia Commons search failed: HTTP ${response.status}`)
  const body = await response.json() as { query?: { pages?: Record<string, CommonsPage> } }
  return Object.values(body.query?.pages ?? {})
    .filter((p) => {
      const info = p.imageinfo?.[0]
      return info !== undefined && ['image/jpeg', 'image/png'].includes(info.mime)
        && Math.min(info.width, info.height) >= MIN_SIDE && info.thumburl !== undefined
    })
    .sort((a, b) => a.index - b.index)
    .slice(0, limit)
}

/**
 * Search, download up to `count` candidates into `refsDir`, and append them to
 * `refsDir/sources.json`. Returns the downloaded images.
 */
export async function fetchReferenceImages(
  refsDir: string,
  subject: string,
  query: string,
  count: number,
): Promise<FetchedImage[]> {
  const pages = await searchCommons(query, count)
  const fetched: FetchedImage[] = []
  const sourcesFile = path.join(refsDir, 'sources.json')
  let sources: FetchedImage[] = []
  try {
    sources = JSON.parse(await readFile(sourcesFile, 'utf8')) as FetchedImage[]
  } catch (error) {
    // ENOENT: the first image of this run; anything else would surface on the write below.
    void error
  }
  for (const page of pages) {
    const info = page.imageinfo![0]
    const response = await fetch(info.thumburl!, {
      headers: { 'user-agent': USER_AGENT },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
    if (!response.ok) continue
    const ext = path.extname(new URL(info.thumburl!).pathname).toLowerCase() === '.png' ? '.png' : '.jpg'
    let file = ''
    for (let n = 1; ; n++) {
      file = path.join(refsDir, `${slug(subject)}_${n}${ext}`)
      if (!sources.some((s) => s.path === file) && !fetched.some((s) => s.path === file)) break
    }
    await writeFile(file, Buffer.from(await response.arrayBuffer()))
    const meta = info.extmetadata ?? {}
    fetched.push({
      path: file,
      subject,
      query,
      title: page.title.replace(/^File:/, ''),
      page: info.descriptionurl,
      license: plainText(meta.LicenseShortName?.value) || 'unknown license',
      artist: plainText(meta.Artist?.value) || 'unknown author',
      width: info.thumbwidth ?? info.width,
      height: info.thumbheight ?? info.height,
    })
  }
  await writeFile(sourcesFile, `${JSON.stringify([...sources, ...fetched], null, 2)}\n`)
  return fetched
}
