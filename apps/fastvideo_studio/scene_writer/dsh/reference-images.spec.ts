import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { fetchReferenceImages, slug } from './reference-images.ts'

function page(index: number, title: string, mime: string, width: number, height: number) {
  return {
    index,
    title: `File:${title}`,
    imageinfo: [{
      mime,
      width,
      height,
      thumburl: `https://upload.example/thumb/${encodeURIComponent(title)}`.replace(/\.png$/, '.png'),
      thumbwidth: 1024,
      thumbheight: Math.round((1024 * height) / width),
      descriptionurl: `https://commons.wikimedia.org/wiki/File:${encodeURIComponent(title)}`,
      extmetadata: {
        LicenseShortName: { value: 'CC BY-SA 4.0' },
        Artist: { value: '<a href="//commons.wikimedia.org/wiki/User:Someone">Someone</a>' },
      },
    }],
  }
}

function stubCommons(pages: unknown[]) {
  const fetchMock = vi.fn(async (url: string) => {
    if (url.startsWith('https://commons.wikimedia.org/w/api.php')) {
      return new Response(JSON.stringify({ query: { pages: Object.fromEntries(pages.map((p, i) => [String(i), p])) } }))
    }
    return new Response(new Uint8Array([1, 2, 3]))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('fetchReferenceImages', () => {
  it('downloads the best photo matches and records their sources', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'h3-refs-'))
    stubCommons([
      page(2, 'Second.jpg', 'image/jpeg', 2000, 3000),
      page(1, 'First portrait (cropped).jpg', 'image/jpeg', 1200, 1600),
      page(3, 'Tiny.jpg', 'image/jpeg', 300, 200),
      page(4, 'Drawing.svg', 'image/svg+xml', 2000, 2000),
    ])

    const images = await fetchReferenceImages(dir, 'Robert Pattinson', 'Robert Pattinson', 3)

    // Search order kept; too-small and non-photo files skipped.
    expect(images.map((i) => i.title)).toEqual(['First portrait (cropped).jpg', 'Second.jpg'])
    expect(images.map((i) => path.basename(i.path))).toEqual(['robert_pattinson_1.jpg', 'robert_pattinson_2.jpg'])
    expect(images.every((i) => existsSync(i.path))).toBe(true)
    expect(images[0]).toMatchObject({ license: 'CC BY-SA 4.0', artist: 'Someone', width: 1024, height: 1365 })
    const sources = JSON.parse(readFileSync(path.join(dir, 'sources.json'), 'utf8'))
    expect(sources).toEqual(images)
  })

  it('numbers a second search for the same subject after the first', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'h3-refs-'))
    stubCommons([page(1, 'A.jpg', 'image/jpeg', 800, 800)])
    await fetchReferenceImages(dir, 'Robert Pattinson', 'Robert Pattinson', 1)
    const again = await fetchReferenceImages(dir, 'Robert Pattinson', 'Robert Pattinson actor', 1)

    expect(path.basename(again[0].path)).toBe('robert_pattinson_2.jpg')
    expect(JSON.parse(readFileSync(path.join(dir, 'sources.json'), 'utf8'))).toHaveLength(2)
  })

  it('reports a failed search', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 503 })))
    await expect(fetchReferenceImages(mkdtempSync(path.join(tmpdir(), 'h3-refs-')), 'x', 'x', 1))
      .rejects.toThrow(/HTTP 503/)
  })
})

describe('slug', () => {
  it('makes a safe file stem', () => {
    expect(slug('Fletcher\'s rehearsal room!')).toBe('fletcher_s_rehearsal_room')
    expect(slug('***')).toBe('reference')
  })
})
