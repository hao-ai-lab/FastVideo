import { createFile } from 'mp4box';

import type { ArchivedAvSegment } from './avPipeline';

const DEFAULT_REMUX_MIME = 'video/mp4';

export type Fmp4RemuxErrorCode =
  | 'no_segments'
  | 'segment_parse_failed'
  | 'remux_failed';

export class Fmp4RemuxError extends Error {
  code: Fmp4RemuxErrorCode;
  cause?: unknown;

  constructor(
    code: Fmp4RemuxErrorCode,
    message: string,
    cause?: unknown,
  ) {
    super(message);
    this.name = 'Fmp4RemuxError';
    this.code = code;
    this.cause = cause;
  }
}

export interface RemuxArchivedFmp4Options {
  includeInProgress?: boolean;
  mimeType?: string;
}

interface ParsedTrack {
  sourceTrackId: number;
  type: string;
  timescale: number;
  width: number;
  height: number;
  channelCount: number;
  sampleRate: number;
  sampleSize: number;
  language: string;
  handler: string;
  descriptionBoxes: any[];
  samples: any[];
}

interface ParsedSegment {
  movieTimescale: number;
  brands: string[];
  tracks: Map<number, ParsedTrack>;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function concatArrayBuffers(chunks: ArrayBuffer[]): ArrayBuffer {
  const totalSize = chunks.reduce(
    (sum, chunk) => sum + chunk.byteLength,
    0,
  );
  const output = new Uint8Array(totalSize);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(new Uint8Array(chunk), offset);
    offset += chunk.byteLength;
  }
  return output.buffer;
}

function normalizeTrackType(codec: string, fallback = 'avc1'): string {
  const source = typeof codec === 'string' ? codec.trim() : '';
  const firstToken = source.split('.')[0] || '';
  if (firstToken.length === 4) {
    return firstToken;
  }
  return fallback;
}

function cloneSampleData(data: unknown): Uint8Array<ArrayBuffer> | null {
  if (data instanceof Uint8Array) {
    return new Uint8Array(data).slice();
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data.slice(0));
  }
  return null;
}

function computeSegmentTiming(samples: any[]): {
  segmentStartDts: number;
  segmentDuration: number;
} {
  if (!samples.length) {
    return {
      segmentStartDts: 0,
      segmentDuration: 0,
    };
  }

  let minDts = Number.POSITIVE_INFINITY;
  let maxEnd = 0;
  for (const sample of samples) {
    const dts = toFiniteNumber(sample?.dts, 0);
    const duration = Math.max(1, toFiniteNumber(sample?.duration, 1));
    minDts = Math.min(minDts, dts);
    maxEnd = Math.max(maxEnd, dts + duration);
  }

  if (!Number.isFinite(minDts)) {
    minDts = 0;
  }

  return {
    segmentStartDts: minDts,
    segmentDuration: Math.max(0, maxEnd - minDts),
  };
}

async function parseArchivedSegment(
  segment: ArchivedAvSegment,
): Promise<ParsedSegment> {
  const segmentBuffer = concatArrayBuffers(segment.chunks);
  if (segmentBuffer.byteLength === 0) {
    throw new Fmp4RemuxError(
      'segment_parse_failed',
      `Segment "${segment.key}" has no bytes.`,
    );
  }

  return new Promise<ParsedSegment>((resolve, reject) => {
    const parser: any = createFile(true);
    const samplesByTrackId = new Map<number, any[]>();
    let readyInfo: any = null;
    let settled = false;

    const fail = (error: unknown, defaultMessage: string): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (error instanceof Fmp4RemuxError) {
        reject(error);
        return;
      }
      reject(
        new Fmp4RemuxError(
          'segment_parse_failed',
          defaultMessage,
          error,
        ),
      );
    };

    const finish = (): void => {
      if (settled) {
        return;
      }

      if (!readyInfo || !Array.isArray(readyInfo?.tracks)) {
        fail(
          null,
          `Unable to parse MP4 metadata for segment "${segment.key}".`,
        );
        return;
      }

      const tracks = new Map<number, ParsedTrack>();
      for (const trackInfo of readyInfo.tracks as any[]) {
        const sourceTrackId = Number(trackInfo?.id);
        if (!Number.isInteger(sourceTrackId)) {
          continue;
        }

        const sourceTrack = parser.getTrackById(sourceTrackId);
        const sampleDescription = sourceTrack?.mdia?.minf?.stbl?.stsd?.entries?.[0];
        const descriptionBoxes = Array.isArray(sampleDescription?.boxes)
          ? sampleDescription.boxes
          : [];

        tracks.set(sourceTrackId, {
          sourceTrackId,
          type: normalizeTrackType(
            String(sampleDescription?.type || trackInfo?.codec || ''),
            trackInfo?.audio ? 'mp4a' : 'avc1',
          ),
          timescale: Math.max(1, toFiniteNumber(trackInfo?.timescale, 1)),
          width: Math.max(0, toFiniteNumber(trackInfo?.video?.width, 0)),
          height: Math.max(0, toFiniteNumber(trackInfo?.video?.height, 0)),
          channelCount: Math.max(
            0,
            toFiniteNumber(trackInfo?.audio?.channel_count, 0),
          ),
          sampleRate: Math.max(
            0,
            toFiniteNumber(trackInfo?.audio?.sample_rate, 0),
          ),
          sampleSize: Math.max(
            0,
            toFiniteNumber(trackInfo?.audio?.sample_size, 0),
          ),
          language:
            typeof trackInfo?.language === 'string'
            && trackInfo.language.trim()
              ? trackInfo.language.trim()
              : 'und',
          handler: trackInfo?.audio ? 'soun' : 'vide',
          descriptionBoxes,
          samples: samplesByTrackId.get(sourceTrackId) || [],
        });
      }

      settled = true;
      resolve({
        movieTimescale: Math.max(1, toFiniteNumber(readyInfo?.timescale, 600)),
        brands: Array.isArray(readyInfo?.brands)
          ? readyInfo.brands.filter(
            (brand: unknown) =>
              typeof brand === 'string' && brand.trim(),
          )
          : [],
        tracks,
      });
    };

    parser.onError = (module: string, message: string) => {
      fail(
        null,
        `MP4 parse error in segment "${segment.key}" (${module}): ${message}`,
      );
    };

    parser.onReady = (info: any) => {
      readyInfo = info;
      const tracks = Array.isArray(info?.tracks) ? info.tracks : [];
      for (const track of tracks) {
        if (!Number.isInteger(track?.id)) {
          continue;
        }
        samplesByTrackId.set(track.id, []);
        parser.setExtractionOptions(track.id, null, {
          nbSamples: Number.MAX_SAFE_INTEGER,
          rapAlignement: false,
        });
      }
      parser.start();
    };

    parser.onSamples = (
      trackId: number,
      _user: unknown,
      samples: any[],
    ) => {
      const nextSamples = samplesByTrackId.get(trackId) || [];
      for (const sample of samples || []) {
        const copiedData = cloneSampleData(sample?.data);
        if (!copiedData || copiedData.byteLength === 0) {
          continue;
        }
        nextSamples.push({
          ...sample,
          data: copiedData,
        });
      }
      samplesByTrackId.set(trackId, nextSamples);
    };

    try {
      const mp4Buffer = segmentBuffer.slice(0) as ArrayBuffer & {
        fileStart?: number;
      };
      mp4Buffer.fileStart = 0;
      parser.appendBuffer(mp4Buffer);
      parser.flush();
      Promise.resolve().then(finish);
    } catch (error) {
      fail(
        error,
        `Failed to append MP4 segment "${segment.key}" to parser.`,
      );
    }
  });
}

export async function remuxArchivedFmp4Segments(
  segments: ArchivedAvSegment[],
  {
    includeInProgress = true,
    mimeType = DEFAULT_REMUX_MIME,
  }: RemuxArchivedFmp4Options = {},
): Promise<Blob> {
  const selectedSegments = (Array.isArray(segments) ? segments : [])
    .filter(
      (segment) =>
        Array.isArray(segment?.chunks)
        && segment.chunks.length > 0
        && (includeInProgress || Boolean(segment?.completed)),
    );

  if (selectedSegments.length === 0) {
    throw new Fmp4RemuxError(
      'no_segments',
      includeInProgress
        ? 'No archived stream data is available for remuxing.'
        : 'No completed segments are available for remuxing.',
    );
  }

  const parsedSegments = await Promise.all(
    selectedSegments.map((segment) => parseArchivedSegment(segment)),
  );

  const firstWithTracks = parsedSegments.find(
    (segment) => segment.tracks.size > 0,
  );
  if (!firstWithTracks) {
    throw new Fmp4RemuxError(
      'remux_failed',
      'No tracks were found while remuxing archived segments.',
    );
  }

  const outputFile: any = createFile(false);
  outputFile.init({
    brands: firstWithTracks.brands.length > 0
      ? firstWithTracks.brands
      : ['isom'],
    timescale: firstWithTracks.movieTimescale,
  });

  const outputTrackBySourceTrack = new Map<number, number>();
  const trackOffsetBySourceTrack = new Map<number, number>();

  for (const [sourceTrackId, parsedTrack] of firstWithTracks.tracks.entries()) {
    const trackOptions: Record<string, unknown> = {
      id: sourceTrackId,
      type: parsedTrack.type,
      timescale: parsedTrack.timescale,
      hdlr: parsedTrack.handler,
      language: parsedTrack.language,
    };
    if (parsedTrack.width > 0) {
      trackOptions.width = parsedTrack.width;
    }
    if (parsedTrack.height > 0) {
      trackOptions.height = parsedTrack.height;
    }
    if (parsedTrack.channelCount > 0) {
      trackOptions.channel_count = parsedTrack.channelCount;
    }
    if (parsedTrack.sampleSize > 0) {
      trackOptions.samplesize = parsedTrack.sampleSize;
    }
    if (parsedTrack.sampleRate > 0) {
      trackOptions.samplerate = parsedTrack.sampleRate * 65536;
    }
    if (parsedTrack.descriptionBoxes.length > 0) {
      trackOptions.description_boxes = parsedTrack.descriptionBoxes;
    }

    const outputTrackId = outputFile.addTrack(trackOptions);
    if (!Number.isInteger(outputTrackId)) {
      throw new Fmp4RemuxError(
        'remux_failed',
        `Unable to create output track for source track ${sourceTrackId}.`,
      );
    }
    outputTrackBySourceTrack.set(sourceTrackId, Number(outputTrackId));
    trackOffsetBySourceTrack.set(sourceTrackId, 0);
  }

  let writtenSampleCount = 0;

  for (const parsedSegment of parsedSegments) {
    for (const [
      sourceTrackId,
      outputTrackId,
    ] of outputTrackBySourceTrack.entries()) {
      const parsedTrack = parsedSegment.tracks.get(sourceTrackId);
      if (!parsedTrack || parsedTrack.samples.length === 0) {
        continue;
      }

      const { segmentStartDts, segmentDuration } = computeSegmentTiming(
        parsedTrack.samples,
      );
      const trackOffset = trackOffsetBySourceTrack.get(sourceTrackId) || 0;

      for (const sample of parsedTrack.samples) {
        const sampleData = cloneSampleData(sample?.data);
        if (!sampleData || sampleData.byteLength === 0) {
          continue;
        }

        const sampleDts = toFiniteNumber(sample?.dts, 0);
        const sampleCts = toFiniteNumber(sample?.cts, sampleDts);
        const sampleDuration = Math.max(
          1,
          Math.round(toFiniteNumber(sample?.duration, 1)),
        );

        outputFile.addSample(outputTrackId, sampleData, {
          sample_description_index: Math.max(
            1,
            Math.round(toFiniteNumber(sample?.description_index, 0)) + 1,
          ),
          duration: sampleDuration,
          dts: Math.round(sampleDts - segmentStartDts + trackOffset),
          cts: Math.round(sampleCts - segmentStartDts + trackOffset),
          is_sync: Boolean(sample?.is_sync),
          is_leading: Math.round(toFiniteNumber(sample?.is_leading, 0)),
          depends_on: Math.round(toFiniteNumber(sample?.depends_on, 0)),
          is_depended_on: Math.round(
            toFiniteNumber(sample?.is_depended_on, 0),
          ),
          has_redundancy: Math.round(
            toFiniteNumber(sample?.has_redundancy, 0),
          ),
          degradation_priority: Math.round(
            toFiniteNumber(sample?.degradation_priority, 0),
          ),
        });
        writtenSampleCount += 1;
      }

      trackOffsetBySourceTrack.set(
        sourceTrackId,
        trackOffset + segmentDuration,
      );
    }
  }

  if (writtenSampleCount === 0) {
    throw new Fmp4RemuxError(
      'remux_failed',
      'No media samples were available for remux output.',
    );
  }

  try {
    const outputStream = outputFile.getBuffer();
    const outputBytes = outputStream.buffer.slice(0, outputStream.byteLength);
    return new Blob([outputBytes], { type: mimeType || DEFAULT_REMUX_MIME });
  } catch (error) {
    throw new Fmp4RemuxError(
      'remux_failed',
      'Failed to build remuxed MP4 buffer.',
      error,
    );
  }
}
