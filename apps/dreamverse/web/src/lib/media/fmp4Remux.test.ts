import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  Fmp4RemuxError,
  remuxArchivedFmp4Segments,
} from './fmp4Remux';

const createFileMock = vi.fn();

vi.mock('mp4box', () => ({
  createFile: (...args: any[]) => createFileMock(...args),
}));

function buildParser({
  tracks,
  samplesByTrack,
  trackType = 'avc1',
  throwOnAppend = false,
}: {
  tracks: Array<Record<string, any>>;
  samplesByTrack: Record<number, any[]>;
  trackType?: string;
  throwOnAppend?: boolean;
}) {
  const parser: any = {
    onReady: null,
    onSamples: null,
    onError: null,
    setExtractionOptions: vi.fn(),
    start: vi.fn(),
    flush: vi.fn(),
    getTrackById: vi.fn((trackId: number) => ({
      mdia: {
        minf: {
          stbl: {
            stsd: {
              entries: [
                {
                  type: trackType,
                  boxes: [{ type: 'avcC' }],
                },
              ],
            },
          },
        },
      },
    })),
    appendBuffer: vi.fn(() => {
      if (throwOnAppend) {
        throw new Error('append failure');
      }
      parser.onReady?.({
        timescale: 1000,
        brands: ['isom'],
        tracks,
      });
      for (const track of tracks) {
        parser.onSamples?.(
          track.id,
          null,
          samplesByTrack[track.id] || [],
        );
      }
    }),
  };
  return parser;
}

function buildWriter() {
  const recordedSamples: Array<Record<string, any>> = [];
  const writer: any = {
    init: vi.fn(),
    addTrack: vi.fn((options: Record<string, any>) => Number(options.id)),
    addSample: vi.fn(
      (trackId: number, data: Uint8Array, options: Record<string, any>) => {
        recordedSamples.push({
          trackId,
          data,
          options,
        });
      },
    ),
    getBuffer: vi.fn(() => {
      const buffer = new ArrayBuffer(64);
      return {
        buffer,
        byteLength: buffer.byteLength,
      };
    }),
    recordedSamples,
  };
  return writer;
}

function buildSample({
  dts,
  duration,
  value,
}: {
  dts: number;
  duration: number;
  value: number;
}) {
  return {
    dts,
    cts: dts,
    duration,
    description_index: 0,
    is_sync: true,
    is_leading: 0,
    depends_on: 0,
    is_depended_on: 0,
    has_redundancy: 0,
    degradation_priority: 0,
    data: new Uint8Array([value]),
  };
}

const BASE_TRACK = {
  id: 1,
  codec: 'avc1.640028',
  timescale: 1000,
  language: 'und',
  video: {
    width: 640,
    height: 360,
  },
};

describe('remuxArchivedFmp4Segments', () => {
  beforeEach(() => {
    createFileMock.mockReset();
  });

  it('remuxes multiple segments into one playable MP4 blob', async () => {
    const parserA = buildParser({
      tracks: [BASE_TRACK],
      samplesByTrack: {
        1: [
          buildSample({ dts: 0, duration: 5, value: 1 }),
          buildSample({ dts: 5, duration: 5, value: 2 }),
        ],
      },
    });
    const parserB = buildParser({
      tracks: [BASE_TRACK],
      samplesByTrack: {
        1: [
          buildSample({ dts: 0, duration: 5, value: 3 }),
          buildSample({ dts: 5, duration: 5, value: 4 }),
        ],
      },
    });
    const writer = buildWriter();

    createFileMock
      .mockImplementationOnce(() => parserA)
      .mockImplementationOnce(() => parserB)
      .mockImplementationOnce(() => writer);

    const blob = await remuxArchivedFmp4Segments([
      {
        key: 'seg-1',
        segmentIdx: 1,
        streamId: 's1',
        mime: 'video/mp4',
        completed: true,
        chunks: [new Uint8Array([1, 2, 3]).buffer],
      },
      {
        key: 'seg-2',
        segmentIdx: 2,
        streamId: 's2',
        mime: 'video/mp4',
        completed: true,
        chunks: [new Uint8Array([4, 5, 6]).buffer],
      },
    ]);

    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('video/mp4');
    expect(writer.addTrack).toHaveBeenCalledTimes(1);
    expect(writer.addSample).toHaveBeenCalledTimes(4);
    const sampleCalls = writer.recordedSamples;
    expect(sampleCalls[0]?.options?.dts).toBe(0);
    expect(sampleCalls[2]?.options?.dts).toBe(10);
  });

  it('fails with includeInProgress=true and succeeds with completed-only fallback', async () => {
    const validParser = buildParser({
      tracks: [BASE_TRACK],
      samplesByTrack: {
        1: [buildSample({ dts: 0, duration: 5, value: 7 })],
      },
    });
    const brokenParser = buildParser({
      tracks: [BASE_TRACK],
      samplesByTrack: {},
      throwOnAppend: true,
    });
    const writerForSuccess = buildWriter();

    createFileMock
      .mockImplementationOnce(() => validParser)
      .mockImplementationOnce(() => brokenParser);

    await expect(
      remuxArchivedFmp4Segments([
        {
          key: 'completed',
          segmentIdx: 1,
          streamId: 's1',
          mime: 'video/mp4',
          completed: true,
          chunks: [new Uint8Array([1]).buffer],
        },
        {
          key: 'in-progress',
          segmentIdx: 2,
          streamId: 's2',
          mime: 'video/mp4',
          completed: false,
          chunks: [new Uint8Array([2]).buffer],
        },
      ], {
        includeInProgress: true,
      }),
    ).rejects.toBeInstanceOf(Fmp4RemuxError);

    createFileMock
      .mockReset()
      .mockImplementationOnce(() => validParser)
      .mockImplementationOnce(() => writerForSuccess);

    const blob = await remuxArchivedFmp4Segments([
      {
        key: 'completed',
        segmentIdx: 1,
        streamId: 's1',
        mime: 'video/mp4',
        completed: true,
        chunks: [new Uint8Array([1]).buffer],
      },
      {
        key: 'in-progress',
        segmentIdx: 2,
        streamId: 's2',
        mime: 'video/mp4',
        completed: false,
        chunks: [new Uint8Array([2]).buffer],
      },
    ], {
      includeInProgress: false,
    });

    expect(blob).toBeInstanceOf(Blob);
    expect(writerForSuccess.addSample).toHaveBeenCalledTimes(1);
  });

  it('throws no_segments when no eligible segments exist', async () => {
    await expect(
      remuxArchivedFmp4Segments([], {
        includeInProgress: false,
      }),
    ).rejects.toMatchObject({
      code: 'no_segments',
    });
  });
});
