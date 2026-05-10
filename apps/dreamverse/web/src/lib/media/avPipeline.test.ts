import { describe, expect, it, vi } from 'vitest';

import { createAvPipeline } from './avPipeline';

type MockMediaSourceCtor = {
  new (): any;
  isTypeSupported: (mime: string) => boolean;
};

type MockMediaSourceReadyState = 'closed' | 'open' | 'ended';

function createBufferedRange(start: number, end: number): any {
  return {
    length: 1,
    start(index: number): number {
      if (index !== 0) {
        throw new RangeError('invalid buffered range index');
      }
      return start;
    },
    end(index: number): number {
      if (index !== 0) {
        throw new RangeError('invalid buffered range index');
      }
      return end;
    },
  };
}

function createFakeVideo(): any {
  const listeners = new Map<string, Array<() => void>>();
  const video: any = {
    currentTime: 0,
    buffered: createBufferedRange(0, 0),
    paused: true,
    disableRemotePlayback: false,
    addEventListener(type: string, listener: () => void): void {
      const next = listeners.get(type) || [];
      next.push(listener);
      listeners.set(type, next);
    },
    removeEventListener(type: string, listener: () => void): void {
      const next = (listeners.get(type) || []).filter((item) => item !== listener);
      listeners.set(type, next);
    },
    dispatch(type: string): void {
      for (const listener of listeners.get(type) || []) {
        listener();
      }
    },
    play: vi.fn(() => {
      video.paused = false;
      return Promise.resolve();
    }),
    pause: vi.fn(() => {
      video.paused = true;
    }),
    load: vi.fn(() => {}),
    removeAttribute: vi.fn(() => {}),
  };

  return video;
}

const fakeVideo = createFakeVideo();

function setAppleMobileNavigator(): void {
  vi.spyOn(window.navigator, 'userAgent', 'get').mockReturnValue(
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)',
  );
  vi.spyOn(window.navigator, 'platform', 'get').mockReturnValue('iPhone');
  vi.spyOn(window.navigator, 'maxTouchPoints', 'get').mockReturnValue(5);
}

function setNonAppleNavigator(): void {
  vi.spyOn(window.navigator, 'userAgent', 'get').mockReturnValue(
    'Mozilla/5.0 (X11; Linux x86_64)',
  );
  vi.spyOn(window.navigator, 'platform', 'get').mockReturnValue('Linux x86_64');
  vi.spyOn(window.navigator, 'maxTouchPoints', 'get').mockReturnValue(0);
}

function setAndroidChromeNavigator(): void {
  vi.spyOn(window.navigator, 'userAgent', 'get').mockReturnValue(
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36',
  );
  vi.spyOn(window.navigator, 'platform', 'get').mockReturnValue('Linux armv8l');
  vi.spyOn(window.navigator, 'maxTouchPoints', 'get').mockReturnValue(5);
}

function overrideMediaSourceApis({
  mediaSourceCtor,
  managedMediaSourceCtor,
}: {
  mediaSourceCtor?: MockMediaSourceCtor;
  managedMediaSourceCtor?: MockMediaSourceCtor;
}): () => void {
  const mediaSourceWindow = window as any;
  const originalMediaSource = mediaSourceWindow.MediaSource;
  const originalManagedMediaSource = mediaSourceWindow.ManagedMediaSource;

  if (mediaSourceCtor) {
    mediaSourceWindow.MediaSource = mediaSourceCtor;
  } else {
    delete mediaSourceWindow.MediaSource;
  }

  if (managedMediaSourceCtor) {
    mediaSourceWindow.ManagedMediaSource = managedMediaSourceCtor;
  } else {
    delete mediaSourceWindow.ManagedMediaSource;
  }

  return () => {
    if (typeof originalMediaSource === 'undefined') {
      delete mediaSourceWindow.MediaSource;
    } else {
      mediaSourceWindow.MediaSource = originalMediaSource;
    }

    if (typeof originalManagedMediaSource === 'undefined') {
      delete mediaSourceWindow.ManagedMediaSource;
    } else {
      mediaSourceWindow.ManagedMediaSource = originalManagedMediaSource;
    }
  };
}

describe('createAvPipeline', () => {
  it('uses native playback fallback on Apple mobile when MMS/MSE are unavailable', () => {
    setAppleMobileNavigator();
    const restoreMediaSourceApis = overrideMediaSourceApis({});

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
      });

      expect(pipeline.usesNativePlaybackFallback()).toBe(true);
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('does not use native playback fallback on Apple mobile when MMS is available', () => {
    setAppleMobileNavigator();

    class FakeManagedMediaSource {
      static isTypeSupported(): boolean {
        return true;
      }
    }

    const restoreMediaSourceApis = overrideMediaSourceApis({
      managedMediaSourceCtor: FakeManagedMediaSource as unknown as MockMediaSourceCtor,
    });

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
      });

      expect(pipeline.usesNativePlaybackFallback()).toBe(false);
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('throws when neither ManagedMediaSource nor MediaSource exists on non-Apple browsers', async () => {
    setNonAppleNavigator();
    const restoreMediaSourceApis = overrideMediaSourceApis({});

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
      });

      await expect(
        pipeline.ensurePipeline('video/mp4'),
      ).rejects.toThrow(
        'ManagedMediaSource/MediaSource APIs are not available in this browser.',
      );
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('keeps live playback enabled on Android Chrome when MediaSource is available', () => {
    setAndroidChromeNavigator();

    class FakeMediaSource {
      static isTypeSupported(): boolean {
        return true;
      }
    }

    const restoreMediaSourceApis = overrideMediaSourceApis({
      mediaSourceCtor: FakeMediaSource as unknown as MockMediaSourceCtor,
    });

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
      });

      expect(pipeline.usesNativePlaybackFallback()).toBe(false);
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('initializes playback with ManagedMediaSource when available', async () => {
    setAppleMobileNavigator();
    fakeVideo.disableRemotePlayback = false;

    class FakeSourceBuffer {
      mode: AppendMode = 'segments';

      updating = false;

      addEventListener(): void {
        // noop for test
      }

      removeEventListener(): void {
        // noop for test
      }

      appendBuffer(): void {
        // noop for test
      }
    }

    class FakeManagedMediaSource {
      static instances: FakeManagedMediaSource[] = [];

      static isTypeSupported(_mime: string): boolean {
        return true;
      }

      readyState: MockMediaSourceReadyState = 'closed';

      sourceBuffers: SourceBufferList = {} as SourceBufferList;

      activeSourceBuffers: SourceBufferList = {} as SourceBufferList;

      duration = 0;

      onsourceopen: ((this: MediaSource, ev: Event) => any) | null = null;

      onsourceended: ((this: MediaSource, ev: Event) => any) | null = null;

      onsourceclose: ((this: MediaSource, ev: Event) => any) | null = null;

      lastSourceBuffer: FakeSourceBuffer | null = null;

      addSourceBuffer = vi.fn((_mime: string) => {
        const sourceBuffer = new FakeSourceBuffer();
        this.lastSourceBuffer = sourceBuffer;
        return sourceBuffer as unknown as SourceBuffer;
      });

      removeSourceBuffer = vi.fn();

      endOfStream = vi.fn(() => {
        this.readyState = 'ended';
      });

      setLiveSeekableRange = vi.fn();

      clearLiveSeekableRange = vi.fn();

      private readonly listeners = new Map<string, Array<() => void>>();

      constructor() {
        FakeManagedMediaSource.instances.push(this);
      }

      addEventListener(type: string, listener: () => void): void {
        const next = this.listeners.get(type) || [];
        next.push(listener);
        this.listeners.set(type, next);

        if (type === 'sourceopen') {
          setTimeout(() => {
            this.readyState = 'open';
            listener();
          }, 0);
        }
      }

      removeEventListener(type: string, listener: () => void): void {
        const next = (this.listeners.get(type) || []).filter(
          (item) => item !== listener,
        );
        this.listeners.set(type, next);
      }

      dispatchEvent(): boolean {
        return true;
      }
    }

    const restoreMediaSourceApis = overrideMediaSourceApis({
      managedMediaSourceCtor: FakeManagedMediaSource as unknown as MockMediaSourceCtor,
    });

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
      });

      await pipeline.ensurePipeline('video/mp4');

      const managedInstance = FakeManagedMediaSource.instances[0];
      expect(managedInstance).toBeTruthy();
      expect(managedInstance?.addSourceBuffer).toHaveBeenCalledTimes(1);
      expect(managedInstance?.addSourceBuffer).toHaveBeenCalledWith('video/mp4');
      expect(managedInstance?.lastSourceBuffer?.mode).toBe('sequence');
      expect(fakeVideo.disableRemotePlayback).toBe(true);
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('waits for ManagedMediaSource startstreaming before appending queued chunks', async () => {
    setAppleMobileNavigator();
    const onAppendError = vi.fn();

    class FakeSourceBuffer {
      mode: AppendMode = 'segments';

      updating = false;

      appendBuffer = vi.fn();

      addEventListener(): void {
        // noop for test
      }

      removeEventListener(): void {
        // noop for test
      }
    }

    class FakeManagedMediaSource {
      static instances: FakeManagedMediaSource[] = [];

      static isTypeSupported(): boolean {
        return true;
      }

      streaming = false;

      readyState: MockMediaSourceReadyState = 'closed';

      sourceBuffers: SourceBufferList = {} as SourceBufferList;

      activeSourceBuffers: SourceBufferList = {} as SourceBufferList;

      duration = 0;

      onsourceopen: ((this: MediaSource, ev: Event) => any) | null = null;

      onsourceended: ((this: MediaSource, ev: Event) => any) | null = null;

      onsourceclose: ((this: MediaSource, ev: Event) => any) | null = null;

      lastSourceBuffer: FakeSourceBuffer | null = null;

      addSourceBuffer = vi.fn(() => {
        const sourceBuffer = new FakeSourceBuffer();
        this.lastSourceBuffer = sourceBuffer;
        return sourceBuffer as unknown as SourceBuffer;
      });

      removeSourceBuffer = vi.fn();

      endOfStream = vi.fn(() => {
        this.readyState = 'ended';
      });

      setLiveSeekableRange = vi.fn();

      clearLiveSeekableRange = vi.fn();

      private readonly listeners = new Map<string, Array<() => void>>();

      constructor() {
        FakeManagedMediaSource.instances.push(this);
      }

      addEventListener(type: string, listener: () => void): void {
        const next = this.listeners.get(type) || [];
        next.push(listener);
        this.listeners.set(type, next);

        if (type === 'sourceopen') {
          setTimeout(() => {
            this.readyState = 'open';
            listener();
          }, 0);
        }
      }

      removeEventListener(type: string, listener: () => void): void {
        const next = (this.listeners.get(type) || []).filter(
          (item) => item !== listener,
        );
        this.listeners.set(type, next);
      }

      dispatch(type: string): void {
        for (const listener of this.listeners.get(type) || []) {
          listener();
        }
      }

      dispatchEvent(): boolean {
        return true;
      }
    }

    const restoreMediaSourceApis = overrideMediaSourceApis({
      managedMediaSourceCtor: FakeManagedMediaSource as unknown as MockMediaSourceCtor,
    });

    try {
      const pipeline = createAvPipeline({
        getVideoEl: () => fakeVideo,
        onAppendError,
      });

      await pipeline.ensurePipeline('video/mp4');
      const managedInstance = FakeManagedMediaSource.instances[0];
      expect(managedInstance).toBeTruthy();
      expect(managedInstance?.streaming).toBe(false);

      pipeline.enqueueChunk(new Uint8Array([9, 8, 7]).buffer);
      expect(managedInstance?.lastSourceBuffer?.appendBuffer).not.toHaveBeenCalled();

      managedInstance!.streaming = true;
      managedInstance!.dispatch('startstreaming');
      expect(managedInstance?.lastSourceBuffer?.appendBuffer).toHaveBeenCalledTimes(1);
      expect(onAppendError).not.toHaveBeenCalled();
    } finally {
      restoreMediaSourceApis();
    }
  });

  it('waits for the startup buffer before first playback', async () => {
    let nowMs = 0;
    const onPlaybackStarted = vi.fn();
    fakeVideo.currentTime = 0;
    fakeVideo.buffered = createBufferedRange(0, 0.2);
    fakeVideo.paused = true;
    fakeVideo.play.mockClear();

    const pipeline = createAvPipeline({
      getVideoEl: () => fakeVideo,
      getNow: () => nowMs,
      onPlaybackStarted,
    });

    pipeline.markStreamStarting();
    pipeline.enqueueChunk(new Uint8Array([1, 2, 3]).buffer);

    nowMs = 400;
    pipeline.maybeStartPlayback();
    expect(onPlaybackStarted).not.toHaveBeenCalled();
    expect(fakeVideo.play).not.toHaveBeenCalled();

    nowMs = 500;
    pipeline.maybeStartPlayback();
    expect(onPlaybackStarted).toHaveBeenCalledTimes(1);
    expect(fakeVideo.play).toHaveBeenCalledTimes(1);
  });

  it('resumes immediately after a later buffering stall once data is available', async () => {
    let nowMs = 0;
    const onPlaybackStarted = vi.fn();
    fakeVideo.currentTime = 0;
    fakeVideo.buffered = createBufferedRange(0, 0.2);
    fakeVideo.paused = true;
    fakeVideo.play.mockClear();

    const pipeline = createAvPipeline({
      getVideoEl: () => fakeVideo,
      getNow: () => nowMs,
      onPlaybackStarted,
    });

    pipeline.markStreamStarting();
    pipeline.enqueueChunk(new Uint8Array([1, 2, 3]).buffer);

    nowMs = 500;
    pipeline.maybeStartPlayback();
    expect(onPlaybackStarted).toHaveBeenCalledTimes(1);
    expect(fakeVideo.play).toHaveBeenCalledTimes(1);

    fakeVideo.paused = true;
    fakeVideo.currentTime = 0.2;
    fakeVideo.buffered = createBufferedRange(0.2, 0.35);
    fakeVideo.dispatch('waiting');

    nowMs = 510;
    pipeline.maybeStartPlayback();
    expect(onPlaybackStarted).toHaveBeenCalledTimes(1);
    expect(fakeVideo.play).toHaveBeenCalledTimes(2);
  });

  it('returns archived chunks and clears them when taken', () => {
    const pipeline = createAvPipeline({
      getVideoEl: () => fakeVideo,
    });

    const originalChunk = new Uint8Array([7, 8, 9]).buffer;
    pipeline.markStreamStarting();
    pipeline.enqueueChunk(originalChunk);

    const archivedChunks = pipeline.takeArchivedStreamChunks();

    expect(archivedChunks).toHaveLength(1);
    expect(archivedChunks[0]).not.toBe(originalChunk);
    expect(Array.from(new Uint8Array(archivedChunks[0]))).toEqual([7, 8, 9]);
    expect(pipeline.hasArchivedChunks()).toBe(false);
  });

  it('tracks archived segments by lifecycle and supports completed-only snapshots', () => {
    const pipeline = createAvPipeline({
      getVideoEl: () => fakeVideo,
    });

    pipeline.markStreamStarting();
    pipeline.noteSegmentInit({
      segmentIdx: 1,
      streamId: 'seg-1',
      mime: 'video/mp4',
    });
    pipeline.enqueueChunk(new Uint8Array([1, 2]).buffer);
    pipeline.noteSegmentComplete({
      segmentIdx: 1,
      streamId: 'seg-1',
    });

    pipeline.noteSegmentInit({
      segmentIdx: 2,
      streamId: 'seg-2',
      mime: 'video/mp4',
    });
    pipeline.enqueueChunk(new Uint8Array([3, 4]).buffer);

    const allSegments = pipeline.buildArchivedSegmentSnapshots({
      includeInProgress: true,
    });
    expect(allSegments).toHaveLength(2);
    expect(allSegments[0]?.completed).toBe(true);
    expect(allSegments[1]?.completed).toBe(false);
    expect(pipeline.hasArchivedCompletedSegments()).toBe(true);

    const completedOnly = pipeline.takeArchivedSegmentSnapshots({
      includeInProgress: false,
    });
    expect(completedOnly).toHaveLength(1);
    expect(completedOnly[0]?.streamId).toBe('seg-1');

    const remaining = pipeline.buildArchivedSegmentSnapshots({
      includeInProgress: true,
    });
    expect(remaining).toHaveLength(1);
    expect(remaining[0]?.streamId).toBe('seg-2');
    expect(remaining[0]?.completed).toBe(false);
  });
});
