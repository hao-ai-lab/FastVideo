export const DEFAULT_AV_MIME = 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"';

const AV_MIME_FALLBACKS = [
  'video/mp4; codecs="avc1.4d401e,mp4a.40.2"',
  'video/mp4; codecs="avc1.42e01e,mp4a.40.2"',
  'video/mp4; codecs="avc1.640028,mp4a.40.2"',
  'video/mp4',
];

const AV_INITIAL_MIN_START_DELAY_MS = 0;
const AV_INITIAL_PREBUFFER_SECONDS = 1.0;
const AV_INITIAL_MAX_START_WAIT_MS = 500;

type MediaSourceLike = MediaSource;

type MediaSourceLikeConstructor = {
  new (): MediaSourceLike;
  isTypeSupported: (mime: string) => boolean;
};

interface ManagedMediaSourceLike extends MediaSource {
  streaming?: boolean;
}

interface MediaSourceWindow extends Window {
  ManagedMediaSource?: MediaSourceLikeConstructor;
  MediaSource?: MediaSourceLikeConstructor;
}

function isAppleMobileBrowser(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  const userAgent = String(navigator.userAgent || '');
  const platform = String(navigator.platform || '');
  const maxTouchPoints = Number(navigator.maxTouchPoints || 0);
  const isAppleMobileUserAgent = /iPhone|iPad|iPod/i.test(userAgent);
  const isTouchMac = /Mac/i.test(platform) && maxTouchPoints > 1;

  return isAppleMobileUserAgent || isTouchMac;
}

function isAndroidChromeBrowser(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  const userAgent = String(navigator.userAgent || '');
  return /Android/i.test(userAgent)
    && /(Chrome|CriOS)/i.test(userAgent)
    && !/(EdgA|OPR|SamsungBrowser)/i.test(userAgent);
}

function resolveMediaSourceConstructor(): {
  ctor: MediaSourceLikeConstructor | null;
  usesManagedMediaSource: boolean;
} {
  if (typeof window === 'undefined') {
    return {
      ctor: null,
      usesManagedMediaSource: false,
    };
  }

  const mediaSourceWindow = window as MediaSourceWindow;
  if (typeof mediaSourceWindow.ManagedMediaSource === 'function') {
    return {
      ctor: mediaSourceWindow.ManagedMediaSource,
      usesManagedMediaSource: true,
    };
  }

  if (typeof mediaSourceWindow.MediaSource === 'function') {
    return {
      ctor: mediaSourceWindow.MediaSource,
      usesManagedMediaSource: false,
    };
  }

  return {
    ctor: null,
    usesManagedMediaSource: false,
  };
}

export interface AvPipeline {
  ensurePipeline(mime: string, forceRecreate?: boolean): Promise<void>;
  enqueueChunk(chunk: ArrayBuffer | ArrayBufferView): void;
  noteSegmentInit(meta?: {
    segmentIdx?: number | null;
    streamId?: string;
    mime?: string;
  }): void;
  noteSegmentComplete(meta?: {
    segmentIdx?: number | null;
    streamId?: string;
  }): void;
  maybeStartPlayback(): void;
  tryEndStream(): void;
  markStreamStarting(): void;
  setStreamCompleted(value: boolean): void;
  hasArchivedChunks(): boolean;
  hasArchivedCompletedSegments(): boolean;
  buildArchivedStreamChunks(): ArrayBuffer[];
  buildArchivedSegmentSnapshots(options?: {
    includeInProgress?: boolean;
  }): ArchivedAvSegment[];
  buildArchivedStreamBlob(): Blob | null;
  takeArchivedStreamChunks(): ArrayBuffer[];
  takeArchivedSegmentSnapshots(options?: {
    includeInProgress?: boolean;
  }): ArchivedAvSegment[];
  takeArchivedStreamBlob(): Blob | null;
  usesNativePlaybackFallback(): boolean;
  reset(): void;
}

export interface ArchivedAvSegment {
  key: string;
  segmentIdx: number | null;
  streamId: string;
  mime: string;
  completed: boolean;
  chunks: ArrayBuffer[];
}

interface CreateAvPipelineParams {
  getVideoEl: () => HTMLVideoElement | null;
  getNow?: () => number;
  onAppendError?: (message: string, error?: any) => void;
  onPlaybackStarted?: () => void;
}

export function createAvPipeline({
  getVideoEl,
  getNow = () => performance.now(),
  onAppendError = () => {},
  onPlaybackStarted = () => {},
}: CreateAvPipelineParams): AvPipeline {
  const {
    ctor: mediaSourceCtor,
    usesManagedMediaSource,
  } = resolveMediaSourceConstructor();
  // Only force native fallback when the browser lacks a usable media
  // source implementation. Android Chrome should stay on the faster
  // live playback path when MediaSource is available.
  const useNativePlaybackFallback = (
    (isAppleMobileBrowser() && !mediaSourceCtor)
    || (isAndroidChromeBrowser() && !mediaSourceCtor)
  );
  let sourceBuffer: SourceBuffer | null = null;
  let mediaSource: MediaSource | null = null;
  let mediaObjectUrl: string | null = null;
  let mediaSourceOpen = false;
  let sourceBufferMime = '';
  let mediaChunkQueue: ArrayBuffer[] = [];
  let sourceBufferUpdateEndHandler: (() => void) | null = null;
  let sourceBufferErrorHandler: ((event: Event) => void) | null = null;
  let mediaSourceOpenHandler: (() => void) | null = null;
  let mediaSourceStartStreamingHandler: (() => void) | null = null;
  let mediaSourceEndStreamingHandler: (() => void) | null = null;
  let pendingMediaEndOfStream = false;
  let managedStreamingActive = !usesManagedMediaSource;
  let firstChunkAtMs: number | null = null;
  let streamCompleted = false;
  let initialPlaybackStarted = false;
  let playbackResumePending = false;
  let archivedChunks: ArrayBuffer[] = [];
  let archivedSegments: ArchivedAvSegment[] = [];
  let activeArchivedSegmentKey: string | null = null;
  let archivedSegmentCounter = 0;
  let boundVideoEl: HTMLVideoElement | null = null;
  let videoWaitingHandler: (() => void) | null = null;
  let videoStalledHandler: (() => void) | null = null;
  let videoPlayingHandler: (() => void) | null = null;
  let videoPauseHandler: (() => void) | null = null;
  let videoPlayHandler: (() => void) | null = null;
  let userPaused = false;
  let stallRecoveryTimerId: ReturnType<typeof setInterval> | null = null;

  function uniqValues(values: any[]): string[] {
    const result: string[] = [];
    for (const value of values) {
      if (typeof value !== 'string') {
        continue;
      }
      const trimmed = value.trim();
      if (!trimmed || result.includes(trimmed)) {
        continue;
      }
      result.push(trimmed);
    }
    return result;
  }

  function isMimeTypeSupported(mime: string): boolean {
    if (!mediaSourceCtor) {
      return false;
    }

    try {
      return mediaSourceCtor.isTypeSupported(mime);
    } catch (error) {
      return false;
    }
  }

  function resolveSupportedAvMime(mime: string): string | null {
    const candidates = uniqValues([
      mime,
      DEFAULT_AV_MIME,
      ...AV_MIME_FALLBACKS,
    ]);
    for (const candidate of candidates) {
      if (isMimeTypeSupported(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  function listAvMimeCandidates(mime: string): string[] {
    return uniqValues([
      mime,
      DEFAULT_AV_MIME,
      ...AV_MIME_FALLBACKS,
    ]);
  }

  function cleanupVideoBindings(): void {
    if (!boundVideoEl) {
      return;
    }

    if (videoWaitingHandler) {
      boundVideoEl.removeEventListener('waiting', videoWaitingHandler);
    }
    if (videoStalledHandler) {
      boundVideoEl.removeEventListener('stalled', videoStalledHandler);
    }
    if (videoPlayingHandler) {
      boundVideoEl.removeEventListener('playing', videoPlayingHandler);
    }
    if (videoPauseHandler) {
      boundVideoEl.removeEventListener('pause', videoPauseHandler);
    }
    if (videoPlayHandler) {
      boundVideoEl.removeEventListener('play', videoPlayHandler);
    }

    boundVideoEl = null;
    videoWaitingHandler = null;
    videoStalledHandler = null;
    videoPlayingHandler = null;
    videoPauseHandler = null;
    videoPlayHandler = null;
    clearStallRecovery();
  }

  function clearStallRecovery(): void {
    if (stallRecoveryTimerId !== null) {
      clearInterval(stallRecoveryTimerId);
      stallRecoveryTimerId = null;
    }
  }

  function startStallRecovery(): void {
    if (stallRecoveryTimerId !== null) {
      return;
    }
    stallRecoveryTimerId = setInterval(() => {
      // Stop polling once playback is running normally.
      if (initialPlaybackStarted && !playbackResumePending) {
        clearStallRecovery();
        return;
      }
      // On iOS Safari, ManagedMediaSource's startstreaming event can be
      // missed or delayed. Poll the streaming property directly to detect
      // when appending is allowed again.
      if (usesManagedMediaSource && !managedStreamingActive && mediaSource) {
        const managed = mediaSource as ManagedMediaSourceLike;
        if (managed.streaming !== false) {
          managedStreamingActive = true;
          flushQueue();
        }
      }
      maybeStartPlayback();
    }, 250);
  }

  function ensureVideoBindings(videoEl: HTMLVideoElement): void {
    if (!videoEl || boundVideoEl === videoEl) {
      return;
    }

    cleanupVideoBindings();
    boundVideoEl = videoEl;
    videoWaitingHandler = () => {
      if (initialPlaybackStarted) {
        playbackResumePending = true;
        startStallRecovery();
      }
    };
    videoStalledHandler = () => {
      if (initialPlaybackStarted) {
        playbackResumePending = true;
        startStallRecovery();
      }
    };
    videoPlayingHandler = () => {
      playbackResumePending = false;
      clearStallRecovery();
    };
    videoPauseHandler = () => {
      if (initialPlaybackStarted) {
        userPaused = true;
      }
    };
    videoPlayHandler = () => {
      userPaused = false;
    };

    videoEl.addEventListener('waiting', videoWaitingHandler);
    videoEl.addEventListener('stalled', videoStalledHandler);
    videoEl.addEventListener('playing', videoPlayingHandler);
    videoEl.addEventListener('pause', videoPauseHandler);
    videoEl.addEventListener('play', videoPlayHandler);
  }

  function cloneChunk(chunk: ArrayBuffer | ArrayBufferView): ArrayBuffer {
    if (chunk instanceof ArrayBuffer) {
      return chunk.slice(0);
    }
    if (ArrayBuffer.isView(chunk)) {
      return new Uint8Array(
        chunk.buffer,
        chunk.byteOffset,
        chunk.byteLength,
      ).slice().buffer;
    }
    return new Uint8Array(chunk).slice().buffer;
  }

  function resetArchivedSegments(): void {
    archivedSegments = [];
    activeArchivedSegmentKey = null;
    archivedSegmentCounter = 0;
  }

  function cloneArchivedSegment(segment: ArchivedAvSegment): ArchivedAvSegment {
    return {
      ...segment,
      chunks: segment.chunks.map((chunk) => cloneChunk(chunk)),
    };
  }

  function buildArchivedSegmentKey(
    segmentIdx: number | null,
    streamId: string,
  ): string {
    return `${segmentIdx !== null ? segmentIdx : 'na'}:${streamId || 'na'}`;
  }

  function normalizeSegmentMeta({
    segmentIdx = null,
    streamId = '',
    mime = '',
  }: {
    segmentIdx?: number | null;
    streamId?: string;
    mime?: string;
  }): {
    segmentIdx: number | null;
    streamId: string;
    mime: string;
  } {
    const normalizedSegmentIdx =
      Number.isInteger(segmentIdx) ? Number(segmentIdx) : null;
    const normalizedStreamId = typeof streamId === 'string'
      && streamId.trim()
      ? streamId.trim()
      : '';
    const normalizedMime = typeof mime === 'string' && mime.trim()
      ? mime.trim()
      : sourceBufferMime || DEFAULT_AV_MIME;
    return {
      segmentIdx: normalizedSegmentIdx,
      streamId: normalizedStreamId,
      mime: normalizedMime,
    };
  }

  function getOrCreateActiveSegment(): ArchivedAvSegment {
    if (activeArchivedSegmentKey) {
      const existing = archivedSegments.find(
        (segment) => segment.key === activeArchivedSegmentKey,
      );
      if (existing) {
        return existing;
      }
    }

    archivedSegmentCounter += 1;
    const streamId = `implicit-${archivedSegmentCounter}`;
    const key = buildArchivedSegmentKey(null, streamId);
    const nextSegment: ArchivedAvSegment = {
      key,
      segmentIdx: null,
      streamId,
      mime: sourceBufferMime || DEFAULT_AV_MIME,
      completed: false,
      chunks: [],
    };
    archivedSegments.push(nextSegment);
    activeArchivedSegmentKey = key;
    return nextSegment;
  }

  function cleanup(): void {
    // NOTE: mediaChunkQueue is intentionally NOT cleared here.
    // ensurePipeline() calls cleanup() to tear down the old
    // MediaSource, but chunks may have already been enqueued for
    // the new stream during the async gap before ensurePipeline
    // runs. Clearing them here drops the fMP4 init segment,
    // causing black video / append errors on mobile.
    // The queue is cleared in reset() and markStreamStarting().
    mediaSourceOpen = false;
    sourceBufferMime = '';
    pendingMediaEndOfStream = false;
    managedStreamingActive = !usesManagedMediaSource;
    firstChunkAtMs = null;
    streamCompleted = false;
    initialPlaybackStarted = false;
    playbackResumePending = false;
    archivedChunks = [];
    resetArchivedSegments();

    if (sourceBuffer && sourceBufferUpdateEndHandler) {
      sourceBuffer.removeEventListener('updateend', sourceBufferUpdateEndHandler);
    }
    if (sourceBuffer && sourceBufferErrorHandler) {
      sourceBuffer.removeEventListener('error', sourceBufferErrorHandler);
    }
    sourceBuffer = null;
    sourceBufferUpdateEndHandler = null;
    sourceBufferErrorHandler = null;

    if (mediaSource && mediaSourceOpenHandler) {
      mediaSource.removeEventListener('sourceopen', mediaSourceOpenHandler);
    }
    if (mediaSource && mediaSourceStartStreamingHandler) {
      mediaSource.removeEventListener(
        'startstreaming',
        mediaSourceStartStreamingHandler,
      );
    }
    if (mediaSource && mediaSourceEndStreamingHandler) {
      mediaSource.removeEventListener(
        'endstreaming',
        mediaSourceEndStreamingHandler,
      );
    }
    if (mediaSource && mediaSource.readyState === 'open') {
      try {
        mediaSource.endOfStream();
      } catch (error) {
        // noop
      }
    }
    mediaSource = null;
    mediaSourceOpenHandler = null;
    mediaSourceStartStreamingHandler = null;
    mediaSourceEndStreamingHandler = null;

    const videoEl = getVideoEl();
    cleanupVideoBindings();
    if (videoEl) {
      try {
        videoEl.pause();
      } catch (error) {
        // noop
      }
      if ((videoEl as any).srcObject) {
        (videoEl as any).srcObject = null;
      } else {
        videoEl.removeAttribute('src');
      }
      videoEl.load();
    }

    if (mediaObjectUrl) {
      URL.revokeObjectURL(mediaObjectUrl);
      mediaObjectUrl = null;
    }
  }

  function flushQueue(): void {
    if (!sourceBuffer || !mediaSourceOpen || !mediaSource) return;
    if (mediaSource.readyState !== 'open') return;
    if (usesManagedMediaSource && !managedStreamingActive) return;
    if (sourceBuffer.updating) return;
    if (mediaChunkQueue.length === 0) return;

    const nextChunk = mediaChunkQueue.shift()!;
    try {
      sourceBuffer.appendBuffer(nextChunk);
    } catch (error) {
      const errorName = (error as { name?: unknown })?.name;
      if (
        usesManagedMediaSource
        && errorName === 'InvalidStateError'
      ) {
        const managedMediaSource = mediaSource as ManagedMediaSourceLike;
        if (managedMediaSource.streaming === false) {
          managedStreamingActive = false;
          mediaChunkQueue = [nextChunk, ...mediaChunkQueue];
          return;
        }
      }
      mediaChunkQueue = [];
      onAppendError('Unable to append media chunk.', error);
    }
  }

  function enqueueChunk(chunk: ArrayBuffer | ArrayBufferView): void {
    if (firstChunkAtMs === null) {
      firstChunkAtMs = getNow();
    }
    const clonedChunk = cloneChunk(chunk);
    archivedChunks.push(clonedChunk);
    const archivedSegment = getOrCreateActiveSegment();
    archivedSegment.chunks.push(clonedChunk);
    mediaChunkQueue.push(clonedChunk);
    flushQueue();
    // If chunks couldn't be flushed (e.g. ManagedMediaSource streaming
    // is not yet active), start the recovery poll so we don't deadlock
    // waiting for a startstreaming event that may be delayed.
    if (mediaChunkQueue.length > 0 && usesManagedMediaSource) {
      startStallRecovery();
    }
  }

  function noteSegmentInit({
    segmentIdx = null,
    streamId = '',
    mime = '',
  }: {
    segmentIdx?: number | null;
    streamId?: string;
    mime?: string;
  } = {}): void {
    const normalized = normalizeSegmentMeta({
      segmentIdx,
      streamId,
      mime,
    });

    let normalizedStreamId = normalized.streamId;
    if (!normalizedStreamId) {
      archivedSegmentCounter += 1;
      normalizedStreamId = `stream-${archivedSegmentCounter}`;
    }

    const key = buildArchivedSegmentKey(
      normalized.segmentIdx,
      normalizedStreamId,
    );
    let segment = archivedSegments.find((entry) => entry.key === key);
    if (!segment) {
      segment = {
        key,
        segmentIdx: normalized.segmentIdx,
        streamId: normalizedStreamId,
        mime: normalized.mime,
        completed: false,
        chunks: [],
      };
      archivedSegments.push(segment);
    } else {
      segment.segmentIdx = normalized.segmentIdx;
      segment.streamId = normalizedStreamId;
      segment.mime = normalized.mime;
      segment.completed = false;
    }

    activeArchivedSegmentKey = key;
  }

  function noteSegmentComplete({
    segmentIdx = null,
    streamId = '',
  }: {
    segmentIdx?: number | null;
    streamId?: string;
  } = {}): void {
    const normalizedSegmentIdx =
      Number.isInteger(segmentIdx) ? Number(segmentIdx) : null;
    const normalizedStreamId = typeof streamId === 'string'
      && streamId.trim()
      ? streamId.trim()
      : '';

    let targetSegment: ArchivedAvSegment | undefined;

    if (normalizedStreamId) {
      targetSegment = archivedSegments.find(
        (segment) =>
          segment.streamId === normalizedStreamId
          && (
            normalizedSegmentIdx === null
            || segment.segmentIdx === normalizedSegmentIdx
          ),
      );
    }

    if (!targetSegment && normalizedSegmentIdx !== null) {
      const matchingSegments = archivedSegments.filter(
        (segment) =>
          segment.segmentIdx === normalizedSegmentIdx && !segment.completed,
      );
      targetSegment = matchingSegments[matchingSegments.length - 1];
    }

    if (!targetSegment && activeArchivedSegmentKey) {
      targetSegment = archivedSegments.find(
        (segment) => segment.key === activeArchivedSegmentKey,
      );
    }

    if (!targetSegment) {
      return;
    }

    targetSegment.completed = true;
    if (targetSegment.key === activeArchivedSegmentKey) {
      activeArchivedSegmentKey = null;
    }
  }

  function getBufferedAheadSeconds(): number {
    const videoEl = getVideoEl();
    if (!videoEl) return 0;

    let buffered = 0;
    try {
      const ranges = videoEl.buffered;
      if (!ranges || ranges.length === 0) return 0;
      const t = Math.max(videoEl.currentTime || 0, 0);

      for (let i = 0; i < ranges.length; i++) {
        const start = ranges.start(i);
        const end = ranges.end(i);
        if (t >= start && t <= end) {
          buffered = Math.max(0, end - t);
          break;
        }
        if (t < start) {
          buffered = Math.max(0, end - start);
          break;
        }
      }
    } catch (error) {
      buffered = 0;
    }

    return buffered;
  }

  function maybeStartPlayback(): void {
    const videoEl = getVideoEl();
    if (!videoEl) return;
    ensureVideoBindings(videoEl);

    const bufferedAhead = getBufferedAheadSeconds();
    if (bufferedAhead <= 0) return;

    if (initialPlaybackStarted) {
      if (!playbackResumePending || userPaused) {
        return;
      }

      playbackResumePending = false;
      const resumePromise = videoEl.play();
      if (resumePromise?.catch) {
        resumePromise.catch((err) => {
          console.warn('av resume play() rejected:', err);
          playbackResumePending = true;
        });
      }
      return;
    }

    if (firstChunkAtMs === null) return;

    const waitedMs = getNow() - firstChunkAtMs;
    const delayElapsed = waitedMs >= AV_INITIAL_MIN_START_DELAY_MS;
    const enoughBuffer = bufferedAhead >= AV_INITIAL_PREBUFFER_SECONDS;
    const maxWaitReached = AV_INITIAL_MAX_START_WAIT_MS > 0
      && waitedMs >= AV_INITIAL_MAX_START_WAIT_MS;

    if (!streamCompleted && (!delayElapsed || (!enoughBuffer && !maxWaitReached))) {
      return;
    }

    initialPlaybackStarted = true;
    playbackResumePending = false;
    onPlaybackStarted();

    const playPromise = videoEl.play();
    if (playPromise?.catch) {
      playPromise.catch((err) => {
        console.warn('av initial play() rejected:', err);
        playbackResumePending = true;
      });
    }
  }

  function tryEndStream(): void {
    if (!mediaSource || !mediaSourceOpen) return;

    if (sourceBuffer?.updating) {
      pendingMediaEndOfStream = true;
      return;
    }

    if (mediaChunkQueue.length > 0) {
      pendingMediaEndOfStream = true;
      return;
    }

    try {
      mediaSource.endOfStream();
    } catch (error) {
      // noop
    }
    mediaSourceOpen = false;
    pendingMediaEndOfStream = false;
  }

  async function ensurePipeline(
    mime: string,
    forceRecreate = false,
  ): Promise<void> {
    const requestedMime = mime || DEFAULT_AV_MIME;
    const selectedMime = resolveSupportedAvMime(requestedMime);
    const candidateMimes = listAvMimeCandidates(requestedMime);

    if (!forceRecreate
      && sourceBuffer
      && mediaSourceOpen
      && sourceBufferMime === selectedMime) {
      return;
    }

    if (useNativePlaybackFallback) {
      const videoEl = getVideoEl();
      if (!videoEl) {
        throw new Error('Video element is not ready.');
      }

      sourceBufferMime = requestedMime;
      return;
    }

    if (!mediaSourceCtor) {
      throw new Error(
        'ManagedMediaSource/MediaSource APIs are not available in this browser.',
      );
    }
    if (!selectedMime) {
      throw new Error(
        `No supported AV MIME type found for requested "${requestedMime}".`,
      );
    }

    const videoEl = getVideoEl();
    if (!videoEl) {
      throw new Error('Video element is not ready.');
    }

    cleanup();
    mediaSource = new mediaSourceCtor();
    if (usesManagedMediaSource) {
      videoEl.disableRemotePlayback = true;
      (videoEl as any).srcObject = mediaSource;
    } else {
      mediaObjectUrl = URL.createObjectURL(mediaSource);
      videoEl.src = mediaObjectUrl;
    }

    await new Promise<void>((resolve, reject) => {
      mediaSourceOpenHandler = () => {
        try {
          let sourceBufferInitialized = false;
          const sourceBufferErrors: string[] = [];
          for (const candidateMime of candidateMimes) {
            if (!isMimeTypeSupported(candidateMime)) {
              continue;
            }
            try {
              sourceBuffer = mediaSource!.addSourceBuffer(candidateMime);
              sourceBufferMime = candidateMime;
              sourceBufferInitialized = true;
              break;
            } catch (error: any) {
              sourceBufferErrors.push(
                `${candidateMime}: ${error?.message || String(error)}`,
              );
            }
          }

          if (!sourceBufferInitialized) {
            throw new Error(
              'Unable to initialize SourceBuffer. Candidates: '
              + `${candidateMimes.join(', ')}`
              + (
                sourceBufferErrors.length
                  ? ` Errors: ${sourceBufferErrors.join('; ')}`
                  : ''
              ),
            );
          }

          ensureVideoBindings(videoEl);
          sourceBuffer!.mode = 'sequence';
          mediaSourceOpen = true;
          if (usesManagedMediaSource) {
            const managedMediaSource = mediaSource! as ManagedMediaSourceLike;
            managedStreamingActive = managedMediaSource.streaming !== false;
            mediaSourceStartStreamingHandler = () => {
              managedStreamingActive = true;
              flushQueue();
            };
            mediaSourceEndStreamingHandler = () => {
              managedStreamingActive = false;
            };
            mediaSource!.addEventListener(
              'startstreaming',
              mediaSourceStartStreamingHandler,
            );
            mediaSource!.addEventListener(
              'endstreaming',
              mediaSourceEndStreamingHandler,
            );
          }

          sourceBufferUpdateEndHandler = () => {
            flushQueue();
            maybeStartPlayback();
            if (pendingMediaEndOfStream) {
              tryEndStream();
            }
          };

          sourceBufferErrorHandler = (event: Event) => {
            onAppendError('SourceBuffer reported a media error.', event);
          };

          sourceBuffer!.addEventListener('updateend', sourceBufferUpdateEndHandler);
          sourceBuffer!.addEventListener('error', sourceBufferErrorHandler);

          flushQueue();
          resolve();
        } catch (error) {
          reject(error);
        }
      };

      mediaSource!.addEventListener('sourceopen', mediaSourceOpenHandler, {
        once: true,
      });
    });
  }

  function markStreamStarting(): void {
    mediaChunkQueue = [];
    streamCompleted = false;
    firstChunkAtMs = null;
    initialPlaybackStarted = false;
    playbackResumePending = false;
    userPaused = false;
    pendingMediaEndOfStream = false;
    archivedChunks = [];
    resetArchivedSegments();
    clearStallRecovery();
  }

  function setStreamCompleted(value: boolean): void {
    streamCompleted = Boolean(value);
  }

  function hasArchivedChunks(): boolean {
    return archivedChunks.length > 0;
  }

  function hasArchivedCompletedSegments(): boolean {
    return archivedSegments.some(
      (segment) => segment.completed && segment.chunks.length > 0,
    );
  }

  function buildArchivedStreamChunks(): ArrayBuffer[] {
    if (!hasArchivedChunks()) {
      return [];
    }
    return archivedChunks.map((chunk) => cloneChunk(chunk));
  }

  function buildArchivedSegmentSnapshots({
    includeInProgress = true,
  }: {
    includeInProgress?: boolean;
  } = {}): ArchivedAvSegment[] {
    return archivedSegments
      .filter(
        (segment) =>
          segment.chunks.length > 0
          && (includeInProgress || segment.completed),
      )
      .map((segment) => cloneArchivedSegment(segment));
  }

  function buildArchivedStreamBlob(): Blob | null {
    const chunks = buildArchivedStreamChunks();
    if (chunks.length === 0) {
      return null;
    }
    return new Blob(chunks, {
      type: sourceBufferMime || DEFAULT_AV_MIME,
    });
  }

  function takeArchivedStreamChunks(): ArrayBuffer[] {
    const chunks = buildArchivedStreamChunks();
    archivedChunks = [];
    resetArchivedSegments();
    return chunks;
  }

  function takeArchivedSegmentSnapshots({
    includeInProgress = true,
  }: {
    includeInProgress?: boolean;
  } = {}): ArchivedAvSegment[] {
    const snapshots = buildArchivedSegmentSnapshots({ includeInProgress });
    if (includeInProgress) {
      resetArchivedSegments();
      return snapshots;
    }

    archivedSegments = archivedSegments.filter((segment) => !segment.completed);
    if (
      activeArchivedSegmentKey
      && !archivedSegments.some((segment) => segment.key === activeArchivedSegmentKey)
    ) {
      activeArchivedSegmentKey = null;
    }
    return snapshots;
  }

  function takeArchivedStreamBlob(): Blob | null {
    const blob = buildArchivedStreamBlob();
    archivedChunks = [];
    return blob;
  }

  function reset(): void {
    mediaChunkQueue = [];
    cleanup();
  }

  return {
    ensurePipeline,
    enqueueChunk,
    noteSegmentInit,
    noteSegmentComplete,
    maybeStartPlayback,
    tryEndStream,
    markStreamStarting,
    setStreamCompleted,
    hasArchivedChunks,
    hasArchivedCompletedSegments,
    buildArchivedStreamChunks,
    buildArchivedSegmentSnapshots,
    buildArchivedStreamBlob,
    takeArchivedStreamChunks,
    takeArchivedSegmentSnapshots,
    takeArchivedStreamBlob,
    usesNativePlaybackFallback() {
      return useNativePlaybackFallback;
    },
    reset,
  };
}
