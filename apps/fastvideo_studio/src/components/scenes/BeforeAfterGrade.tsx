'use client';

import * as React from 'react';

import { applyGradeInPlace, isNeutralGrade } from '@/lib/grading';

export interface BeforeAfterGradeProps {
  /** the video the snapshot is captured from -- must be `crossOrigin="anonymous"` for pixel access to work */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  brightness: number;
  contrast: number;
  saturation: number;
  /** called with the video's currentTime whenever a frame is (re)captured, so a caller with multiple
   * sections can tell which one the previewed frame falls into and keep its own selection in sync */
  onFrameCaptured?: (timeSeconds: number) => void;
}

// Downscaled so the per-pixel pass stays instant on every slider tick.
const PREVIEW_WIDTH = 320;

/**
 * A still frame from `videoRef`, shown twice: untouched, and with the current
 * color values applied via `applyGradeInPlace` -- entirely in the browser, so
 * dragging a slider never waits on the server. Captures a frame automatically
 * once the video has one to show; "Use current frame" re-captures whatever the
 * video is currently paused/seeked to, for comparing a specific moment.
 */
export function BeforeAfterGrade({ videoRef, brightness, contrast, saturation, onFrameCaptured }: BeforeAfterGradeProps) {
  const beforeRef = React.useRef<HTMLCanvasElement>(null);
  const afterRef = React.useRef<HTMLCanvasElement>(null);
  const sourceRef = React.useRef<ImageData | null>(null);
  const [ready, setReady] = React.useState(false);
  // Kept in a ref, not a `capture` dependency, so an inline arrow function prop
  // doesn't force the video-listener effect below to tear down and re-add itself every render.
  const onFrameCapturedRef = React.useRef(onFrameCaptured);
  onFrameCapturedRef.current = onFrameCaptured;

  const capture = React.useCallback(() => {
    const video = videoRef.current;
    const before = beforeRef.current;
    if (!video || !before || video.readyState < 2 || !video.videoWidth) return;

    const scale = PREVIEW_WIDTH / video.videoWidth;
    const width = PREVIEW_WIDTH;
    const height = Math.round(video.videoHeight * scale);
    before.width = width;
    before.height = height;
    const ctx = before.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, width, height);
    try {
      sourceRef.current = ctx.getImageData(0, 0, width, height);
      setReady(true);
      onFrameCapturedRef.current?.(video.currentTime);
    } catch {
      // A cross-origin frame the browser wouldn't expose pixels for -- comparison stays unavailable.
      sourceRef.current = null;
      setReady(false);
    }
  }, [videoRef]);

  React.useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.readyState >= 2) capture();
    video.addEventListener('loadeddata', capture);
    video.addEventListener('seeked', capture);
    return () => {
      video.removeEventListener('loadeddata', capture);
      video.removeEventListener('seeked', capture);
    };
  }, [videoRef, capture]);

  React.useEffect(() => {
    const source = sourceRef.current;
    const after = afterRef.current;
    if (!source || !after) return;
    after.width = source.width;
    after.height = source.height;
    const ctx = after.getContext('2d');
    if (!ctx) return;
    const copy = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);
    if (!isNeutralGrade(brightness, contrast, saturation)) {
      applyGradeInPlace(copy.data, brightness, contrast, saturation);
    }
    ctx.putImageData(copy, 0, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- redraws whenever a new frame is captured (`ready`) too
  }, [brightness, contrast, saturation, ready]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Before / after</p>
        <button type="button" onClick={capture} className="text-xs text-accent-blue hover:underline">
          Use current frame
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-1">
          <canvas ref={beforeRef} role="img" aria-label="Before color grading" className="w-full rounded-md border border-border bg-black" />
          <span className="text-center text-[11px] text-muted-foreground">Before</span>
        </div>
        <div className="flex flex-col gap-1">
          <canvas ref={afterRef} role="img" aria-label="After color grading" className="w-full rounded-md border border-border bg-black" />
          <span className="text-center text-[11px] text-muted-foreground">After</span>
        </div>
      </div>
      {!ready && (
        <p className="text-xs text-muted-foreground">Play or seek the preview above to load a frame to compare.</p>
      )}
    </div>
  );
}
