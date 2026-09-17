'use client';

import { Download, ImageOff, Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { downloadJobVideo, getJobVideoUrl } from '@/lib/api';
import { getJobThumbnailUrl, isJobImage } from '@/lib/jobResults';
import type { Job } from '@/lib/types';
import { cn, downloadBlob } from '@/lib/utils';

export function JobResultMetadata({ job }: { job: Job }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
      <span>{job.width} × {job.height}</span>
      <span>{job.num_frames} frames</span>
      <span>{job.num_inference_steps} steps</span>
      <span>Guidance {job.guidance_scale}</span>
      <span>Seed {job.seed}</span>
    </div>
  );
}

function ResultDownloadButton({
  isImage,
  downloading,
  onDownload,
  className,
}: {
  isImage: boolean;
  downloading: boolean;
  onDownload: () => Promise<void>;
  className: string;
}) {
  const label = `Download ${isImage ? 'image' : 'video'}`;
  return (
    <Button
      type="button"
      size="icon"
      variant="ghost"
      aria-label={label}
      aria-busy={downloading}
      title={downloading ? 'Downloading…' : label}
      disabled={downloading}
      onClick={(event) => {
        event.stopPropagation();
        void onDownload();
      }}
      className={cn(
        'absolute z-10 rounded-full border-white/20 bg-black/70 text-white hover:bg-black/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white',
        className,
      )}
    >
      {downloading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Download className="size-4" aria-hidden />}
    </Button>
  );
}

function NativeResultVideo({ job, onError }: { job: Job; onError: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const source = getJobVideoUrl(job.id);
  useEffect(() => {
    const video = videoRef.current;
    // Restore the source after React's development StrictMode cleanup/replay.
    if (video && video.getAttribute('src') !== source) video.setAttribute('src', source);
    return () => {
      // Stop playback and cancel buffered media when a result leaves the viewport.
      if (video) {
        video.pause();
        video.removeAttribute('src');
        video.load();
      }
    };
  }, [source]);

  return (
    <video
      ref={videoRef}
      src={source}
      poster={getJobThumbnailUrl(job.id)}
      aria-label={job.prompt ? `Generated video: ${job.prompt}` : 'Generated video'}
      data-studio-result-video
      className="h-full w-full object-contain"
      controls
      playsInline
      preload="none"
      onError={onError}
      onPlay={(event) => {
        const playing = event.currentTarget;
        document.querySelectorAll<HTMLVideoElement>('video[data-studio-result-video]').forEach((video) => {
          if (video !== playing) video.pause();
        });
      }}
    />
  );
}

/** Native players mount near the viewport; leaving it releases their media buffers. */
export default function JobResultPreview({
  job,
  thumbnailEnabled = true,
  className,
}: {
  job: Job;
  thumbnailEnabled?: boolean;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [nearViewport, setNearViewport] = useState(false);
  const [mediaFailed, setMediaFailed] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const isImage = isJobImage(job);

  useEffect(() => {
    if (!thumbnailEnabled || isImage) return;
    if (typeof IntersectionObserver === 'undefined') {
      setNearViewport(true);
      return;
    }
    const element = containerRef.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      ([entry]) => setNearViewport(entry.isIntersecting),
      { rootMargin: '200px' },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [thumbnailEnabled, isImage]);

  async function download() {
    if (downloading) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      const blob = await downloadJobVideo(job.id);
      const extension = job.output_path?.split('.').pop()?.toLowerCase() || 'mp4';
      downloadBlob(blob, `job_${job.id}.${extension}`);
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : 'Download failed');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div ref={containerRef} className={cn('rounded-md', className)} onClick={(event) => event.stopPropagation()}>
      <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-[inherit] bg-black">
        {!thumbnailEnabled || mediaFailed ? (
          <div className="flex flex-col items-center gap-2 px-4 py-6 text-center text-sm text-slate-300">
            {mediaFailed && <ImageOff className="size-5" aria-hidden />}
            <span role="status">{mediaFailed ? 'Preview unavailable' : 'Result available'}</span>
            <a
              href={getJobVideoUrl(job.id)}
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-4 hover:text-white"
            >
              Open original
            </a>
            {mediaFailed && (
              <Button type="button" variant="secondary" size="sm" onClick={() => setMediaFailed(false)}>
                Retry preview
              </Button>
            )}
          </div>
        ) : isImage ? (
          <a
            href={getJobVideoUrl(job.id)}
            target="_blank"
            rel="noreferrer"
            aria-label="Open original image"
            className="flex h-full w-full items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={getJobThumbnailUrl(job.id)}
              alt={job.prompt || 'Generated image'}
              loading="lazy"
              decoding="async"
              width={320}
              height={180}
              className="h-full w-full object-contain"
              onError={() => setMediaFailed(true)}
            />
          </a>
        ) : nearViewport ? (
          <NativeResultVideo job={job} onError={() => setMediaFailed(true)} />
        ) : null}
        <ResultDownloadButton
          isImage={isImage}
          downloading={downloading}
          onDownload={download}
          className="right-2 top-2"
        />
      </div>
      {downloadError && <p role="alert" className="mt-2 text-xs text-destructive">{downloadError}</p>}
    </div>
  );
}
