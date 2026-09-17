'use client';

import { Download, ImageOff, Loader2, Play } from 'lucide-react';
import { useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
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

/** Posters are lazy images; the full media element exists only while opened. */
export default function JobResultPreview({
  job,
  thumbnailEnabled = true,
  className,
}: {
  job: Job;
  thumbnailEnabled?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [thumbnailFailed, setThumbnailFailed] = useState(false);
  const [mediaFailed, setMediaFailed] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const isImage = isJobImage(job);
  const label = job.name?.trim() || job.model_id;

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

  function changeOpen(value: boolean) {
    setOpen(value);
    if (value) {
      setMediaFailed(false);
      setDownloadError(null);
    }
  }

  return (
    <>
      <div className={cn('rounded-md', className)} onClick={(event) => event.stopPropagation()}>
        <div className="relative overflow-hidden rounded-[inherit]">
          <button
            type="button"
            aria-label={`Preview result: ${label}`}
            onClick={(event) => {
              event.stopPropagation();
              changeOpen(true);
            }}
            className="group relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-[inherit] bg-muted text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
          >
            {thumbnailEnabled && !thumbnailFailed ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={getJobThumbnailUrl(job.id)}
                alt=""
                loading="lazy"
                decoding="async"
                width={320}
                height={180}
                onError={() => setThumbnailFailed(true)}
                className="absolute inset-0 h-full w-full object-cover"
              />
            ) : (
              <span className="flex flex-col items-center gap-1.5 text-xs">
                <ImageOff className="size-5" aria-hidden />
                {thumbnailFailed ? 'Thumbnail unavailable' : 'Open result'}
              </span>
            )}
            <span className="absolute bottom-2 left-2 inline-flex items-center gap-1.5 rounded-full bg-black/70 px-2.5 py-1 text-[11px] font-medium text-white transition-colors group-hover:bg-accent-blue">
              <Play className="size-3 fill-current" aria-hidden />
              {isImage ? 'View image' : 'Watch video'}
            </span>
          </button>
          <ResultDownloadButton
            isImage={isImage}
            downloading={downloading}
            onDownload={download}
            className="bottom-1 right-1"
          />
        </div>
        {!open && downloadError && <p role="alert" className="mt-2 text-xs text-destructive">{downloadError}</p>}
      </div>
      <Dialog open={open} onOpenChange={changeOpen}>
        {open && (
          <DialogContent className="max-h-[90vh] max-w-4xl overflow-y-auto" onClick={(event) => event.stopPropagation()}>
            <DialogHeader className="pr-10">
              <DialogTitle>{label}</DialogTitle>
              <DialogDescription className="break-all">{job.model_id}</DialogDescription>
            </DialogHeader>
            <div className="relative flex min-h-48 items-center justify-center overflow-hidden rounded-lg bg-black">
              {mediaFailed ? (
                <div role="status" className="flex flex-col items-center gap-2 px-4 py-12 text-center text-slate-300">
                  <ImageOff className="size-7" aria-hidden />
                  <span className="text-sm font-medium">Preview unavailable</span>
                  <span className="text-xs">The generated file could not be loaded.</span>
                </div>
              ) : isImage ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={getJobVideoUrl(job.id)}
                  alt={job.prompt || 'Generated image'}
                  className="max-h-[55vh] w-full object-contain"
                  onError={() => setMediaFailed(true)}
                />
              ) : (
                <video
                  src={getJobVideoUrl(job.id)}
                  aria-label={job.prompt ? `Generated video: ${job.prompt}` : 'Generated video'}
                  className="max-h-[55vh] w-full"
                  controls
                  playsInline
                  preload="metadata"
                  onError={() => setMediaFailed(true)}
                />
              )}
              <ResultDownloadButton
                isImage={isImage}
                downloading={downloading}
                onDownload={download}
                className="right-2 top-2"
              />
            </div>
            <p className="text-sm text-foreground">{job.prompt || 'No prompt recorded.'}</p>
            <JobResultMetadata job={job} />
            <div className="border-t border-border pt-4">
              <span className="font-mono text-xs text-muted-foreground" title={job.id}>Job {job.id.slice(0, 8)}</span>
            </div>
            {downloadError && <p role="alert" className="text-sm text-destructive">{downloadError}</p>}
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}
