'use client';

import * as React from 'react';
import { FileAudio, FileVideo, ImageOff } from 'lucide-react';

import type { MediaType } from '@/lib/api';

export interface ReferenceThumbnailProps {
  mediaType: MediaType;
  /** where the image can be loaded from; null when it cannot be previewed */
  src: string | null;
  alt: string;
}

/** A small square preview of a reference: the image itself, or an icon for video, audio, or an unreadable image. */
export function ReferenceThumbnail({ mediaType, src, alt }: ReferenceThumbnailProps) {
  const [failed, setFailed] = React.useState(false);
  React.useEffect(() => setFailed(false), [src]);

  const box = 'flex size-12 shrink-0 items-center justify-center overflow-hidden rounded border border-border bg-muted';
  if (mediaType === 'image' && src && !failed) {
    return (
      <a href={src} target="_blank" rel="noreferrer" className={box} title="Open full size">
        {/* eslint-disable-next-line @next/next/no-img-element -- served by the Studio API, not a static asset */}
        <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} className="size-full object-cover" />
      </a>
    );
  }
  const Icon = mediaType === 'video' ? FileVideo : mediaType === 'audio' ? FileAudio : ImageOff;
  return (
    <span className={box} aria-hidden>
      <Icon className="size-5 text-muted-foreground" />
    </span>
  );
}
