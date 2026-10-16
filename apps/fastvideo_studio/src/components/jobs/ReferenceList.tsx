'use client';

import * as React from 'react';
import { ArrowDown, ArrowUp, X } from 'lucide-react';

import { getUploadPreviewUrl } from '@/lib/api';
import { labelReferences, type H3Reference } from '@/lib/h3References';

export interface ReferenceListProps {
  references: H3Reference[];
  onMove: (index: number, delta: number) => void;
  onRemove: (id: string) => void;
  disabled?: boolean;
}

function ReferencePreview({ reference }: { reference: H3Reference }) {
  const src = getUploadPreviewUrl(reference.source);
  if (reference.media_type === 'image') {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt=""
        className="size-12 shrink-0 rounded border border-border object-cover"
      />
    );
  }
  if (reference.media_type === 'video') {
    return (
      <video
        src={src}
        muted
        preload="metadata"
        className="size-12 shrink-0 rounded border border-border bg-black object-cover"
        aria-hidden
      />
    );
  }
  return (
    <audio
      src={src}
      controls
      preload="none"
      className="h-8 w-40 shrink-0"
      aria-label={`Play ${reference.fileName}`}
    />
  );
}

/**
 * The ordered reference list. Order is what the model sees, and the labels the
 * prompt refers to (<Picture 1>, <Video 1>, …) are numbered in that order.
 */
export default function ReferenceList({
  references,
  onMove,
  onRemove,
  disabled = false,
}: ReferenceListProps) {
  const labels = labelReferences(references);
  if (references.length === 0) return null;
  return (
    <ol
      aria-label="References in the order the model sees them"
      className="mt-1 flex list-none flex-col gap-1.5 p-0"
    >
      {references.map((reference, index) => (
        <li
          key={reference.id}
          data-testid="reference-item"
          className="flex items-center gap-2 rounded-md border border-border px-2 py-1.5 text-xs text-muted-foreground"
        >
          <span className="w-4 text-right tabular-nums">{index + 1}</span>
          <ReferencePreview reference={reference} />
          <code className="shrink-0 font-mono text-accent-blue">
            {labels[index]}
          </code>
          <span className="min-w-0 truncate" title={reference.source}>
            {reference.fileName}
          </span>
          <span className="ml-auto flex shrink-0 items-center gap-0.5">
            <button
              type="button"
              onClick={() => onMove(index, -1)}
              disabled={disabled || index === 0}
              aria-label={`Move ${labels[index]} up`}
              className="rounded p-1 hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ArrowUp className="size-3.5" aria-hidden />
            </button>
            <button
              type="button"
              onClick={() => onMove(index, 1)}
              disabled={disabled || index === references.length - 1}
              aria-label={`Move ${labels[index]} down`}
              className="rounded p-1 hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ArrowDown className="size-3.5" aria-hidden />
            </button>
            <button
              type="button"
              onClick={() => onRemove(reference.id)}
              disabled={disabled}
              aria-label={`Remove ${labels[index]}`}
              className="rounded p-1 hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-40"
            >
              <X className="size-3.5" aria-hidden />
            </button>
          </span>
        </li>
      ))}
    </ol>
  );
}
