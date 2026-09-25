'use client';

import * as React from 'react';

import { H3ShotPanel } from '@/components/jobs/H3ShotPanel';
import { formatCutTime, resolveShotStarts, type Shot } from '@/lib/h3Shots';
import { cn } from '@/lib/utils';

export interface H3ShotStoryboardProps {
  shots: Shot[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAdd: () => void;
}

/**
 * The shot list as a storyboard: one panel per shot, left to right in
 * playback order. Panels are schematic (no generated imagery exists yet) --
 * subject size reflects shot type and an overlay marks camera movement --
 * so the sequence, who's in frame, and where the dialogue falls read at a
 * glance without scrolling the per-shot editors.
 */
export function H3ShotStoryboard({ shots, selectedId, onSelect, onAdd }: H3ShotStoryboardProps) {
  const starts = React.useMemo(() => resolveShotStarts(shots), [shots]);
  return (
    <div className="flex items-stretch gap-2 overflow-x-auto pb-1" aria-label="Storyboard">
      {shots.map((shot, i) => {
        const selected = shot.id === selectedId;
        return (
          <React.Fragment key={shot.id}>
            {i > 0 && (
              <span className="flex shrink-0 items-center text-muted-foreground/60" aria-hidden>
                ›
              </span>
            )}
            <button
              type="button"
              onClick={() => onSelect(shot.id)}
              aria-current={selected ? 'true' : undefined}
              className={cn(
                'flex w-52 shrink-0 flex-col gap-1.5 rounded-xl border bg-card p-2 text-left transition-colors hover:bg-accent/40',
                selected ? 'border-accent-blue ring-2 ring-accent-blue/50' : 'border-border',
              )}
            >
              <H3ShotPanel shot={shot} number={i + 1} timeLabel={formatCutTime(starts[i])} />
            </button>
          </React.Fragment>
        );
      })}

      <button
        type="button"
        onClick={onAdd}
        className="flex min-h-32 w-28 shrink-0 flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-border text-xs text-muted-foreground transition-colors hover:bg-accent/40 hover:text-foreground"
      >
        <span className="text-lg leading-none">+</span>
        Add shot
      </button>
    </div>
  );
}
