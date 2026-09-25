'use client';

import * as React from 'react';

import { H3_TAG_CATEGORIES, type H3Tag } from '@/lib/h3Tags';
import { cn } from '@/lib/utils';

export interface H3TagConsoleProps {
  /** Called with the tag to insert into whichever field is currently focused. */
  onInsert: (tag: H3Tag) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * A microexpression "console" for MiniMax-H3 speech tags: click a tag to
 * insert it at the cursor of whichever prompt field was last focused. Kept
 * as plain buttons (not a menu) so every tag stays one click away.
 */
export function H3TagConsole({ onInsert, disabled, className }: H3TagConsoleProps) {
  return (
    <div
      className={cn(
        'flex flex-col gap-1.5 rounded-xl border border-input bg-secondary/40 p-2',
        className,
      )}
    >
      <span className="px-0.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        Speech tags — click to insert at the cursor
      </span>
      <div className="flex flex-col gap-1.5">
        {H3_TAG_CATEGORIES.map((category) => (
          <div key={category.id} className="flex flex-wrap items-center gap-1">
            <span className="mr-0.5 w-full shrink-0 text-[10px] text-muted-foreground/80 sm:w-auto">
              {category.label}
            </span>
            {category.tags.map((tag) => (
              <button
                key={tag.id}
                type="button"
                title={`${tag.description} · e.g. "${tag.example}"`}
                // Keep the target textarea focused (and its selection intact)
                // so the click lands where the user was typing, not nowhere.
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => onInsert(tag)}
                disabled={disabled}
                className="rounded-full border border-input bg-card px-2 py-0.5 font-mono text-[11px] text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
              >
                {tag.label}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
