'use client';

import * as React from 'react';

import { H3_PRESETS, H3_TAG_CATEGORIES, type H3Preset, type H3Tag } from '@/lib/h3Tags';
import { cn } from '@/lib/utils';

// Reuses the app's existing status hues as a purely visual category key --
// no status meaning implied, just four distinguishable accents that already
// work in both themes.
export const CATEGORY_STYLE: Record<string, { chip: string; dot: string }> = {
  pace: { chip: 'border-blue-400/35 bg-blue-500/10 text-blue-900 dark:text-blue-100', dot: 'bg-blue-500' },
  delivery: { chip: 'border-amber-400/35 bg-amber-500/10 text-amber-900 dark:text-amber-100', dot: 'bg-amber-500' },
  vocal: { chip: 'border-emerald-400/35 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100', dot: 'bg-emerald-500' },
  body: { chip: 'border-rose-400/35 bg-rose-500/10 text-rose-900 dark:text-rose-100', dot: 'bg-rose-500' },
};

export interface H3TagPaletteProps {
  onInsertTag: (tag: H3Tag) => void;
  onApplyPreset: (preset: H3Preset) => void;
  disabled?: boolean;
  className?: string;
}

/** Full speech-tag palette + emotion presets. Buttons keep the target textarea focused so the click lands at its cursor. */
export function H3TagPalette({ onInsertTag, onApplyPreset, disabled, className }: H3TagPaletteProps) {
  return (
    <div className={className}>
      {H3_TAG_CATEGORIES.map((cat) => {
        const style = CATEGORY_STYLE[cat.id] ?? CATEGORY_STYLE.body;
        return (
          <div key={cat.id} className="mb-4">
            <div className="mb-1.5 flex items-center gap-2 px-0.5">
              <span className={cn('size-2 rounded-full', style.dot)} />
              <span className="text-xs font-semibold text-foreground">{cat.label}</span>
            </div>
            <div className="flex flex-col gap-1">
              {cat.tags.map((tag) => (
                <button
                  key={tag.id}
                  type="button"
                  title={tag.example}
                  disabled={disabled}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => onInsertTag(tag)}
                  className="flex items-baseline justify-between gap-2 rounded-lg border border-input bg-card px-2.5 py-1.5 text-left transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span className="font-mono text-[11.5px] text-foreground">{tag.label}</span>
                  <span className="truncate text-[10.5px] text-muted-foreground">{tag.description}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}

      <div className="mb-1.5 mt-5 px-0.5 text-xs font-semibold text-foreground">Emotion presets</div>
      <div className="flex flex-wrap gap-1.5">
        {H3_PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            disabled={disabled}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onApplyPreset(p)}
            className="rounded-full border border-input bg-card px-2.5 py-1 text-[11.5px] text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
